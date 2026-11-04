"""Training and inference for the 3D CNN baselines.

Kept apart from :mod:`iwdd.training` because the data contract differs: clip
models see raw pixels and emit one logit per window, whereas the sequence
models see cached embeddings and emit one logit per frame.

To score a clip model with the same contest metric as everything else, the
per-clip logits of a video are scattered back onto a time axis -- each clip
contributes its score to the frames it covers, and overlapping clips are
averaged. The result is a per-frame curve that feeds the identical
:class:`~iwdd.postprocess.EventDetector`, so the comparison stays apples to
apples.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .annotations import AnnotationStore
from .config import ExperimentConfig
from .datasets import ClipDataset, build_clip_loader
from .labeling import FrameLabeler
from .metrics import EvaluationReport, IWDDEvaluator
from .models import BaseClipModel, build_model
from .postprocess import DetectionResult, EventDetector
from .training import VideoPrediction
from .utils import ResourceProbe, Timer, get_logger, resolve_device

LOGGER = get_logger()


class ClipTrainer:
    """Trains a 3D CNN on raw clips and scores it with the contest metric.

    Attributes:
        config: Full experiment configuration.
        device: Resolved torch device.
        model: The clip model being trained.
        evaluator: Contest-rule scorer.
        detector: Post-processing trigger.
        probe: Inference throughput and memory statistics.
    """

    def __init__(self, config: ExperimentConfig, model: BaseClipModel) -> None:
        """Bind a clip model to a configuration.

        Args:
            config: Full experiment configuration.
            model: The clip model to train.
        """
        self.config = config
        self.device = resolve_device("auto")
        self.model = model.to(self.device)
        self.evaluator = IWDDEvaluator(config.metric)
        self.detector = EventDetector(config.postprocess)
        self.probe = ResourceProbe()

        weight = torch.tensor([config.train.positive_weight], device=self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=weight)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None = None) -> list[dict]:
        """Train the model, keeping the weights with the best validation loss.

        Args:
            train_loader: Loader over training clips.
            val_loader: Optional loader over validation clips.

        Returns:
            The per-epoch loss history.
        """
        optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.train.learning_rate,
            weight_decay=self.config.train.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.config.train.scheduler_factor,
            patience=self.config.train.scheduler_patience,
        )

        history: list[dict] = []
        best_loss = float("inf")
        best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

        for epoch in range(self.config.train.epochs):
            train_loss = self._run_epoch(train_loader, optimizer)
            val_loss = self._run_epoch(val_loader, None) if val_loader else train_loss
            scheduler.step(val_loss)

            history.append(
                {"epoch": epoch, "train_loss": round(train_loss, 5), "val_loss": round(val_loss, 5)}
            )
            LOGGER.info("epoch %2d | train %.5f | val %.5f", epoch, train_loss, val_loss)

            if val_loss < best_loss - 1e-6:
                best_loss = val_loss
                best_state = {
                    k: v.detach().clone() for k, v in self.model.state_dict().items()
                }

        self.model.load_state_dict(best_state)
        return history

    @torch.no_grad()
    def predict(self, loader: DataLoader, store: AnnotationStore) -> list[VideoPrediction]:
        """Score every clip and reassemble per-video score curves.

        Args:
            loader: Loader over the clips to score; must not be shuffled.
            store: Annotations supplying ground-truth onsets and labels.

        Returns:
            One prediction record per video that produced at least one clip.
        """
        self.model.eval()
        buckets: dict[str, list[tuple[float, float]]] = defaultdict(list)

        for batch in loader:
            clips = batch["clip"].to(self.device)
            with Timer("inference") as timer:
                logits = self.model(clips)
                scores = torch.sigmoid(logits)
            self.probe.add(frames=clips.shape[0] * clips.shape[2], seconds=timer.elapsed)

            starts = batch["start_time"].numpy()
            ends = batch["end_time"].numpy()
            values = scores.detach().cpu().numpy()
            for video_id, start, end, score in zip(
                batch["video_id"], starts, ends, values
            ):
                centre = float((start + end) / 2.0)
                buckets[str(video_id)].append((centre, float(score)))

        self.probe.refresh_peak_memory()

        predictions: list[VideoPrediction] = []
        for video_id, points in buckets.items():
            points.sort(key=lambda item: item[0])
            timestamps = np.asarray([p[0] for p in points], dtype=np.float32)
            scores = np.asarray([p[1] for p in points], dtype=np.float32)
            annotation = store.get(video_id)
            predictions.append(
                VideoPrediction(
                    video_id=video_id,
                    scores=scores,
                    timestamps=timestamps,
                    onset=float(annotation.onset) if annotation.onset is not None else -1.0,
                    video_label=int(annotation.is_positive),
                )
            )
        return predictions

    def evaluate(
        self,
        loader: DataLoader,
        store: AnnotationStore,
    ) -> tuple[EvaluationReport, list[DetectionResult]]:
        """Score a clip loader end to end.

        Args:
            loader: Loader over the clips to score.
            store: Annotations supplying ground truth.

        Returns:
            A tuple of the evaluation report and the raw detection results.
        """
        predictions = self.predict(loader, store)
        results = [self.detector.detect(p.scores, p.timestamps) for p in predictions]
        report = self.evaluator.evaluate(
            results,
            [p.onset for p in predictions],
            [p.video_id for p in predictions],
        )
        return report, results

    def _run_epoch(self, loader: DataLoader | None, optimizer) -> float:
        """Run one epoch in training or evaluation mode.

        Args:
            loader: Loader to iterate; None returns 0.0.
            optimizer: Optimiser to step, or None for evaluation.

        Returns:
            The mean loss over the epoch.
        """
        if loader is None:
            return 0.0

        training = optimizer is not None
        self.model.train(training)
        total_loss = 0.0
        num_batches = 0

        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch in loader:
                clips = batch["clip"].to(self.device)
                targets = batch["label"].to(self.device)
                logits = self.model(clips)
                loss = self.criterion(logits, targets)

                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.train.grad_clip
                    )
                    optimizer.step()

                total_loss += float(loss.item())
                num_batches += 1

        return total_loss / max(num_batches, 1)


def run_clip_experiment(
    config: ExperimentConfig,
    train_store: AnnotationStore,
    test_store: AnnotationStore,
) -> EvaluationReport:
    """Train a clip model on one split and evaluate it on another.

    Args:
        config: Full experiment configuration naming a clip architecture.
        train_store: Videos to train on.
        test_store: Held-out videos to evaluate on.

    Returns:
        The contest-rule evaluation report on the held-out split.
    """
    labeler = FrameLabeler(config.labeling)

    train_dataset = ClipDataset(list(train_store), config.sampling, config.model, labeler)
    test_dataset = ClipDataset(list(test_store), config.sampling, config.model, labeler)

    train_loader = build_clip_loader(
        train_dataset,
        config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
    )
    test_loader = build_clip_loader(
        test_dataset,
        config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
    )

    model = build_model(config.model)
    if not isinstance(model, BaseClipModel):
        raise TypeError(f"{config.model.name!r} is not a clip model")

    trainer = ClipTrainer(config, model)
    trainer.fit(train_loader)
    report, _ = trainer.evaluate(test_loader, test_store)

    output_dir = Path(config.paths.output_dir) / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "clip_model.pt")

    LOGGER.info(
        "clip model %s | P %.4f | R %.4f | F1 %.4f",
        config.model.name,
        report.precision,
        report.recall,
        report.f1,
    )
    return report
