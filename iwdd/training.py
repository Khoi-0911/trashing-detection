"""Training, inference and cross-validation for both model families.

Optimisation follows the paper (Sections 3.3 and 4.2): weighted binary
cross-entropy with ``w+ = 8``, Adam at 1e-3, ReduceLROnPlateau with factor 0.5
and patience 3, gradient clipping at norm 1.0, batch size 4, 20 epochs.

Two details the paper leaves out but which this corpus forces:

* Sequences range from ~14 to ~548 frames, so the loss is masked. Averaging
  over padded positions would let short clips be dominated by padding.
* The loss is averaged over *valid frames across the whole batch* rather than
  per sequence, so a 109s video does not count 38x more than a 2.9s one in a
  way that silently reweights the corpus.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .datasets import (
    SequenceBatch,
    SequenceDataset,
    build_sequence_loader,
    stratified_folds,
)
from .metrics import EvaluationReport, IWDDEvaluator, aggregate_folds
from .models import BaseTemporalModel, build_model, input_kind_of
from .postprocess import DetectionResult, EventDetector
from .utils import ResourceProbe, Timer, get_logger, human_bytes, resolve_device, set_seed

LOGGER = get_logger()


@dataclass
class VideoPrediction:
    """Per-frame scores produced for one video.

    Attributes:
        video_id: Stem of the source video.
        scores: Float array of shape (N,) with per-frame probabilities.
        timestamps: Float array of shape (N,) in seconds.
        onset: Ground-truth onset in seconds; -1 for negatives.
        video_label: Ground-truth video-level label.
    """

    video_id: str
    scores: np.ndarray
    timestamps: np.ndarray
    onset: float
    video_label: int


@dataclass
class FoldResult:
    """Outcome of training and evaluating a single fold.

    Attributes:
        fold: Zero-based fold index.
        report: Contest-rule evaluation on the held-out videos.
        best_epoch: Epoch with the lowest validation loss.
        history: Per-epoch training and validation losses.
        num_parameters: Trainable parameter count of the model.
        checkpoint_path: Where the best weights were written, if saved.
    """

    fold: int
    report: EvaluationReport
    best_epoch: int
    history: list[dict[str, float]] = field(default_factory=list)
    num_parameters: int = 0
    checkpoint_path: str | None = None


class MaskedWeightedBCE(nn.Module):
    """Weighted binary cross-entropy that ignores padded frames.

    Attributes:
        positive_weight: The ``w+`` term applied to positive frames.
        negative_weight: The ``w-`` term applied to negative frames.
    """

    def __init__(self, positive_weight: float = 8.0, negative_weight: float = 1.0) -> None:
        """Initialise the loss.

        Args:
            positive_weight: Weight applied to positive frames.
            negative_weight: Weight applied to negative frames.
        """
        super().__init__()
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the masked weighted loss.

        Args:
            logits: Float tensor of shape (B, T) with raw logits.
            targets: Float tensor of shape (B, T) with binary labels.
            mask: Bool tensor of shape (B, T); True marks a real frame.

        Returns:
            A scalar loss averaged over valid frames.
        """
        per_frame = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        weights = torch.where(
            targets > 0.5,
            torch.full_like(targets, self.positive_weight),
            torch.full_like(targets, self.negative_weight),
        )
        valid = mask.to(per_frame.dtype)
        total = (per_frame * weights * valid).sum()
        denominator = (weights * valid).sum().clamp_min(1.0)
        return total / denominator


class Trainer:
    """Trains one model on one data split and runs inference.

    Attributes:
        config: Full experiment configuration.
        device: Resolved torch device.
        model: The model being trained.
        evaluator: Contest-rule scorer.
        detector: Post-processing trigger.
        probe: Inference throughput and memory statistics.
    """

    def __init__(self, config: ExperimentConfig, model: BaseTemporalModel) -> None:
        """Bind a model to a configuration.

        Args:
            config: Full experiment configuration.
            model: The model to train.
        """
        self.config = config
        self.device = resolve_device("auto")
        self.model = model.to(self.device)
        self.criterion = MaskedWeightedBCE(
            config.train.positive_weight, config.train.negative_weight
        )
        self.evaluator = IWDDEvaluator(config.metric)
        self.detector = EventDetector(config.postprocess)
        self.probe = ResourceProbe()

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> tuple[list[dict[str, float]], int, dict[str, torch.Tensor]]:
        """Train the model and keep the best-scoring weights.

        Args:
            train_loader: Loader over the training videos.
            val_loader: Optional loader used for scheduling and model selection.

        Returns:
            A tuple of the loss history, the best epoch index and the best
            state dictionary.
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.train.learning_rate,
            weight_decay=self.config.train.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.config.train.scheduler_factor,
            patience=self.config.train.scheduler_patience,
        )

        history: list[dict[str, float]] = []
        best_loss = float("inf")
        best_epoch = 0
        best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        stale_epochs = 0

        for epoch in range(self.config.train.epochs):
            train_loss = self._run_epoch(train_loader, optimizer)
            val_loss = self._run_epoch(val_loader, None) if val_loader else train_loss
            scheduler.step(val_loss)

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 5),
                    "val_loss": round(val_loss, 5),
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )
            LOGGER.info(
                "epoch %2d | train %.5f | val %.5f | lr %.2e",
                epoch,
                train_loss,
                val_loss,
                optimizer.param_groups[0]["lr"],
            )

            if val_loss < best_loss - 1e-6:
                best_loss = val_loss
                best_epoch = epoch
                best_state = {
                    k: v.detach().clone() for k, v in self.model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                patience = self.config.train.early_stopping_patience
                if patience and stale_epochs >= patience:
                    LOGGER.info("Early stopping at epoch %d", epoch)
                    break

        self.model.load_state_dict(best_state)
        return history, best_epoch, best_state

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> list[VideoPrediction]:
        """Produce per-frame probability curves for every video in a loader.

        Args:
            loader: Loader over the videos to score.

        Returns:
            One prediction record per video.
        """
        self.model.eval()
        predictions: list[VideoPrediction] = []

        for batch in loader:
            batch = batch.to(self.device)
            with Timer("inference") as timer:
                logits = self.model(batch.features, batch.mask)
                scores = torch.sigmoid(logits)
            self.probe.add(frames=int(batch.mask.sum().item()), seconds=timer.elapsed)

            scores_np = scores.detach().cpu().numpy()
            mask_np = batch.mask.detach().cpu().numpy()
            times_np = batch.timestamps.detach().cpu().numpy()

            for row, video_id in enumerate(batch.video_ids):
                valid = mask_np[row]
                predictions.append(
                    VideoPrediction(
                        video_id=video_id,
                        scores=scores_np[row][valid],
                        timestamps=times_np[row][valid],
                        onset=float(batch.onsets[row].item()),
                        video_label=int(batch.video_labels[row].item()),
                    )
                )

        self.probe.refresh_peak_memory()
        return predictions

    def evaluate(self, loader: DataLoader) -> tuple[EvaluationReport, list[DetectionResult]]:
        """Score a loader end to end: predict, post-process, then match.

        Args:
            loader: Loader over the videos to score.

        Returns:
            A tuple of the evaluation report and the raw detection results.
        """
        predictions = self.predict(loader)
        results = [
            self.detector.detect(item.scores, item.timestamps) for item in predictions
        ]
        report = self.evaluator.evaluate(
            results,
            [item.onset for item in predictions],
            [item.video_id for item in predictions],
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
                batch = batch.to(self.device)
                logits = self.model(batch.features, batch.mask)
                loss = self.criterion(logits, batch.labels, batch.mask)

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


class CrossValidator:
    """Runs stratified k-fold cross-validation over an embedding dataset.

    Attributes:
        config: Full experiment configuration.
        dataset: The videos to cross-validate over.
    """

    def __init__(self, config: ExperimentConfig, dataset: SequenceDataset) -> None:
        """Bind a dataset to a configuration.

        Args:
            config: Full experiment configuration.
            dataset: Videos to cross-validate over.
        """
        self.config = config
        self.dataset = dataset

    def run(self, save_checkpoints: bool = True) -> tuple[list[FoldResult], dict[str, float]]:
        """Train and evaluate one model per fold.

        Args:
            save_checkpoints: Write the best weights of each fold to disk.

        Returns:
            A tuple of the per-fold results and the aggregated metrics.
        """
        folds = stratified_folds(
            self.dataset.video_labels,
            self.config.train.num_folds,
            self.config.train.seed,
        )
        output_dir = Path(self.config.paths.output_dir) / self.config.run_name
        output_dir.mkdir(parents=True, exist_ok=True)

        results: list[FoldResult] = []
        for fold, (train_idx, val_idx) in enumerate(folds):
            LOGGER.info(
                "=== fold %d/%d | train %d | val %d ===",
                fold + 1,
                len(folds),
                len(train_idx),
                len(val_idx),
            )
            set_seed(self.config.train.seed + fold)

            train_loader = build_sequence_loader(
                self.dataset.subset(train_idx),
                self.config.train.batch_size,
                shuffle=True,
                num_workers=self.config.train.num_workers,
            )
            val_loader = build_sequence_loader(
                self.dataset.subset(val_idx),
                self.config.train.batch_size,
                shuffle=False,
                num_workers=self.config.train.num_workers,
            )

            model = build_model(self.config.model, input_dim=self.dataset.feature_dim)
            trainer = Trainer(self.config, model)
            history, best_epoch, best_state = trainer.fit(train_loader, val_loader)
            report, _ = trainer.evaluate(val_loader)

            checkpoint_path = None
            if save_checkpoints:
                checkpoint_path = str(output_dir / f"fold{fold}.pt")
                torch.save(best_state, checkpoint_path)
                size = Path(checkpoint_path).stat().st_size
                LOGGER.info("Saved %s (%s)", checkpoint_path, human_bytes(size))

            LOGGER.info(
                "fold %d | P %.4f | R %.4f | F1 %.4f | delay %.2fs",
                fold,
                report.precision,
                report.recall,
                report.f1,
                report.mean_delay,
            )
            results.append(
                FoldResult(
                    fold=fold,
                    report=report,
                    best_epoch=best_epoch,
                    history=history,
                    num_parameters=model.num_parameters,
                    checkpoint_path=checkpoint_path,
                )
            )

        aggregated = aggregate_folds([r.report for r in results], self.config.metric)
        aggregated["num_parameters"] = results[0].num_parameters if results else 0
        LOGGER.info(
            "CV done | F1 %.4f +/- %.4f | P %.4f | R %.4f",
            aggregated.get("f1_mean", 0.0),
            aggregated.get("f1_std", 0.0),
            aggregated.get("precision_mean", 0.0),
            aggregated.get("recall_mean", 0.0),
        )
        return results, aggregated


class BenchmarkRunner:
    """Sweeps the architecture axis against a fixed feature cache.

    Reproduces one row of paper Table 1: with the captions already cached, each
    architecture costs only minutes, so the full six-way comparison is cheap.

    Attributes:
        config: Base configuration; the architecture name is overridden per run.
        dataset: The videos to evaluate on.
    """

    def __init__(self, config: ExperimentConfig, dataset: SequenceDataset) -> None:
        """Bind a dataset to a base configuration.

        Args:
            config: Base experiment configuration.
            dataset: Videos to evaluate on.
        """
        self.config = config
        self.dataset = dataset

    def run(self, architectures: Sequence[str]) -> dict[str, dict[str, float]]:
        """Cross-validate every named architecture.

        Args:
            architectures: Registry keys to evaluate.

        Returns:
            A mapping from architecture name to its aggregated metrics.
        """
        from dataclasses import replace

        table: dict[str, dict[str, float]] = {}
        for name in architectures:
            if input_kind_of(name) != "sequence":
                LOGGER.warning("Skipping %s: not a sequence model", name)
                continue

            LOGGER.info("########## %s ##########", name)
            config = replace(
                self.config,
                model=replace(self.config.model, name=name),
                run_name=f"{self.config.run_name}/{name}",
            )
            _, aggregated = CrossValidator(config, self.dataset).run(save_checkpoints=False)
            table[name] = aggregated

        self._write_table(table)
        return table

    def _write_table(self, table: dict[str, dict[str, float]]) -> None:
        """Persist the comparison table as JSON and Markdown.

        Args:
            table: Mapping from architecture name to aggregated metrics.
        """
        output_dir = Path(self.config.paths.output_dir) / self.config.run_name
        output_dir.mkdir(parents=True, exist_ok=True)

        (output_dir / "benchmark.json").write_text(
            json.dumps(table, indent=2), encoding="utf-8"
        )

        lines = [
            "| Architecture | Precision | Recall | F1 | F1 std | Delay (s) | Params |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, metrics in sorted(table.items(), key=lambda kv: -kv[1].get("f1_mean", 0)):
            lines.append(
                "| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.2f} | {:,} |".format(
                    name,
                    metrics.get("precision_mean", 0.0),
                    metrics.get("recall_mean", 0.0),
                    metrics.get("f1_mean", 0.0),
                    metrics.get("f1_std", 0.0),
                    metrics.get("mean_delay_s_mean", 0.0),
                    int(metrics.get("num_parameters", 0)),
                )
            )
        (output_dir / "benchmark.md").write_text("\n".join(lines), encoding="utf-8")
        LOGGER.info("Wrote benchmark table to %s", output_dir / "benchmark.md")
