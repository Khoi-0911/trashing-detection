"""IWDD: Illegal Waste Dumping Detection on the MIVIA-IWDD-500 benchmark.

Reimplementation of the vision-language + temporal-classifier pipeline from
Kairanbay, M. "Vision-Language Temporal Analysis for Illegal Waste Dumping
Detection in Surveillance Videos" (WACV 2026 Workshops, pp. 559-566).

Pipeline stages:
    1. Frame sampling at a fixed rate (5 FPS by default).
    2. Multi-aspect captioning with a frozen VLM (5 focused prompts per frame).
    3. Sentence embedding of the captions (Sentence-BERT, 384-d).
    4. Temporal classification (CNN+GRU and variants) producing per-frame scores.
    5. Post-processing (moving average + dual threshold) into a single event onset.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
