#!/usr/bin/env python3
"""Public model API for Q-Spy fewshot.py novelty.py baselines.py consumed by the experiment runner."""

from models.base import (
    BaseBudgetedModel,
    BaseFewShotBaseline,
    BaseFewShotModel,
    BaseNoveltyBaseline,
    BaseNoveltyModel,
)
from models.baselines import (
    BASELINE_REGISTRY,
    HistGradientBoostingBaseline,
    IsolationForestBaseline,
    KNNBaseline,
    LOFBaseline,
    LaplacianSVMBaseline,
    LogisticRegressionBaseline,
    RBFOCSVMBaseline,
    RBFSVMBaseline,
    build_baseline,
)
from models.fewshot import FewShotSVC, assert_equal_search_budget, evaluate_fewshot
from models.novelty import NoveltyOCSVM, evaluate_novelty, projected_quantum_kernel_matrix

__all__ = [
    "BASELINE_REGISTRY",
    "BaseBudgetedModel",
    "BaseFewShotBaseline",
    "BaseFewShotModel",
    "BaseNoveltyBaseline",
    "BaseNoveltyModel",
    "FewShotSVC",
    "HistGradientBoostingBaseline",
    "IsolationForestBaseline",
    "KNNBaseline",
    "LOFBaseline",
    "LaplacianSVMBaseline",
    "LogisticRegressionBaseline",
    "NoveltyOCSVM",
    "RBFOCSVMBaseline",
    "RBFSVMBaseline",
    "assert_equal_search_budget",
    "build_baseline",
    "evaluate_fewshot",
    "evaluate_novelty",
    "projected_quantum_kernel_matrix",
]