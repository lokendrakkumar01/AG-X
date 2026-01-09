"""
AG-X 2026 Experiment Management
================================

Reproducible experiment tracking, versioning, and report generation.
"""

from agx.experiments.manager import ExperimentManager, Experiment
from agx.experiments.comparison import ExperimentComparison, StatisticalTest
from agx.experiments.reports import ReportGenerator, Report

__all__ = [
    "ExperimentManager",
    "Experiment",
    "ExperimentComparison",
    "StatisticalTest",
    "ReportGenerator",
    "Report",
]
