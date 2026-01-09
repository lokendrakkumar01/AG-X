"""
Experiment Comparison
======================

Statistical comparison and validation of experiment results.
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from scipy import stats


@dataclass
class StatisticalTest:
    """Result of a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    interpretation: str = ""


class ExperimentComparison:
    """
    Compare experiment results with statistical validation.
    
    Provides t-tests, ANOVA, effect size calculations, and confidence intervals.
    """
    
    def __init__(self, significance_level: float = 0.05):
        self.alpha = significance_level
    
    def t_test(self, 
               group1: np.ndarray, 
               group2: np.ndarray,
               paired: bool = False) -> StatisticalTest:
        """
        Perform t-test between two groups.
        
        Args:
            group1, group2: Data arrays
            paired: Whether to use paired t-test
        """
        if paired:
            stat, p = stats.ttest_rel(group1, group2)
            test_name = "Paired t-test"
        else:
            stat, p = stats.ttest_ind(group1, group2)
            test_name = "Independent t-test"
        
        # Cohen's d effect size
        pooled_std = np.sqrt((np.var(group1) + np.var(group2)) / 2)
        effect_size = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0
        
        # Interpretation
        if abs(effect_size) < 0.2:
            effect_interp = "negligible"
        elif abs(effect_size) < 0.5:
            effect_interp = "small"
        elif abs(effect_size) < 0.8:
            effect_interp = "medium"
        else:
            effect_interp = "large"
        
        interp = f"{'Significant' if p < self.alpha else 'Not significant'} difference with {effect_interp} effect size."
        
        return StatisticalTest(
            test_name=test_name,
            statistic=stat,
            p_value=p,
            is_significant=p < self.alpha,
            effect_size=effect_size,
            interpretation=interp,
        )
    
    def anova(self, *groups: np.ndarray) -> StatisticalTest:
        """Perform one-way ANOVA across multiple groups."""
        stat, p = stats.f_oneway(*groups)
        
        # Eta-squared effect size
        all_data = np.concatenate(groups)
        grand_mean = np.mean(all_data)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
        ss_total = np.sum((all_data - grand_mean)**2)
        eta_sq = ss_between / ss_total if ss_total > 0 else 0
        
        return StatisticalTest(
            test_name="One-way ANOVA",
            statistic=stat,
            p_value=p,
            is_significant=p < self.alpha,
            effect_size=eta_sq,
            interpretation=f"{'Significant' if p < self.alpha else 'No significant'} difference between groups (η² = {eta_sq:.3f}).",
        )
    
    def confidence_interval(self, 
                           data: np.ndarray,
                           confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for mean."""
        n = len(data)
        mean = np.mean(data)
        se = stats.sem(data)
        
        t_crit = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin = t_crit * se
        
        return (mean - margin, mean + margin)
    
    def compare_experiments(self,
                           exp1_metrics: Dict[str, np.ndarray],
                           exp2_metrics: Dict[str, np.ndarray]) -> Dict[str, StatisticalTest]:
        """Compare metrics between two experiments."""
        results = {}
        
        common_metrics = set(exp1_metrics.keys()) & set(exp2_metrics.keys())
        
        for metric in common_metrics:
            data1 = np.asarray(exp1_metrics[metric])
            data2 = np.asarray(exp2_metrics[metric])
            
            results[metric] = self.t_test(data1, data2)
        
        return results
    
    def bootstrap_confidence(self,
                            data: np.ndarray,
                            statistic_fn=np.mean,
                            n_bootstrap: int = 1000,
                            confidence: float = 0.95) -> Tuple[float, float]:
        """Bootstrap confidence interval for any statistic."""
        n = len(data)
        stats_list = []
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            stats_list.append(statistic_fn(sample))
        
        lower = np.percentile(stats_list, (1 - confidence) / 2 * 100)
        upper = np.percentile(stats_list, (1 + confidence) / 2 * 100)
        
        return (lower, upper)
    
    def summary_statistics(self, data: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive summary statistics."""
        return {
            "mean": np.mean(data),
            "std": np.std(data),
            "median": np.median(data),
            "min": np.min(data),
            "max": np.max(data),
            "q25": np.percentile(data, 25),
            "q75": np.percentile(data, 75),
            "iqr": np.percentile(data, 75) - np.percentile(data, 25),
            "skewness": stats.skew(data),
            "kurtosis": stats.kurtosis(data),
        }
    
    def generate_comparison_report(self,
                                   comparisons: Dict[str, StatisticalTest]) -> str:
        """Generate markdown comparison report."""
        lines = ["# Experiment Comparison Report\n"]
        
        for metric, test in comparisons.items():
            lines.append(f"## {metric}\n")
            lines.append(f"- **Test:** {test.test_name}")
            lines.append(f"- **Statistic:** {test.statistic:.4f}")
            lines.append(f"- **p-value:** {test.p_value:.4f}")
            lines.append(f"- **Significant:** {'Yes' if test.is_significant else 'No'}")
            if test.effect_size is not None:
                lines.append(f"- **Effect Size:** {test.effect_size:.4f}")
            lines.append(f"- **Interpretation:** {test.interpretation}\n")
        
        return "\n".join(lines)
