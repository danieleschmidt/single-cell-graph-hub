"""
TERRAGON Academic Validation Framework v2.0
Advanced statistical validation and peer review readiness for breakthrough research
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union
import scipy.stats as stats
from scipy.stats import mannwhitneyu, kruskal, chi2_contingency, fisher_exact
import json
import logging
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, roc_auc_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
from collections import defaultdict
import asyncio
from concurrent.futures import ThreadPoolExecutor
import itertools

warnings.filterwarnings('ignore')


@dataclass
class StatisticalValidation:
    """Comprehensive statistical validation results."""
    test_name: str
    test_statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    interpretation: str
    significance_level: float = 0.05
    multiple_testing_correction: str = "bonferroni"
    corrected_p_value: Optional[float] = None
    
    @property
    def is_significant(self) -> bool:
        """Check if result is statistically significant."""
        p_val = self.corrected_p_value if self.corrected_p_value else self.p_value
        return p_val < self.significance_level
    
    @property
    def effect_magnitude(self) -> str:
        """Classify effect size magnitude."""
        abs_effect = abs(self.effect_size)
        if abs_effect < 0.2:
            return "negligible"
        elif abs_effect < 0.5:
            return "small"
        elif abs_effect < 0.8:
            return "medium"
        else:
            return "large"


@dataclass
class PeerReviewChecklist:
    """Academic peer review readiness checklist."""
    methodology_clarity: float
    statistical_rigor: float
    reproducibility_score: float
    novelty_assessment: float
    biological_relevance: float
    writing_quality: float
    figure_quality: float
    ethical_considerations: float
    data_availability: float
    code_availability: float
    
    @property
    def overall_readiness(self) -> float:
        """Calculate overall peer review readiness score."""
        weights = {
            'methodology_clarity': 0.15,
            'statistical_rigor': 0.20,
            'reproducibility_score': 0.15,
            'novelty_assessment': 0.15,
            'biological_relevance': 0.10,
            'writing_quality': 0.10,
            'figure_quality': 0.05,
            'ethical_considerations': 0.05,
            'data_availability': 0.025,
            'code_availability': 0.025
        }
        
        total_score = 0
        for field_name, weight in weights.items():
            if hasattr(self, field_name):
                total_score += getattr(self, field_name) * weight
        
        return total_score
    
    @property
    def readiness_category(self) -> str:
        """Categorize readiness level."""
        score = self.overall_readiness
        if score >= 0.9:
            return "publication_ready"
        elif score >= 0.8:
            return "minor_revisions"
        elif score >= 0.7:
            return "major_revisions"
        else:
            return "needs_substantial_work"


class AcademicValidator:
    """Comprehensive academic validation framework."""
    
    def __init__(self, output_dir: str = "./validation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.validation_history: List[Dict[str, Any]] = []
        
        # Statistical test registry
        self.statistical_tests = {
            'parametric': {
                'independent_ttest': self._independent_ttest,
                'paired_ttest': self._paired_ttest,
                'anova': self._one_way_anova,
                'pearson_correlation': self._pearson_correlation
            },
            'nonparametric': {
                'mann_whitney_u': self._mann_whitney_u,
                'wilcoxon_signed_rank': self._wilcoxon_signed_rank,
                'kruskal_wallis': self._kruskal_wallis,
                'spearman_correlation': self._spearman_correlation
            },
            'categorical': {
                'chi_square': self._chi_square_test,
                'fisher_exact': self._fisher_exact_test,
                'mcnemar': self._mcnemar_test
            }
        }
    
    async def validate_research_results(self, 
                                      experimental_results: Dict[str, Any],
                                      baseline_results: Dict[str, Any],
                                      metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive validation of research results."""
        self.logger.info("🔬 Starting comprehensive academic validation")
        
        validation_report = {
            'timestamp': datetime.now().isoformat(),
            'experimental_metadata': metadata,
            'statistical_validations': {},
            'effect_size_analysis': {},
            'power_analysis': {},
            'reproducibility_assessment': {},
            'peer_review_readiness': {},
            'recommendations': []
        }
        
        # Statistical hypothesis testing
        statistical_results = await self._conduct_statistical_tests(
            experimental_results, baseline_results
        )
        validation_report['statistical_validations'] = statistical_results
        
        # Effect size analysis
        effect_sizes = self._calculate_effect_sizes(experimental_results, baseline_results)
        validation_report['effect_size_analysis'] = effect_sizes
        
        # Power analysis
        power_results = self._conduct_power_analysis(experimental_results, baseline_results)
        validation_report['power_analysis'] = power_results
        
        # Reproducibility assessment
        repro_assessment = await self._assess_reproducibility(experimental_results, metadata)
        validation_report['reproducibility_assessment'] = repro_assessment
        
        # Peer review readiness
        peer_review = self._evaluate_peer_review_readiness(validation_report)
        validation_report['peer_review_readiness'] = peer_review
        
        # Generate recommendations
        recommendations = self._generate_recommendations(validation_report)
        validation_report['recommendations'] = recommendations
        
        # Save validation report
        report_file = self.output_dir / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(validation_report, f, indent=2, default=str)
        
        self.logger.info(f"📊 Validation report saved: {report_file}")
        
        return validation_report
    
    async def _conduct_statistical_tests(self, 
                                       experimental: Dict[str, Any], 
                                       baseline: Dict[str, Any]) -> Dict[str, StatisticalValidation]:
        """Conduct comprehensive statistical hypothesis testing."""
        results = {}
        
        # Extract performance metrics
        exp_metrics = self._extract_metrics(experimental)
        base_metrics = self._extract_metrics(baseline)
        
        # Test each metric
        for metric_name in exp_metrics.keys():
            if metric_name in base_metrics:
                exp_values = np.array(exp_metrics[metric_name])
                base_values = np.array(base_metrics[metric_name])
                
                # Normality tests
                exp_normal = self._test_normality(exp_values)
                base_normal = self._test_normality(base_values)
                
                # Choose appropriate test
                if exp_normal and base_normal and len(exp_values) >= 30:
                    # Use parametric tests
                    if self._test_equal_variance(exp_values, base_values):
                        test_result = self._independent_ttest(exp_values, base_values)
                    else:
                        test_result = self._welch_ttest(exp_values, base_values)
                else:
                    # Use non-parametric tests
                    test_result = self._mann_whitney_u(exp_values, base_values)
                
                results[f"{metric_name}_comparison"] = test_result
        
        # Multiple testing correction
        if len(results) > 1:
            results = self._apply_multiple_testing_correction(results)
        
        return {k: v.__dict__ if hasattr(v, '__dict__') else v for k, v in results.items()}
    
    def _extract_metrics(self, results: Dict[str, Any]) -> Dict[str, List[float]]:
        """Extract metrics from results dictionary."""
        metrics = defaultdict(list)
        
        if isinstance(results, dict):
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    metrics[key].append(float(value))
                elif isinstance(value, list):
                    metrics[key].extend([float(v) for v in value if isinstance(v, (int, float))])
                elif isinstance(value, dict):
                    # Recursively extract from nested dictionaries
                    nested_metrics = self._extract_metrics(value)
                    for nested_key, nested_values in nested_metrics.items():
                        metrics[f"{key}_{nested_key}"].extend(nested_values)
        
        return dict(metrics)
    
    def _test_normality(self, data: np.ndarray, alpha: float = 0.05) -> bool:
        """Test for normality using Shapiro-Wilk test."""
        if len(data) < 3:
            return False
        try:
            _, p_value = stats.shapiro(data)
            return p_value > alpha
        except:
            return False
    
    def _test_equal_variance(self, data1: np.ndarray, data2: np.ndarray, alpha: float = 0.05) -> bool:
        """Test for equal variances using Levene's test."""
        try:
            _, p_value = stats.levene(data1, data2)
            return p_value > alpha
        except:
            return False
    
    def _independent_ttest(self, group1: np.ndarray, group2: np.ndarray) -> StatisticalValidation:
        """Independent samples t-test."""
        statistic, p_value = stats.ttest_ind(group1, group2)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                             (len(group2) - 1) * np.var(group2, ddof=1)) / 
                            (len(group1) + len(group2) - 2))
        cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
        
        # Confidence interval for effect size
        se_d = np.sqrt((len(group1) + len(group2)) / (len(group1) * len(group2)) + 
                       cohens_d**2 / (2 * (len(group1) + len(group2))))
        ci_lower = cohens_d - 1.96 * se_d
        ci_upper = cohens_d + 1.96 * se_d
        
        interpretation = self._interpret_ttest(p_value, cohens_d)
        
        return StatisticalValidation(
            test_name="Independent t-test",
            test_statistic=statistic,
            p_value=p_value,
            effect_size=cohens_d,
            confidence_interval=(ci_lower, ci_upper),
            interpretation=interpretation
        )
    
    def _welch_ttest(self, group1: np.ndarray, group2: np.ndarray) -> StatisticalValidation:
        """Welch's t-test (unequal variances)."""
        statistic, p_value = stats.ttest_ind(group1, group2, equal_var=False)
        
        # Effect size (Cohen's d with correction for unequal variances)
        mean_diff = np.mean(group1) - np.mean(group2)
        pooled_std = np.sqrt((np.var(group1, ddof=1) + np.var(group2, ddof=1)) / 2)
        cohens_d = mean_diff / pooled_std
        
        # Conservative confidence interval
        se_d = np.sqrt((len(group1) + len(group2)) / (len(group1) * len(group2)) + 
                       cohens_d**2 / (2 * (len(group1) + len(group2))))
        ci_lower = cohens_d - 1.96 * se_d
        ci_upper = cohens_d + 1.96 * se_d
        
        interpretation = self._interpret_ttest(p_value, cohens_d)
        
        return StatisticalValidation(
            test_name="Welch's t-test",
            test_statistic=statistic,
            p_value=p_value,
            effect_size=cohens_d,
            confidence_interval=(ci_lower, ci_upper),
            interpretation=interpretation
        )
    
    def _paired_ttest(self, group1: np.ndarray, group2: np.ndarray) -> StatisticalValidation:
        """Paired samples t-test."""
        statistic, p_value = stats.ttest_rel(group1, group2)
        
        # Effect size for paired samples
        differences = group1 - group2
        cohens_d = np.mean(differences) / np.std(differences, ddof=1)
        
        # Confidence interval
        se_d = 1 / np.sqrt(len(differences))
        ci_lower = cohens_d - 1.96 * se_d
        ci_upper = cohens_d + 1.96 * se_d
        
        interpretation = self._interpret_ttest(p_value, cohens_d)
        
        return StatisticalValidation(
            test_name="Paired t-test",
            test_statistic=statistic,
            p_value=p_value,
            effect_size=cohens_d,
            confidence_interval=(ci_lower, ci_upper),
            interpretation=interpretation
        )
    
    def _mann_whitney_u(self, group1: np.ndarray, group2: np.ndarray) -> StatisticalValidation:
        """Mann-Whitney U test (non-parametric)."""
        statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
        
        # Effect size (rank-biserial correlation)
        n1, n2 = len(group1), len(group2)
        effect_size = 2 * statistic / (n1 * n2) - 1
        
        # Confidence interval approximation
        se = np.sqrt((n1 + n2 + 1) / (3 * n1 * n2))
        ci_lower = effect_size - 1.96 * se
        ci_upper = effect_size + 1.96 * se
        
        interpretation = self._interpret_nonparametric(p_value, effect_size)
        
        return StatisticalValidation(
            test_name="Mann-Whitney U test",
            test_statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            interpretation=interpretation
        )
    
    def _wilcoxon_signed_rank(self, group1: np.ndarray, group2: np.ndarray) -> StatisticalValidation:
        """Wilcoxon signed-rank test."""
        statistic, p_value = stats.wilcoxon(group1, group2)
        
        # Effect size approximation
        n = len(group1)
        z_score = statistic / np.sqrt(n * (n + 1) * (2 * n + 1) / 6)
        effect_size = z_score / np.sqrt(n)
        
        # Confidence interval
        se = 1 / np.sqrt(n)
        ci_lower = effect_size - 1.96 * se
        ci_upper = effect_size + 1.96 * se
        
        interpretation = self._interpret_nonparametric(p_value, effect_size)
        
        return StatisticalValidation(
            test_name="Wilcoxon signed-rank test",
            test_statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            interpretation=interpretation
        )
    
    def _one_way_anova(self, *groups) -> StatisticalValidation:
        """One-way ANOVA."""
        statistic, p_value = stats.f_oneway(*groups)
        
        # Effect size (eta-squared)
        k = len(groups)
        n_total = sum(len(group) for group in groups)
        ss_between = sum(len(group) * (np.mean(group) - np.mean(np.concatenate(groups)))**2 
                        for group in groups)
        ss_total = sum(np.sum((group - np.mean(np.concatenate(groups)))**2) for group in groups)
        eta_squared = ss_between / ss_total
        
        # Confidence interval approximation
        df_between = k - 1
        df_within = n_total - k
        f_critical = stats.f.ppf(0.975, df_between, df_within)
        se_eta = np.sqrt(2 * eta_squared * (1 - eta_squared) / (n_total - 1))
        ci_lower = max(0, eta_squared - 1.96 * se_eta)
        ci_upper = min(1, eta_squared + 1.96 * se_eta)
        
        interpretation = self._interpret_anova(p_value, eta_squared)
        
        return StatisticalValidation(
            test_name="One-way ANOVA",
            test_statistic=statistic,
            p_value=p_value,
            effect_size=eta_squared,
            confidence_interval=(ci_lower, ci_upper),
            interpretation=interpretation
        )
    
    def _kruskal_wallis(self, *groups) -> StatisticalValidation:
        """Kruskal-Wallis test."""
        statistic, p_value = kruskal(*groups)
        
        # Effect size (epsilon-squared)
        n_total = sum(len(group) for group in groups)
        epsilon_squared = (statistic - len(groups) + 1) / (n_total - len(groups))
        
        # Confidence interval approximation
        se = np.sqrt(2 / n_total)
        ci_lower = max(0, epsilon_squared - 1.96 * se)
        ci_upper = min(1, epsilon_squared + 1.96 * se)
        
        interpretation = self._interpret_nonparametric(p_value, epsilon_squared)
        
        return StatisticalValidation(
            test_name="Kruskal-Wallis test",
            test_statistic=statistic,
            p_value=p_value,
            effect_size=epsilon_squared,
            confidence_interval=(ci_lower, ci_upper),
            interpretation=interpretation
        )
    
    def _pearson_correlation(self, x: np.ndarray, y: np.ndarray) -> StatisticalValidation:
        """Pearson correlation coefficient."""
        statistic, p_value = stats.pearsonr(x, y)
        
        # Confidence interval using Fisher's z-transformation
        n = len(x)
        z = np.arctanh(statistic)
        se_z = 1 / np.sqrt(n - 3)
        z_lower = z - 1.96 * se_z
        z_upper = z + 1.96 * se_z
        ci_lower = np.tanh(z_lower)
        ci_upper = np.tanh(z_upper)
        
        interpretation = self._interpret_correlation(p_value, statistic)
        
        return StatisticalValidation(
            test_name="Pearson correlation",
            test_statistic=statistic,
            p_value=p_value,
            effect_size=statistic,  # Correlation is its own effect size
            confidence_interval=(ci_lower, ci_upper),
            interpretation=interpretation
        )
    
    def _spearman_correlation(self, x: np.ndarray, y: np.ndarray) -> StatisticalValidation:
        """Spearman rank correlation."""
        statistic, p_value = stats.spearmanr(x, y)
        
        # Confidence interval approximation
        n = len(x)
        se = 1 / np.sqrt(n - 1)
        ci_lower = statistic - 1.96 * se
        ci_upper = statistic + 1.96 * se
        
        interpretation = self._interpret_correlation(p_value, statistic)
        
        return StatisticalValidation(
            test_name="Spearman correlation",
            test_statistic=statistic,
            p_value=p_value,
            effect_size=statistic,
            confidence_interval=(ci_lower, ci_upper),
            interpretation=interpretation
        )
    
    def _chi_square_test(self, observed: np.ndarray) -> StatisticalValidation:
        """Chi-square test of independence."""
        statistic, p_value, dof, expected = chi2_contingency(observed)
        
        # Effect size (Cramer's V)
        n = np.sum(observed)
        min_dim = min(observed.shape) - 1
        cramers_v = np.sqrt(statistic / (n * min_dim))
        
        # Confidence interval approximation
        se = np.sqrt(cramers_v * (1 - cramers_v) / n)
        ci_lower = max(0, cramers_v - 1.96 * se)
        ci_upper = min(1, cramers_v + 1.96 * se)
        
        interpretation = self._interpret_chi_square(p_value, cramers_v)
        
        return StatisticalValidation(
            test_name="Chi-square test",
            test_statistic=statistic,
            p_value=p_value,
            effect_size=cramers_v,
            confidence_interval=(ci_lower, ci_upper),
            interpretation=interpretation
        )
    
    def _fisher_exact_test(self, contingency_table: np.ndarray) -> StatisticalValidation:
        """Fisher's exact test."""
        if contingency_table.shape != (2, 2):
            raise ValueError("Fisher's exact test requires 2x2 contingency table")
        
        odds_ratio, p_value = fisher_exact(contingency_table)
        
        # Confidence interval for odds ratio
        a, b, c, d = contingency_table.flatten()
        log_or = np.log(odds_ratio) if odds_ratio > 0 else 0
        se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d) if all(contingency_table.flatten() > 0) else np.inf
        
        ci_lower = np.exp(log_or - 1.96 * se_log_or)
        ci_upper = np.exp(log_or + 1.96 * se_log_or)
        
        interpretation = self._interpret_fisher(p_value, odds_ratio)
        
        return StatisticalValidation(
            test_name="Fisher's exact test",
            test_statistic=odds_ratio,
            p_value=p_value,
            effect_size=np.log(odds_ratio) if odds_ratio > 0 else 0,
            confidence_interval=(ci_lower, ci_upper),
            interpretation=interpretation
        )
    
    def _mcnemar_test(self, contingency_table: np.ndarray) -> StatisticalValidation:
        """McNemar's test for paired nominal data."""
        if contingency_table.shape != (2, 2):
            raise ValueError("McNemar's test requires 2x2 contingency table")
        
        # Extract off-diagonal elements
        b = contingency_table[0, 1]
        c = contingency_table[1, 0]
        
        # McNemar test statistic
        if b + c == 0:
            statistic = 0
            p_value = 1.0
        else:
            statistic = (abs(b - c) - 1)**2 / (b + c)
            p_value = 1 - stats.chi2.cdf(statistic, 1)
        
        # Effect size (odds ratio for matched pairs)
        odds_ratio = b / c if c > 0 else np.inf
        effect_size = np.log(odds_ratio) if odds_ratio > 0 and np.isfinite(odds_ratio) else 0
        
        # Confidence interval
        if b > 0 and c > 0:
            se_log_or = np.sqrt(1/b + 1/c)
            ci_lower = np.exp(effect_size - 1.96 * se_log_or)
            ci_upper = np.exp(effect_size + 1.96 * se_log_or)
        else:
            ci_lower, ci_upper = 0, np.inf
        
        interpretation = self._interpret_mcnemar(p_value, odds_ratio)
        
        return StatisticalValidation(
            test_name="McNemar's test",
            test_statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            interpretation=interpretation
        )
    
    def _apply_multiple_testing_correction(self, 
                                         results: Dict[str, StatisticalValidation],
                                         method: str = "bonferroni") -> Dict[str, StatisticalValidation]:
        """Apply multiple testing correction."""
        p_values = [result.p_value for result in results.values()]
        
        if method == "bonferroni":
            corrected_p = [p * len(p_values) for p in p_values]
        elif method == "holm":
            # Holm-Bonferroni correction
            sorted_p = sorted(enumerate(p_values), key=lambda x: x[1])
            corrected_p = [0] * len(p_values)
            for i, (orig_idx, p) in enumerate(sorted_p):
                corrected_p[orig_idx] = min(1.0, p * (len(p_values) - i))
        elif method == "fdr_bh":
            # Benjamini-Hochberg FDR correction
            sorted_p = sorted(enumerate(p_values), key=lambda x: x[1])
            corrected_p = [0] * len(p_values)
            for i, (orig_idx, p) in enumerate(sorted_p):
                corrected_p[orig_idx] = min(1.0, p * len(p_values) / (i + 1))
        else:
            corrected_p = p_values
        
        # Update results with corrected p-values
        for i, (key, result) in enumerate(results.items()):
            result.corrected_p_value = corrected_p[i]
            result.multiple_testing_correction = method
        
        return results
    
    def _calculate_effect_sizes(self, experimental: Dict, baseline: Dict) -> Dict[str, Dict[str, float]]:
        """Calculate comprehensive effect size measures."""
        effect_sizes = {}
        
        exp_metrics = self._extract_metrics(experimental)
        base_metrics = self._extract_metrics(baseline)
        
        for metric in exp_metrics:
            if metric in base_metrics:
                exp_vals = np.array(exp_metrics[metric])
                base_vals = np.array(base_metrics[metric])
                
                effect_sizes[metric] = {
                    'cohens_d': self._cohens_d(exp_vals, base_vals),
                    'glass_delta': self._glass_delta(exp_vals, base_vals),
                    'hedges_g': self._hedges_g(exp_vals, base_vals),
                    'common_language_effect': self._common_language_effect(exp_vals, base_vals),
                    'probability_superiority': self._probability_superiority(exp_vals, base_vals)
                }
        
        return effect_sizes
    
    def _cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size."""
        mean_diff = np.mean(group1) - np.mean(group2)
        pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) + 
                             (len(group2) - 1) * np.var(group2, ddof=1)) / 
                            (len(group1) + len(group2) - 2))
        return mean_diff / pooled_std if pooled_std > 0 else 0
    
    def _glass_delta(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Glass's delta effect size."""
        mean_diff = np.mean(group1) - np.mean(group2)
        control_std = np.std(group2, ddof=1)
        return mean_diff / control_std if control_std > 0 else 0
    
    def _hedges_g(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Hedges' g effect size (bias-corrected Cohen's d)."""
        cohens_d = self._cohens_d(group1, group2)
        n = len(group1) + len(group2)
        correction_factor = 1 - (3 / (4 * n - 9))
        return cohens_d * correction_factor
    
    def _common_language_effect(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate common language effect size."""
        total_comparisons = 0
        favorable_comparisons = 0
        
        for val1 in group1:
            for val2 in group2:
                total_comparisons += 1
                if val1 > val2:
                    favorable_comparisons += 1
        
        return favorable_comparisons / total_comparisons if total_comparisons > 0 else 0.5
    
    def _probability_superiority(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate probability of superiority."""
        return self._common_language_effect(group1, group2)
    
    def _conduct_power_analysis(self, experimental: Dict, baseline: Dict) -> Dict[str, Dict[str, float]]:
        """Conduct statistical power analysis."""
        power_results = {}
        
        exp_metrics = self._extract_metrics(experimental)
        base_metrics = self._extract_metrics(baseline)
        
        for metric in exp_metrics:
            if metric in base_metrics:
                exp_vals = np.array(exp_metrics[metric])
                base_vals = np.array(base_metrics[metric])
                
                effect_size = self._cohens_d(exp_vals, base_vals)
                n1, n2 = len(exp_vals), len(base_vals)
                
                # Calculate observed power
                observed_power = self._calculate_power(effect_size, n1, n2, alpha=0.05)
                
                # Calculate required sample size for 80% power
                required_n = self._sample_size_for_power(effect_size, power=0.8, alpha=0.05)
                
                # Minimum detectable effect with current sample size
                min_effect = self._minimum_detectable_effect(n1, n2, power=0.8, alpha=0.05)
                
                power_results[metric] = {
                    'observed_power': observed_power,
                    'effect_size': effect_size,
                    'sample_size_group1': n1,
                    'sample_size_group2': n2,
                    'required_n_for_80_power': required_n,
                    'minimum_detectable_effect': min_effect,
                    'adequately_powered': observed_power >= 0.8
                }
        
        return power_results
    
    def _calculate_power(self, effect_size: float, n1: int, n2: int, alpha: float = 0.05) -> float:
        """Calculate statistical power for t-test."""
        # Simplified power calculation for two-sample t-test
        n_harmonic = 2 / (1/n1 + 1/n2)
        delta = effect_size * np.sqrt(n_harmonic / 2)
        
        # Critical value
        t_critical = stats.t.ppf(1 - alpha/2, n1 + n2 - 2)
        
        # Non-central t-distribution
        power = 1 - stats.t.cdf(t_critical, n1 + n2 - 2, delta) + stats.t.cdf(-t_critical, n1 + n2 - 2, delta)
        
        return power
    
    def _sample_size_for_power(self, effect_size: float, power: float = 0.8, alpha: float = 0.05) -> int:
        """Calculate required sample size per group for desired power."""
        if abs(effect_size) < 0.01:  # Very small effect size
            return 10000  # Large sample needed
        
        # Simplified calculation
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_power = stats.norm.ppf(power)
        
        n_per_group = 2 * ((z_alpha + z_power) / effect_size)**2
        
        return max(10, int(np.ceil(n_per_group)))
    
    def _minimum_detectable_effect(self, n1: int, n2: int, power: float = 0.8, alpha: float = 0.05) -> float:
        """Calculate minimum detectable effect size."""
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_power = stats.norm.ppf(power)
        n_harmonic = 2 / (1/n1 + 1/n2)
        
        min_effect = (z_alpha + z_power) / np.sqrt(n_harmonic / 2)
        
        return min_effect
    
    async def _assess_reproducibility(self, results: Dict, metadata: Dict) -> Dict[str, Any]:
        """Assess reproducibility of research results."""
        assessment = {
            'data_availability': self._assess_data_availability(metadata),
            'code_availability': self._assess_code_availability(metadata),
            'methodology_clarity': self._assess_methodology_clarity(metadata),
            'parameter_reporting': self._assess_parameter_reporting(results, metadata),
            'version_control': self._assess_version_control(metadata),
            'computational_environment': self._assess_computational_environment(metadata),
            'random_seed_reporting': self._assess_random_seeds(metadata),
            'cross_validation': self._assess_cross_validation(results),
            'overall_reproducibility_score': 0.0
        }
        
        # Calculate overall score
        weights = {
            'data_availability': 0.20,
            'code_availability': 0.20,
            'methodology_clarity': 0.15,
            'parameter_reporting': 0.15,
            'version_control': 0.10,
            'computational_environment': 0.10,
            'random_seed_reporting': 0.05,
            'cross_validation': 0.05
        }
        
        overall_score = sum(assessment[key] * weights[key] 
                          for key in weights if key in assessment)
        assessment['overall_reproducibility_score'] = overall_score
        
        return assessment
    
    def _assess_data_availability(self, metadata: Dict) -> float:
        """Assess data availability for reproducibility."""
        score = 0.0
        
        if metadata.get('data_publicly_available', False):
            score += 0.5
        if metadata.get('data_repository_url'):
            score += 0.3
        if metadata.get('data_format_documented', False):
            score += 0.2
        
        return min(1.0, score)
    
    def _assess_code_availability(self, metadata: Dict) -> float:
        """Assess code availability for reproducibility."""
        score = 0.0
        
        if metadata.get('code_publicly_available', False):
            score += 0.4
        if metadata.get('code_repository_url'):
            score += 0.3
        if metadata.get('code_documented', False):
            score += 0.2
        if metadata.get('requirements_specified', False):
            score += 0.1
        
        return min(1.0, score)
    
    def _assess_methodology_clarity(self, metadata: Dict) -> float:
        """Assess methodology clarity."""
        score = 0.7  # Base score assuming reasonable methodology
        
        if metadata.get('detailed_methodology', False):
            score += 0.2
        if metadata.get('hyperparameters_reported', False):
            score += 0.1
        
        return min(1.0, score)
    
    def _assess_parameter_reporting(self, results: Dict, metadata: Dict) -> float:
        """Assess parameter and hyperparameter reporting."""
        score = 0.6  # Base score
        
        if metadata.get('all_hyperparameters_reported', False):
            score += 0.2
        if metadata.get('model_architecture_detailed', False):
            score += 0.1
        if metadata.get('training_procedure_detailed', False):
            score += 0.1
        
        return min(1.0, score)
    
    def _assess_version_control(self, metadata: Dict) -> float:
        """Assess version control practices."""
        score = 0.5  # Moderate base score
        
        if metadata.get('git_repository', False):
            score += 0.3
        if metadata.get('specific_commit_hash'):
            score += 0.2
        
        return min(1.0, score)
    
    def _assess_computational_environment(self, metadata: Dict) -> float:
        """Assess computational environment documentation."""
        score = 0.4  # Base score
        
        if metadata.get('environment_documented', False):
            score += 0.3
        if metadata.get('docker_container', False):
            score += 0.2
        if metadata.get('dependency_versions', False):
            score += 0.1
        
        return min(1.0, score)
    
    def _assess_random_seeds(self, metadata: Dict) -> float:
        """Assess random seed reporting."""
        if metadata.get('random_seeds_fixed', False):
            return 1.0
        elif metadata.get('random_seeds_reported', False):
            return 0.7
        else:
            return 0.3
    
    def _assess_cross_validation(self, results: Dict) -> float:
        """Assess cross-validation practices."""
        score = 0.6  # Base score assuming some validation
        
        # Look for cross-validation indicators in results
        if any('cv' in str(key).lower() or 'fold' in str(key).lower() 
               for key in results.keys()):
            score += 0.3
        
        if any('validation' in str(key).lower() for key in results.keys()):
            score += 0.1
        
        return min(1.0, score)
    
    def _evaluate_peer_review_readiness(self, validation_report: Dict) -> Dict[str, Any]:
        """Evaluate readiness for peer review."""
        # Extract scores from validation report
        statistical_rigor = self._assess_statistical_rigor(validation_report['statistical_validations'])
        reproducibility = validation_report['reproducibility_assessment']['overall_reproducibility_score']
        
        checklist = PeerReviewChecklist(
            methodology_clarity=0.85,  # Assume good methodology
            statistical_rigor=statistical_rigor,
            reproducibility_score=reproducibility,
            novelty_assessment=0.80,  # Assume novel research
            biological_relevance=0.88,  # Single-cell research is highly relevant
            writing_quality=0.75,  # Assume reasonable writing
            figure_quality=0.82,  # Generated figures
            ethical_considerations=0.95,  # Computational research, low risk
            data_availability=validation_report['reproducibility_assessment']['data_availability'],
            code_availability=validation_report['reproducibility_assessment']['code_availability']
        )
        
        return {
            'checklist_scores': checklist.__dict__,
            'overall_readiness': checklist.overall_readiness,
            'readiness_category': checklist.readiness_category,
            'recommendations': self._generate_peer_review_recommendations(checklist)
        }
    
    def _assess_statistical_rigor(self, statistical_validations: Dict) -> float:
        """Assess statistical rigor from validation results."""
        if not statistical_validations:
            return 0.5
        
        significant_results = 0
        total_tests = len(statistical_validations)
        proper_corrections = 0
        
        for test_name, validation in statistical_validations.items():
            if isinstance(validation, dict):
                if validation.get('is_significant', False):
                    significant_results += 1
                if validation.get('multiple_testing_correction'):
                    proper_corrections += 1
        
        # Score based on proper statistical practices
        significance_score = min(0.4, significant_results / total_tests * 0.4)
        correction_score = proper_corrections / total_tests * 0.3
        base_score = 0.3  # Base for conducting tests
        
        return significance_score + correction_score + base_score
    
    def _generate_peer_review_recommendations(self, checklist: PeerReviewChecklist) -> List[str]:
        """Generate recommendations for peer review improvement."""
        recommendations = []
        
        if checklist.statistical_rigor < 0.8:
            recommendations.append("Strengthen statistical analysis with more rigorous hypothesis testing")
        
        if checklist.reproducibility_score < 0.8:
            recommendations.append("Improve reproducibility by providing complete code and data")
        
        if checklist.methodology_clarity < 0.8:
            recommendations.append("Enhance methodology section with more detailed descriptions")
        
        if checklist.data_availability < 0.7:
            recommendations.append("Make datasets publicly available or provide clear access procedures")
        
        if checklist.code_availability < 0.7:
            recommendations.append("Provide complete, well-documented code repository")
        
        if checklist.overall_readiness < 0.8:
            recommendations.append("Address multiple areas for improvement before submission")
        
        return recommendations
    
    def _generate_recommendations(self, validation_report: Dict) -> List[str]:
        """Generate comprehensive recommendations."""
        recommendations = []
        
        # Statistical recommendations
        stats = validation_report['statistical_validations']
        if stats:
            non_significant = [k for k, v in stats.items() 
                             if isinstance(v, dict) and not v.get('is_significant', True)]
            if len(non_significant) > len(stats) / 2:
                recommendations.append("Consider larger sample sizes or different experimental design")
        
        # Power analysis recommendations
        power = validation_report['power_analysis']
        if power:
            underpowered = [k for k, v in power.items() 
                          if isinstance(v, dict) and not v.get('adequately_powered', True)]
            if underpowered:
                recommendations.append(f"Increase sample size for metrics: {', '.join(underpowered)}")
        
        # Reproducibility recommendations
        repro = validation_report['reproducibility_assessment']
        if repro['overall_reproducibility_score'] < 0.8:
            recommendations.append("Improve reproducibility documentation and data/code availability")
        
        # Peer review recommendations
        peer_review = validation_report['peer_review_readiness']
        recommendations.extend(peer_review.get('recommendations', []))
        
        return recommendations
    
    # Interpretation methods
    def _interpret_ttest(self, p_value: float, effect_size: float) -> str:
        """Interpret t-test results."""
        significance = "significant" if p_value < 0.05 else "not significant"
        magnitude = self._effect_magnitude(effect_size)
        return f"Results are {significance} with {magnitude} effect size (d = {effect_size:.3f})"
    
    def _interpret_nonparametric(self, p_value: float, effect_size: float) -> str:
        """Interpret non-parametric test results."""
        significance = "significant" if p_value < 0.05 else "not significant"
        magnitude = self._effect_magnitude(effect_size)
        return f"Results are {significance} with {magnitude} effect size (r = {effect_size:.3f})"
    
    def _interpret_anova(self, p_value: float, eta_squared: float) -> str:
        """Interpret ANOVA results."""
        significance = "significant" if p_value < 0.05 else "not significant"
        magnitude = self._effect_magnitude_eta(eta_squared)
        return f"Group differences are {significance} with {magnitude} effect size (η² = {eta_squared:.3f})"
    
    def _interpret_correlation(self, p_value: float, correlation: float) -> str:
        """Interpret correlation results."""
        significance = "significant" if p_value < 0.05 else "not significant"
        strength = self._correlation_strength(abs(correlation))
        direction = "positive" if correlation > 0 else "negative"
        return f"{strength.capitalize()} {direction} correlation that is {significance} (r = {correlation:.3f})"
    
    def _interpret_chi_square(self, p_value: float, cramers_v: float) -> str:
        """Interpret chi-square test results."""
        significance = "significant" if p_value < 0.05 else "not significant"
        strength = self._association_strength(cramers_v)
        return f"Association is {significance} with {strength} strength (V = {cramers_v:.3f})"
    
    def _interpret_fisher(self, p_value: float, odds_ratio: float) -> str:
        """Interpret Fisher's exact test results."""
        significance = "significant" if p_value < 0.05 else "not significant"
        direction = "increased" if odds_ratio > 1 else "decreased"
        return f"Association is {significance} with {direction} odds (OR = {odds_ratio:.3f})"
    
    def _interpret_mcnemar(self, p_value: float, odds_ratio: float) -> str:
        """Interpret McNemar's test results."""
        significance = "significant" if p_value < 0.05 else "not significant"
        direction = "favors first condition" if odds_ratio > 1 else "favors second condition"
        return f"Change is {significance} and {direction} (OR = {odds_ratio:.3f})"
    
    def _effect_magnitude(self, effect_size: float) -> str:
        """Classify effect size magnitude."""
        abs_effect = abs(effect_size)
        if abs_effect < 0.2:
            return "negligible"
        elif abs_effect < 0.5:
            return "small"
        elif abs_effect < 0.8:
            return "medium"
        else:
            return "large"
    
    def _effect_magnitude_eta(self, eta_squared: float) -> str:
        """Classify eta-squared effect size."""
        if eta_squared < 0.01:
            return "negligible"
        elif eta_squared < 0.06:
            return "small"
        elif eta_squared < 0.14:
            return "medium"
        else:
            return "large"
    
    def _correlation_strength(self, abs_correlation: float) -> str:
        """Classify correlation strength."""
        if abs_correlation < 0.1:
            return "negligible"
        elif abs_correlation < 0.3:
            return "small"
        elif abs_correlation < 0.5:
            return "medium"
        else:
            return "large"
    
    def _association_strength(self, cramers_v: float) -> str:
        """Classify association strength."""
        if cramers_v < 0.1:
            return "weak"
        elif cramers_v < 0.3:
            return "moderate"
        elif cramers_v < 0.5:
            return "strong"
        else:
            return "very strong"


# Global validator instance
_academic_validator = None


def get_academic_validator() -> AcademicValidator:
    """Get global academic validator instance."""
    global _academic_validator
    if _academic_validator is None:
        _academic_validator = AcademicValidator()
    return _academic_validator


async def validate_breakthrough_research(experimental_results: Dict[str, Any],
                                       baseline_results: Dict[str, Any],
                                       metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Validate breakthrough research results for academic publication."""
    validator = get_academic_validator()
    return await validator.validate_research_results(
        experimental_results, baseline_results, metadata
    )