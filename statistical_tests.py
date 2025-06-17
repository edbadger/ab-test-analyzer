import numpy as np
import scipy.stats as stats
from typing import Dict, Tuple, Any
import math

class ABTestAnalyzer:
    """
    A class to perform statistical significance testing for A/B tests.
    Supports both two-proportion z-tests and chi-square tests.
    """
    
    def __init__(self):
        pass
    
    def two_proportion_z_test(
        self, 
        conversions_a: int, 
        sample_size_a: int, 
        conversions_b: int, 
        sample_size_b: int, 
        significance_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Perform a two-proportion z-test to compare conversion rates between two groups.
        
        Args:
            conversions_a: Number of conversions in group A (control)
            sample_size_a: Total sample size for group A
            conversions_b: Number of conversions in group B (treatment)
            sample_size_b: Total sample size for group B
            significance_level: Confidence level (e.g., 0.95 for 95%)
        
        Returns:
            Dictionary containing test results and statistics
        """
        
        # Calculate conversion rates
        p_a = conversions_a / sample_size_a
        p_b = conversions_b / sample_size_b
        
        # Calculate pooled proportion
        p_pooled = (conversions_a + conversions_b) / (sample_size_a + sample_size_b)
        
        # Calculate standard error
        se = math.sqrt(p_pooled * (1 - p_pooled) * (1/sample_size_a + 1/sample_size_b))
        
        # Calculate z-statistic
        z_stat = (p_b - p_a) / se if se > 0 else 0
        
        # Calculate p-value (two-tailed test)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        # Determine critical value
        alpha = 1 - significance_level
        z_critical = stats.norm.ppf(1 - alpha/2)
        
        # Check significance
        is_significant = abs(z_stat) > z_critical
        
        # Calculate confidence interval for the difference
        diff = p_b - p_a
        se_diff = math.sqrt((p_a * (1 - p_a) / sample_size_a) + (p_b * (1 - p_b) / sample_size_b))
        margin_of_error = z_critical * se_diff
        
        ci_lower = diff - margin_of_error
        ci_upper = diff + margin_of_error
        
        # Calculate effect size (Cohen's h)
        cohens_h = 2 * (math.asin(math.sqrt(p_b)) - math.asin(math.sqrt(p_a)))
        
        return {
            'test_type': 'Two-Proportion Z-Test',
            'test_statistic': z_stat,
            'p_value': p_value,
            'critical_value': z_critical,
            'is_significant': is_significant,
            'significance_level': significance_level,
            'alpha': alpha,
            'conversion_rate_a': p_a,
            'conversion_rate_b': p_b,
            'difference': diff,
            'confidence_interval_diff': (ci_lower, ci_upper),
            'effect_size_cohens_h': cohens_h,
            'pooled_proportion': p_pooled,
            'standard_error': se,
            'margin_of_error': margin_of_error
        }
    
    def chi_square_test(
        self, 
        conversions_a: int, 
        sample_size_a: int, 
        conversions_b: int, 
        sample_size_b: int, 
        significance_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Perform a chi-square test of independence to compare conversion rates.
        
        Args:
            conversions_a: Number of conversions in group A (control)
            sample_size_a: Total sample size for group A
            conversions_b: Number of conversions in group B (treatment)
            sample_size_b: Total sample size for group B
            significance_level: Confidence level (e.g., 0.95 for 95%)
        
        Returns:
            Dictionary containing test results and statistics
        """
        
        # Create contingency table
        # Rows: [Converted, Not Converted]
        # Columns: [Group A, Group B]
        non_conversions_a = sample_size_a - conversions_a
        non_conversions_b = sample_size_b - conversions_b
        
        observed = np.array([
            [conversions_a, conversions_b],
            [non_conversions_a, non_conversions_b]
        ])
        
        # Perform chi-square test
        chi2_stat, p_value, dof, expected = stats.chi2_contingency(observed)
        
        # Calculate critical value
        alpha = 1 - significance_level
        chi2_critical = stats.chi2.ppf(1 - alpha, dof)
        
        # Check significance
        is_significant = chi2_stat > chi2_critical
        
        # Calculate conversion rates
        p_a = conversions_a / sample_size_a
        p_b = conversions_b / sample_size_b
        
        # Calculate Cramér's V (effect size for chi-square)
        n = sample_size_a + sample_size_b
        cramers_v = math.sqrt(chi2_stat / (n * min(observed.shape) - 1))
        
        # Calculate confidence interval using normal approximation
        # (Similar to z-test for large samples)
        p_pooled = (conversions_a + conversions_b) / (sample_size_a + sample_size_b)
        se_diff = math.sqrt((p_a * (1 - p_a) / sample_size_a) + (p_b * (1 - p_b) / sample_size_b))
        z_critical = stats.norm.ppf(1 - alpha/2)
        diff = p_b - p_a
        margin_of_error = z_critical * se_diff
        
        ci_lower = diff - margin_of_error
        ci_upper = diff + margin_of_error
        
        return {
            'test_type': 'Chi-Square Test',
            'test_statistic': chi2_stat,
            'p_value': p_value,
            'critical_value': chi2_critical,
            'degrees_of_freedom': dof,
            'is_significant': is_significant,
            'significance_level': significance_level,
            'alpha': alpha,
            'conversion_rate_a': p_a,
            'conversion_rate_b': p_b,
            'difference': diff,
            'confidence_interval_diff': (ci_lower, ci_upper),
            'effect_size_cramers_v': cramers_v,
            'contingency_table': observed.tolist(),
            'expected_frequencies': expected.tolist(),
            'margin_of_error': margin_of_error
        }
    
    def calculate_statistical_power(
        self, 
        sample_size_a: int, 
        sample_size_b: int, 
        p_a: float, 
        p_b: float, 
        significance_level: float = 0.95
    ) -> float:
        """
        Calculate the statistical power of the test.
        
        Args:
            sample_size_a: Sample size for group A
            sample_size_b: Sample size for group B
            p_a: True conversion rate for group A
            p_b: True conversion rate for group B
            significance_level: Significance level
        
        Returns:
            Statistical power (probability of detecting true effect)
        """
        
        alpha = 1 - significance_level
        z_alpha = stats.norm.ppf(1 - alpha/2)
        
        # Calculate pooled proportion under null hypothesis
        p_pooled = (p_a + p_b) / 2
        
        # Standard error under null hypothesis
        se_null = math.sqrt(p_pooled * (1 - p_pooled) * (1/sample_size_a + 1/sample_size_b))
        
        # Standard error under alternative hypothesis
        se_alt = math.sqrt((p_a * (1 - p_a) / sample_size_a) + (p_b * (1 - p_b) / sample_size_b))
        
        # Critical difference
        critical_diff = z_alpha * se_null
        
        # True difference
        true_diff = abs(p_b - p_a)
        
        # Z-score for power calculation
        z_beta = (true_diff - critical_diff) / se_alt
        
        # Statistical power
        power = stats.norm.cdf(z_beta)
        
        return max(0, min(1, power))
    
    def minimum_sample_size(
        self, 
        p_a: float, 
        p_b: float, 
        significance_level: float = 0.95, 
        power: float = 0.8,
        ratio: float = 1.0
    ) -> Tuple[int, int]:
        """
        Calculate minimum sample sizes needed for desired power.
        
        Args:
            p_a: Expected conversion rate for group A
            p_b: Expected conversion rate for group B
            significance_level: Desired significance level
            power: Desired statistical power
            ratio: Sample size ratio (n_b / n_a)
        
        Returns:
            Tuple of (sample_size_a, sample_size_b)
        """
        
        alpha = 1 - significance_level
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        # Effect size
        diff = abs(p_b - p_a)
        
        if diff == 0:
            raise ValueError("No difference between groups - cannot calculate sample size")
        
        # Pooled proportion
        p_pooled = (p_a + ratio * p_b) / (1 + ratio)
        
        # Calculate sample size for group A
        numerator = (z_alpha * math.sqrt((1 + 1/ratio) * p_pooled * (1 - p_pooled)) + 
                    z_beta * math.sqrt(p_a * (1 - p_a) + p_b * (1 - p_b) / ratio)) ** 2
        
        denominator = diff ** 2
        
        n_a = int(math.ceil(numerator / denominator))
        n_b = int(math.ceil(n_a * ratio))
        
        return n_a, n_b
