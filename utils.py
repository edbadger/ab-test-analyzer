import pandas as pd
from typing import List, Dict, Any
import math

def validate_inputs(conversions_a: int, sample_size_a: int, conversions_b: int, sample_size_b: int) -> List[str]:
    """
    Validate input parameters for A/B test analysis.
    
    Args:
        conversions_a: Number of conversions in group A
        sample_size_a: Sample size for group A
        conversions_b: Number of conversions in group B
        sample_size_b: Sample size for group B
    
    Returns:
        List of validation error messages (empty if all inputs are valid)
    """
    errors = []
    
    # Check for negative values
    if conversions_a < 0:
        errors.append("Conversions for Group A cannot be negative")
    if conversions_b < 0:
        errors.append("Conversions for Group B cannot be negative")
    if sample_size_a <= 0:
        errors.append("Sample size for Group A must be greater than 0")
    if sample_size_b <= 0:
        errors.append("Sample size for Group B must be greater than 0")
    
    # Check if conversions exceed sample sizes
    if conversions_a > sample_size_a:
        errors.append("Conversions for Group A cannot exceed sample size")
    if conversions_b > sample_size_b:
        errors.append("Conversions for Group B cannot exceed sample size")
    
    # Check for minimum sample size requirements
    if sample_size_a < 30:
        errors.append("Sample size for Group A should be at least 30 for reliable results")
    if sample_size_b < 30:
        errors.append("Sample size for Group B should be at least 30 for reliable results")
    
    # Check for minimum expected frequencies (rule of thumb for chi-square)
    if conversions_a < 5 or (sample_size_a - conversions_a) < 5:
        errors.append("Group A should have at least 5 conversions and 5 non-conversions for reliable chi-square test")
    if conversions_b < 5 or (sample_size_b - conversions_b) < 5:
        errors.append("Group B should have at least 5 conversions and 5 non-conversions for reliable chi-square test")
    
    return errors

def format_results(results: Dict[str, Any], rate_a: float, rate_b: float, significance_level: float) -> str:
    """
    Format statistical test results into human-readable interpretation.
    
    Args:
        results: Dictionary containing test results
        rate_a: Conversion rate for group A
        rate_b: Conversion rate for group B
        significance_level: Significance level used in the test
    
    Returns:
        Formatted interpretation string
    """
    
    interpretation = ""
    
    # Basic result interpretation
    if results['is_significant']:
        interpretation += f"🎯 **The test shows a statistically significant difference** at the {significance_level*100:.0f}% confidence level.\n\n"
        
        if rate_b > rate_a:
            interpretation += f"📈 **Treatment (B) performs significantly better** than Control (A).\n"
            relative_improvement = ((rate_b - rate_a) / rate_a) * 100
            interpretation += f"• Relative improvement: **{relative_improvement:.1f}%**\n"
        else:
            interpretation += f"📉 **Treatment (B) performs significantly worse** than Control (A).\n"
            relative_decline = ((rate_a - rate_b) / rate_a) * 100
            interpretation += f"• Relative decline: **{relative_decline:.1f}%**\n"
        
        absolute_diff = abs(rate_b - rate_a) * 100
        interpretation += f"• Absolute difference: **{absolute_diff:.2f} percentage points**\n\n"
        
    else:
        interpretation += f"⚠️ **No statistically significant difference detected** at the {significance_level*100:.0f}% confidence level.\n\n"
        interpretation += "This could mean:\n"
        interpretation += "• There is truly no difference between the groups\n"
        interpretation += "• The sample size is too small to detect the difference\n"
        interpretation += "• The effect size is smaller than expected\n\n"
    
    # P-value interpretation
    p_val = results['p_value']
    interpretation += f"📊 **P-value: {p_val:.6f}**\n"
    
    if p_val < 0.001:
        interpretation += "• Very strong evidence against the null hypothesis\n"
    elif p_val < 0.01:
        interpretation += "• Strong evidence against the null hypothesis\n"
    elif p_val < 0.05:
        interpretation += "• Moderate evidence against the null hypothesis\n"
    elif p_val < 0.1:
        interpretation += "• Weak evidence against the null hypothesis\n"
    else:
        interpretation += "• Little to no evidence against the null hypothesis\n"
    
    interpretation += "\n"
    
    # Effect size interpretation
    if 'effect_size_cohens_h' in results:
        cohens_h = abs(results['effect_size_cohens_h'])
        interpretation += f"📏 **Effect Size (Cohen's h): {cohens_h:.3f}**\n"
        
        if cohens_h < 0.2:
            interpretation += "• Small effect size\n"
        elif cohens_h < 0.5:
            interpretation += "• Medium effect size\n"
        elif cohens_h < 0.8:
            interpretation += "• Large effect size\n"
        else:
            interpretation += "• Very large effect size\n"
    
    elif 'effect_size_cramers_v' in results:
        cramers_v = results['effect_size_cramers_v']
        interpretation += f"📏 **Effect Size (Cramér's V): {cramers_v:.3f}**\n"
        
        if cramers_v < 0.1:
            interpretation += "• Small effect size\n"
        elif cramers_v < 0.3:
            interpretation += "• Medium effect size\n"
        elif cramers_v < 0.5:
            interpretation += "• Large effect size\n"
        else:
            interpretation += "• Very large effect size\n"
    
    interpretation += "\n"
    
    # Confidence interval interpretation
    if 'confidence_interval_diff' in results:
        ci_lower, ci_upper = results['confidence_interval_diff']
        interpretation += f"🎯 **{significance_level*100:.0f}% Confidence Interval for Difference:**\n"
        interpretation += f"• [{ci_lower:.4f}, {ci_upper:.4f}]\n"
        
        if ci_lower > 0:
            interpretation += "• We are confident the treatment has a positive effect\n"
        elif ci_upper < 0:
            interpretation += "• We are confident the treatment has a negative effect\n"
        else:
            interpretation += "• The effect could be positive, negative, or zero\n"
    
    interpretation += "\n"
    
    # Recommendations
    interpretation += "💡 **Recommendations:**\n"
    
    if results['is_significant']:
        if rate_b > rate_a:
            interpretation += "• ✅ **Implement the treatment** - it shows significant improvement\n"
            interpretation += "• 📊 Monitor key metrics after implementation\n"
            interpretation += "• 🔄 Consider running follow-up tests to validate results\n"
        else:
            interpretation += "• ❌ **Do not implement the treatment** - it performs significantly worse\n"
            interpretation += "• 🔍 Investigate why the treatment underperformed\n"
            interpretation += "• 🛠️ Consider alternative approaches\n"
    else:
        interpretation += "• 📈 **Consider increasing sample size** for more statistical power\n"
        interpretation += "• ⏱️ **Run the test longer** to collect more data\n"
        interpretation += "• 🎯 **Review effect size expectations** - the true effect might be smaller\n"
        interpretation += "• 🔄 **Consider a different approach** if business impact is needed urgently\n"
    
    return interpretation

def create_results_dataframe(results: Dict[str, Any], test_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Create a pandas DataFrame with test results for export.
    
    Args:
        results: Dictionary containing test results
        test_data: Dictionary containing original test data
    
    Returns:
        DataFrame with formatted results
    """
    
    data = []
    
    # Test configuration
    data.append(['Test Configuration', 'Test Type', results['test_type']])
    data.append(['Test Configuration', 'Significance Level', f"{results['significance_level']*100:.1f}%"])
    data.append(['Test Configuration', 'Alpha', f"{results['alpha']:.3f}"])
    
    # Input data
    data.append(['Input Data', 'Control Conversions', test_data['control_conversions']])
    data.append(['Input Data', 'Control Sample Size', test_data['control_sample_size']])
    data.append(['Input Data', 'Treatment Conversions', test_data['treatment_conversions']])
    data.append(['Input Data', 'Treatment Sample Size', test_data['treatment_sample_size']])
    
    # Calculated rates
    data.append(['Calculated Rates', 'Control Rate', f"{test_data['control_rate']:.4f}"])
    data.append(['Calculated Rates', 'Treatment Rate', f"{test_data['treatment_rate']:.4f}"])
    data.append(['Calculated Rates', 'Absolute Difference', f"{results['difference']:.4f}"])
    
    relative_change = ((test_data['treatment_rate'] - test_data['control_rate']) / test_data['control_rate']) * 100
    data.append(['Calculated Rates', 'Relative Change', f"{relative_change:.2f}%"])
    
    # Statistical results
    data.append(['Statistical Results', 'Test Statistic', f"{results['test_statistic']:.6f}"])
    data.append(['Statistical Results', 'P-Value', f"{results['p_value']:.6f}"])
    data.append(['Statistical Results', 'Critical Value', f"{results['critical_value']:.6f}"])
    data.append(['Statistical Results', 'Is Significant', 'Yes' if results['is_significant'] else 'No'])
    
    # Confidence interval
    if 'confidence_interval_diff' in results:
        ci_lower, ci_upper = results['confidence_interval_diff']
        data.append(['Confidence Interval', 'Lower Bound', f"{ci_lower:.6f}"])
        data.append(['Confidence Interval', 'Upper Bound', f"{ci_upper:.6f}"])
        data.append(['Confidence Interval', 'Margin of Error', f"{results['margin_of_error']:.6f}"])
    
    # Effect size
    if 'effect_size_cohens_h' in results:
        data.append(['Effect Size', "Cohen's h", f"{results['effect_size_cohens_h']:.6f}"])
    elif 'effect_size_cramers_v' in results:
        data.append(['Effect Size', "Cramér's V", f"{results['effect_size_cramers_v']:.6f}"])
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=['Category', 'Metric', 'Value'])
    
    return df

def interpret_effect_size(effect_size: float, effect_type: str = 'cohens_h') -> str:
    """
    Interpret effect size magnitude.
    
    Args:
        effect_size: Calculated effect size value
        effect_type: Type of effect size ('cohens_h' or 'cramers_v')
    
    Returns:
        String interpretation of effect size
    """
    
    abs_effect = abs(effect_size)
    
    if effect_type == 'cohens_h':
        if abs_effect < 0.2:
            return "Small"
        elif abs_effect < 0.5:
            return "Medium"
        elif abs_effect < 0.8:
            return "Large"
        else:
            return "Very Large"
    
    elif effect_type == 'cramers_v':
        if abs_effect < 0.1:
            return "Small"
        elif abs_effect < 0.3:
            return "Medium"
        elif abs_effect < 0.5:
            return "Large"
        else:
            return "Very Large"
    
    return "Unknown"

def calculate_business_impact(
    conversion_diff: float, 
    sample_size: int, 
    avg_order_value: float = None,
    time_period_days: int = None
) -> Dict[str, float]:
    """
    Calculate potential business impact of the A/B test results.
    
    Args:
        conversion_diff: Difference in conversion rates
        sample_size: Total sample size per time period
        avg_order_value: Average order value (optional)
        time_period_days: Time period in days (optional)
    
    Returns:
        Dictionary with business impact metrics
    """
    
    impact = {}
    
    # Additional conversions per period
    additional_conversions = conversion_diff * sample_size
    impact['additional_conversions_per_period'] = additional_conversions
    
    # Revenue impact (if AOV provided)
    if avg_order_value is not None:
        revenue_impact = additional_conversions * avg_order_value
        impact['revenue_impact_per_period'] = revenue_impact
        
        # Annualized revenue impact (if time period provided)
        if time_period_days is not None:
            periods_per_year = 365 / time_period_days
            annual_revenue_impact = revenue_impact * periods_per_year
            impact['annual_revenue_impact'] = annual_revenue_impact
    
    return impact
