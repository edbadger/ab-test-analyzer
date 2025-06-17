import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import io
import json

from statistical_tests import ABTestAnalyzer
from utils import validate_inputs, format_results, create_results_dataframe

# Page configuration
st.set_page_config(
    page_title="A/B Test Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None

def main():
    st.title("🚀 Amazing A/B Test Tool")
    st.markdown("Analyze your A/B test results to determine statistical significance and get actionable insights.")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Significance level selection
        significance_level = st.selectbox(
            "Significance Level",
            options=[0.90, 0.95, 0.99],
            index=1,
            format_func=lambda x: f"{x*100:.0f}% ({1-x:.3f} α)"
        )
        
        # Test type selection
        test_type = st.selectbox(
            "Statistical Test",
            options=["Two-Proportion Z-Test", "Chi-Square Test"],
            help="Z-test is typically used for conversion rates, Chi-square for categorical data"
        )
        
        # Input method selection
        input_method = st.radio(
            "Input Method",
            options=["Conversion Rates", "Raw Counts"],
            help="Choose whether to input conversion rates directly or raw success/total counts"
        )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📋 Test Data Input")
        
        if input_method == "Conversion Rates":
            st.subheader("Group A (Control)")
            sample_size_a = st.number_input(
                "Sample Size A",
                min_value=1,
                value=1000,
                help="Total number of visitors/users in control group"
            )
            conversion_rate_a = st.number_input(
                "Conversion Rate A (%)",
                min_value=0.0,
                max_value=100.0,
                value=5.0,
                step=0.1,
                help="Conversion rate as percentage"
            ) / 100
            
            st.subheader("Group B (Treatment)")
            sample_size_b = st.number_input(
                "Sample Size B",
                min_value=1,
                value=1000,
                help="Total number of visitors/users in treatment group"
            )
            conversion_rate_b = st.number_input(
                "Conversion Rate B (%)",
                min_value=0.0,
                max_value=100.0,
                value=6.0,
                step=0.1,
                help="Conversion rate as percentage"
            ) / 100
            
            # Calculate conversions from rates
            conversions_a = int(sample_size_a * conversion_rate_a)
            conversions_b = int(sample_size_b * conversion_rate_b)
            
        else:  # Raw Counts
            st.subheader("Group A (Control)")
            conversions_a = st.number_input(
                "Conversions A",
                min_value=0,
                value=50,
                help="Number of successful conversions in control group"
            )
            sample_size_a = st.number_input(
                "Total Visitors A",
                min_value=1,
                value=1000,
                help="Total number of visitors/users in control group"
            )
            
            st.subheader("Group B (Treatment)")
            conversions_b = st.number_input(
                "Conversions B",
                min_value=0,
                value=60,
                help="Number of successful conversions in treatment group"
            )
            sample_size_b = st.number_input(
                "Total Visitors B",
                min_value=1,
                value=1000,
                help="Total number of visitors/users in treatment group"
            )
            
            # Calculate rates from raw counts
            conversion_rate_a = conversions_a / sample_size_a
            conversion_rate_b = conversions_b / sample_size_b
        
        # Validate inputs
        validation_errors = validate_inputs(
            conversions_a, sample_size_a, conversions_b, sample_size_b
        )
        
        if validation_errors:
            for error in validation_errors:
                st.error(error)
            st.stop()
        
        # Calculate button
        if st.button("🧮 Calculate Statistical Significance", type="primary"):
            analyzer = ABTestAnalyzer()
            
            try:
                if test_type == "Two-Proportion Z-Test":
                    results = analyzer.two_proportion_z_test(
                        conversions_a, sample_size_a, conversions_b, sample_size_b, significance_level
                    )
                else:  # Chi-Square Test
                    results = analyzer.chi_square_test(
                        conversions_a, sample_size_a, conversions_b, sample_size_b, significance_level
                    )
                
                st.session_state.results = results
                st.success("✅ Analysis completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Error in calculation: {str(e)}")
    
    with col2:
        st.header("📈 Results & Interpretation")
        
        if st.session_state.results is not None:
            results = st.session_state.results
            
            # Key metrics display
            col2_1, col2_2 = st.columns(2)
            
            with col2_1:
                st.metric(
                    "Control Rate (A)",
                    f"{conversion_rate_a:.2%}",
                    f"{conversions_a}/{sample_size_a}"
                )
                
            with col2_2:
                st.metric(
                    "Treatment Rate (B)",
                    f"{conversion_rate_b:.2%}",
                    f"{conversions_b}/{sample_size_b}",
                    delta=f"{(conversion_rate_b - conversion_rate_a):.2%}"
                )
            
            # Statistical significance results
            st.subheader("🔬 Statistical Analysis")
            
            # Significance indicator
            if results['is_significant']:
                st.success(f"✅ **STATISTICALLY SIGNIFICANT** at {significance_level*100:.0f}% confidence level")
            else:
                st.warning(f"⚠️ **NOT STATISTICALLY SIGNIFICANT** at {significance_level*100:.0f}% confidence level")
            
            # Key statistics
            col_stats1, col_stats2, col_stats3 = st.columns(3)
            
            with col_stats1:
                st.metric("P-Value", f"{results['p_value']:.6f}")
            
            with col_stats2:
                st.metric("Test Statistic", f"{results['test_statistic']:.4f}")
            
            with col_stats3:
                st.metric("Critical Value", f"{results['critical_value']:.4f}")
            
            # Confidence intervals
            if 'confidence_interval_diff' in results:
                st.subheader("📊 Confidence Intervals")
                ci_lower, ci_upper = results['confidence_interval_diff']
                st.info(f"**{significance_level*100:.0f}% Confidence Interval for Difference:** [{ci_lower:.4f}, {ci_upper:.4f}]")
                
                if ci_lower > 0:
                    st.success("✅ The confidence interval is entirely above zero, confirming a positive effect.")
                elif ci_upper < 0:
                    st.success("✅ The confidence interval is entirely below zero, confirming a negative effect.")
                else:
                    st.warning("⚠️ The confidence interval includes zero, suggesting no significant difference.")
            
            # Effect size and practical significance
            st.subheader("📏 Effect Size Analysis")
            
            relative_improvement = ((conversion_rate_b - conversion_rate_a) / conversion_rate_a) * 100
            absolute_improvement = (conversion_rate_b - conversion_rate_a) * 100
            
            col_effect1, col_effect2 = st.columns(2)
            
            with col_effect1:
                st.metric("Relative Improvement", f"{relative_improvement:.1f}%")
            
            with col_effect2:
                st.metric("Absolute Improvement", f"{absolute_improvement:.2f} percentage points")
            
            # Interpretation and recommendations
            st.subheader("💡 Interpretation & Recommendations")
            
            interpretation = format_results(results, conversion_rate_a, conversion_rate_b, significance_level)
            st.markdown(interpretation)
            
            # Power analysis (sample size recommendations)
            st.subheader("⚡ Sample Size & Power Analysis")
            
            total_sample_size = sample_size_a + sample_size_b
            st.info(f"**Current Total Sample Size:** {total_sample_size:,}")
            
            # Simple power estimation based on effect size
            effect_size = abs(conversion_rate_b - conversion_rate_a)
            if effect_size > 0:
                # Rough estimate for required sample size per group for 80% power
                estimated_n_per_group = int((3.84 * (conversion_rate_a * (1 - conversion_rate_a) + conversion_rate_b * (1 - conversion_rate_b))) / (effect_size ** 2))
                st.info(f"**Estimated Sample Size per Group for 80% Power:** {estimated_n_per_group:,}")
            
            # Visualization
            st.subheader("📊 Visual Comparison")
            
            # Create comparison chart
            fig = go.Figure()
            
            groups = ['Control (A)', 'Treatment (B)']
            rates = [conversion_rate_a * 100, conversion_rate_b * 100]
            colors = ['#FF6B6B', '#4ECDC4']
            
            fig.add_trace(go.Bar(
                x=groups,
                y=rates,
                marker_color=colors,
                text=[f'{rate:.2f}%' for rate in rates],
                textposition='auto',
            ))
            
            fig.update_layout(
                title='Conversion Rate Comparison',
                yaxis_title='Conversion Rate (%)',
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Export results
            st.subheader("💾 Export Results")
            
            # Create downloadable report
            results_df = create_results_dataframe(results, {
                'control_conversions': conversions_a,
                'control_sample_size': sample_size_a,
                'treatment_conversions': conversions_b,
                'treatment_sample_size': sample_size_b,
                'control_rate': conversion_rate_a,
                'treatment_rate': conversion_rate_b,
                'significance_level': significance_level,
                'test_type': test_type
            })
            
            col_export1, col_export2 = st.columns(2)
            
            with col_export1:
                csv_buffer = io.StringIO()
                results_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📄 Download CSV Report",
                    data=csv_buffer.getvalue(),
                    file_name=f"ab_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col_export2:
                json_data = {
                    'test_data': {
                        'control_conversions': conversions_a,
                        'control_sample_size': sample_size_a,
                        'treatment_conversions': conversions_b,
                        'treatment_sample_size': sample_size_b,
                        'control_rate': conversion_rate_a,
                        'treatment_rate': conversion_rate_b
                    },
                    'test_configuration': {
                        'significance_level': significance_level,
                        'test_type': test_type,
                        'input_method': input_method
                    },
                    'results': results,
                    'timestamp': datetime.now().isoformat()
                }
                
                st.download_button(
                    label="📋 Download JSON Report",
                    data=json.dumps(json_data, indent=2),
                    file_name=f"ab_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        else:
            st.info("👈 Enter your A/B test data on the left and click 'Calculate Statistical Significance' to see results.")
            
            # Show example data format
            st.subheader("📋 Example Data Format")
            example_data = pd.DataFrame({
                'Group': ['Control (A)', 'Treatment (B)'],
                'Conversions': [50, 60],
                'Sample Size': [1000, 1000],
                'Conversion Rate': ['5.0%', '6.0%']
            })
            st.table(example_data)

if __name__ == "__main__":
    main()
