import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Report History & Analysis",
    page_icon="📊",
    layout="wide"
)

def local_css():
    st.markdown("""
        <style>
        .main { background-color: #0E1117; color: #FAFAFA; }
        div[data-testid="stMetric"] { 
            background-color: #262730; 
            padding: 1rem; 
            border-radius: 10px; 
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        div[data-testid="stMetricLabel"] { font-size: 1.1rem; color: #A0AEC0; }
        h1, h2, h3 { color: #FFFFFF; }
        .stButton>button { width: 100%; }
        .report-card {
            background-color: #1E1E1E;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 4px solid #4CAF50;
            margin-bottom: 1rem;
        }
        .risk-high { color: #ff4b4b; font-weight: bold; }
        .risk-low { color: #ffa500; font-weight: bold; }
        .risk-normal { color: #00cc66; font-weight: bold; }
        </style>
    """, unsafe_allow_html=True)

local_css()

@st.cache_data
def load_health_parameters(file_path):
    """Loads and parses the vital dataset to create a structured dictionary of health parameters."""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, encoding='latin1', engine='python', on_bad_lines='skip')
        else:
            df = pd.read_excel(file_path)
        
        df.columns = df.columns.str.strip()
        health_params = {}
        df_cleaned = df.dropna(subset=['Organ/System', 'Parameter', 'Optimal Range', 'Toxicity'])

        for _, row in df_cleaned.iterrows():
            organ = row['Organ/System']
            parameter = row['Parameter']
            unit = row['Unit']
            
            try:
                lower_bound_match = re.search(r'(\d+\.?\d*)', str(row['Optimal Range']))
                upper_bound_match = re.search(r'(\d+\.?\d*)', str(row['Toxicity']))
                if lower_bound_match and upper_bound_match:
                    lower = float(lower_bound_match.group(1))
                    upper = float(upper_bound_match.group(1))
                    if organ not in health_params:
                        health_params[organ] = {}
                    health_params[organ][parameter] = {"range": (lower, upper), "unit": unit}
            except (ValueError, TypeError):
                continue
        return health_params
    except FileNotFoundError:
        st.error(f"Dataset file not found at: {file_path}")
        return {}
    except Exception as e:
        st.error(f"An error occurred while loading the health parameter data: {e}")
        return {}

# Try to load from either file format
HEALTH_TEST_PARAMETERS = {}
if os.path.exists('vitaldataset.xlsx'):
    HEALTH_TEST_PARAMETERS = load_health_parameters('vitaldataset.xlsx')
elif os.path.exists('vitaldataset.xlsx - Sheet1.csv'):
    HEALTH_TEST_PARAMETERS = load_health_parameters('vitaldataset.xlsx - Sheet1.csv')

if not st.session_state.get('logged_in', False):
    st.error("🚫 Access Denied")
    st.info("You must be logged in to view report history.")
    if st.button("Go to Login"):
        st.switch_page("report.py")
    st.stop()

@st.cache_data
def load_user_reports(user_id):
    """Load all reports for the logged-in user from multiple possible sources."""
    all_reports = []
    
    # Try loading from extended dataset
    if os.path.exists('Common dataframe.xlsx'):
        try:
            df = pd.read_excel('Common dataframe.xlsx')
            df['Date'] = pd.to_datetime(df['Date'])
            user_data = df[df['ID'] == int(user_id)]
            if not user_data.empty:
                all_reports.append(user_data)
        except Exception as e:
            st.warning(f"Could not load extended dataset: {e}")
    
    # Try loading from dummy filled dataset
    if os.path.exists('Common dataframe.xlsx'):
        try:
            df = pd.read_excel('Common dataframe.xlsx')
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            else:
                df['Date'] = datetime.now()
            user_data = df[df['ID'] == int(user_id)]
            if not user_data.empty:
                all_reports.append(user_data)
        except Exception as e:
            st.warning(f"Could not load Common dataset: {e}")
    
    if all_reports:
        combined_df = pd.concat(all_reports, ignore_index=True)
        combined_df = combined_df.sort_values('Date', ascending=False)
        return combined_df
    
    return pd.DataFrame()

def calculate_risk_score(value, lower, upper):
    """Calculate a normalized risk score for a parameter."""
    # Handle edge cases where ranges might be zero
    if upper == 0 and lower == 0:
        return 0  # Cannot calculate risk if both bounds are zero
    
    if lower <= value <= upper:
        return 0  # Normal range
    elif value > upper:
        if upper == 0:
            return 100  # Maximum risk if upper limit is zero and value is above
        return min(((value - upper) / upper) * 100, 100)  # High risk
    else:
        if lower == 0:
            return 100  # Maximum risk if lower limit is zero and value is below
        return min(((lower - value) / lower) * 100, 100)  # Low risk

def analyze_single_report(report_row, organ_system=None):
    """Analyze a single report and return detailed metrics."""
    results = []
    organs_to_check = [organ_system] if organ_system else HEALTH_TEST_PARAMETERS.keys()
    
    for organ in organs_to_check:
        params = HEALTH_TEST_PARAMETERS.get(organ, {})
        for param, values in params.items():
            if param in report_row.index and pd.notna(report_row[param]):
                try:
                    user_value = pd.to_numeric(report_row[param], errors='coerce')
                    if pd.isna(user_value):
                        continue
                    
                    lower, upper = values["range"]
                    unit = values["unit"]
                    
                    # Validate that ranges are valid numbers
                    if not (isinstance(lower, (int, float)) and isinstance(upper, (int, float))):
                        continue
                    
                    # Skip if ranges are invalid (both zero or negative range)
                    if (lower == 0 and upper == 0) or upper < lower:
                        continue
                    
                    status = "Normal"
                    risk_score = 0
                    
                    if user_value > upper:
                        status = "High"
                        risk_score = calculate_risk_score(user_value, lower, upper)
                    elif user_value < lower:
                        status = "Low"
                        risk_score = calculate_risk_score(user_value, lower, upper)
                    
                    results.append({
                        "organ": organ,
                        "parameter": param,
                        "value": user_value,
                        "unit": unit,
                        "lower": lower,
                        "upper": upper,
                        "status": status,
                        "risk_score": risk_score
                    })
                except (ValueError, TypeError, ZeroDivisionError) as e:
                    # Skip parameters that cause errors
                    continue
    
    return pd.DataFrame(results)

def create_comparison_chart(user_reports, parameter, organ_system):
    """Create a trend comparison chart for a specific parameter."""
    if parameter not in user_reports.columns:
        return None
    
    # Convert to numeric and drop NaN values
    plot_data = user_reports[['Date', parameter]].copy()
    plot_data[parameter] = pd.to_numeric(plot_data[parameter], errors='coerce')
    plot_data = plot_data.dropna()
    
    if plot_data.empty:
        return None
    
    # Get normal range
    params = HEALTH_TEST_PARAMETERS.get(organ_system, {})
    param_info = params.get(parameter, {})
    lower, upper = param_info.get("range", (0, 0))
    unit = param_info.get("unit", "")
    
    # Skip if invalid range
    if lower == 0 and upper == 0:
        return None
    
    fig = go.Figure()
    
    # Add actual values line
    fig.add_trace(go.Scatter(
        x=plot_data['Date'],
        y=plot_data[parameter],
        mode='lines+markers',
        name='Your Values',
        line=dict(color='#4CAF50', width=3),
        marker=dict(size=10)
    ))
    
    # Add normal range bands only if valid
    if upper > 0:
        fig.add_trace(go.Scatter(
            x=plot_data['Date'],
            y=[upper] * len(plot_data),
            mode='lines',
            name='Upper Normal',
            line=dict(color='#ff4b4b', width=1, dash='dash')
        ))
    
    if lower >= 0:
        fig.add_trace(go.Scatter(
            x=plot_data['Date'],
            y=[lower] * len(plot_data),
            mode='lines',
            name='Lower Normal',
            line=dict(color='#ffa500', width=1, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(76, 175, 80, 0.1)'
        ))
    
    fig.update_layout(
        title=f'{parameter} Trend Over Time',
        xaxis_title='Date',
        yaxis_title=f'Value ({unit})',
        template='plotly_dark',
        hovermode='x unified',
        height=400
    )
    
    return fig

# Main UI
st.title(f"📊 Report History for {st.session_state.get('name', 'User')}")
st.markdown("---")

# Load user reports
user_reports = load_user_reports(st.session_state.username)

if user_reports.empty:
    st.info("📭 No previous reports found. Upload your first report to get started!")
    st.markdown("---")
    if st.button("➕ Upload New Report"):
        st.switch_page("pages/_evaluate_Report.py")
else:
    # Summary Statistics
    st.header("📈 Overall Health Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Reports",
            value=len(user_reports),
            delta=f"Last: {user_reports['Date'].iloc[0].strftime('%d %b %Y')}"
        )
    
    with col2:
        # Calculate average risk across all parameters in latest report
        latest_analysis = analyze_single_report(user_reports.iloc[0])
        if not latest_analysis.empty:
            avg_risk = latest_analysis[latest_analysis['status'] != 'Normal']['risk_score'].mean()
            risk_count = len(latest_analysis[latest_analysis['status'] != 'Normal'])
            st.metric(
                label="Risk Parameters",
                value=risk_count,
                delta=f"{avg_risk:.1f}% Avg Risk" if risk_count > 0 else "All Normal"
            )
        else:
            st.metric(label="Risk Parameters", value=0, delta="All Normal")
    
    with col3:
        date_range = (user_reports['Date'].max() - user_reports['Date'].min()).days
        st.metric(
            label="Tracking Period",
            value=f"{date_range} days",
            delta=f"Since {user_reports['Date'].min().strftime('%b %Y')}"
        )
    
    with col4:
        # Count improving vs worsening trends
        improving = 0
        if len(user_reports) > 1:
            st.metric(
                label="Report Frequency",
                value=f"{len(user_reports)/max(date_range/30, 1):.1f}/month",
                delta="Keep tracking!"
            )
        else:
            st.metric(label="Report Frequency", value="First Report", delta="Start tracking")
    
    st.markdown("---")
    
    # Report Selection and Analysis
    st.header("🔍 Detailed Report Analysis")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Report List", "📊 Trend Analysis", "⚠️ Risk Analysis", "📈 Comparative View"])
    
    with tab1:
        st.subheader("Your Report History")
        
        for idx, (_, report) in enumerate(user_reports.iterrows()):
            with st.expander(f"Report #{len(user_reports)-idx} - {report['Date'].strftime('%d %B %Y')}", expanded=(idx==0)):
                analysis = analyze_single_report(report)
                
                if not analysis.empty:
                    # Show summary
                    risk_params = analysis[analysis['status'] != 'Normal']
                    normal_params = analysis[analysis['status'] == 'Normal']
                    
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Total Parameters", len(analysis))
                    col_b.metric("Normal", len(normal_params), delta="✓", delta_color="normal")
                    col_c.metric("Needs Attention", len(risk_params), delta="⚠", delta_color="inverse")
                    
                    # Show risk parameters if any
                    if not risk_params.empty:
                        st.markdown("**⚠️ Parameters Requiring Attention:**")
                        risk_params_sorted = risk_params.sort_values('risk_score', ascending=False)
                        
                        cols = st.columns(min(len(risk_params_sorted), 3))
                        for i, (_, param) in enumerate(risk_params_sorted.iterrows()):
                            col_idx = i % 3
                            with cols[col_idx]:
                                status_class = "risk-high" if param['status'] == 'High' else "risk-low"
                                st.markdown(f"**{param['parameter']}**")
                                st.markdown(f"<span class='{status_class}'>{param['value']:.2f} {param['unit']}</span>", unsafe_allow_html=True)
                                st.caption(f"Normal: {param['lower']}-{param['upper']} {param['unit']}")
                else:
                    st.info("No analyzable parameters found in this report.")
    
    with tab2:
        st.subheader("Parameter Trend Analysis")
        
        if len(user_reports) > 1:
            # Select organ system
            organ_select = st.selectbox("Select Organ/System", options=list(HEALTH_TEST_PARAMETERS.keys()), key="trend_organ")
            
            # Get available parameters for this organ
            available_params = []
            for param in HEALTH_TEST_PARAMETERS.get(organ_select, {}).keys():
                if param in user_reports.columns and user_reports[param].notna().any():
                    available_params.append(param)
            
            if available_params:
                selected_params = st.multiselect(
                    "Select Parameters to Visualize",
                    options=available_params,
                    default=available_params[:3] if len(available_params) >= 3 else available_params
                )
                
                if selected_params:
                    for param in selected_params:
                        fig = create_comparison_chart(user_reports, param, organ_select)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning(f"No data available for {param}")
                else:
                    st.info("Select at least one parameter to view trends.")
            else:
                st.info(f"No historical data available for {organ_select} parameters.")
        else:
            st.info("Upload at least 2 reports to see trend analysis.")
    
    with tab3:
        st.subheader("Risk Factor Analysis")
        
        # Analyze latest report for risks
        latest_analysis = analyze_single_report(user_reports.iloc[0])
        high_risk = latest_analysis[latest_analysis['status'] != 'Normal'].sort_values('risk_score', ascending=False)
        
        if not high_risk.empty:
            # Create risk heatmap
            fig = px.bar(
                high_risk.head(10),
                x='risk_score',
                y='parameter',
                orientation='h',
                color='status',
                color_discrete_map={'High': '#ff4b4b', 'Low': '#ffa500'},
                title='Top 10 Risk Factors in Latest Report',
                labels={'risk_score': 'Risk Score (%)', 'parameter': 'Parameter'}
            )
            fig.update_layout(template='plotly_dark', height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed risk breakdown
            st.markdown("---")
            st.markdown("**Detailed Risk Breakdown:**")
            
            for organ in high_risk['organ'].unique():
                organ_risks = high_risk[high_risk['organ'] == organ]
                with st.expander(f"{organ} ({len(organ_risks)} parameters)", expanded=True):
                    for _, risk in organ_risks.iterrows():
                        col1, col2, col3 = st.columns([2, 1, 2])
                        with col1:
                            st.markdown(f"**{risk['parameter']}**")
                        with col2:
                            status_class = "risk-high" if risk['status'] == 'High' else "risk-low"
                            st.markdown(f"<span class='{status_class}'>{risk['value']:.2f} {risk['unit']}</span>", unsafe_allow_html=True)
                        with col3:
                            st.caption(f"Normal: {risk['lower']}-{risk['upper']} {risk['unit']}")
        else:
            st.success("✅ Excellent! All parameters are within normal range in your latest report.")
    
    with tab4:
        st.subheader("Comparative Analysis Across Reports")
        
        if len(user_reports) > 1:
            # Select parameter for comparison
            all_params = []
            for organ_params in HEALTH_TEST_PARAMETERS.values():
                all_params.extend(organ_params.keys())
            
            available_for_comparison = [p for p in all_params if p in user_reports.columns and user_reports[p].notna().sum() > 1]
            
            if available_for_comparison:
                compare_param = st.selectbox("Select Parameter for Comparison", options=available_for_comparison)
                
                # Create comparison dataframe
                compare_data = user_reports[['Date', compare_param]].dropna()
                compare_data['Report_Number'] = range(len(compare_data), 0, -1)
                
                # Find which organ this parameter belongs to
                param_organ = None
                for organ, params in HEALTH_TEST_PARAMETERS.items():
                    if compare_param in params:
                        param_organ = organ
                        break
                
                if param_organ:
                    param_info = HEALTH_TEST_PARAMETERS[param_organ][compare_param]
                    lower, upper = param_info["range"]
                    
                    # Create comparative visualization
                    fig = go.Figure()
                    
                    # Add bar chart
                    colors = ['#ff4b4b' if val > upper or val < lower else '#4CAF50' 
                             for val in compare_data[compare_param]]
                    
                    fig.add_trace(go.Bar(
                        x=compare_data['Date'],
                        y=compare_data[compare_param],
                        marker_color=colors,
                        name='Values',
                        text=compare_data[compare_param].round(2),
                        textposition='outside'
                    ))
                    
                    # Add reference lines
                    fig.add_hline(y=upper, line_dash="dash", line_color="#ff4b4b", 
                                 annotation_text="Upper Limit", annotation_position="right")
                    fig.add_hline(y=lower, line_dash="dash", line_color="#ffa500", 
                                 annotation_text="Lower Limit", annotation_position="right")
                    
                    fig.update_layout(
                        title=f'{compare_param} Across All Reports',
                        xaxis_title='Report Date',
                        yaxis_title=f'Value ({param_info["unit"]})',
                        template='plotly_dark',
                        showlegend=False,
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistics
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Average", f"{compare_data[compare_param].mean():.2f}")
                    col2.metric("Minimum", f"{compare_data[compare_param].min():.2f}")
                    col3.metric("Maximum", f"{compare_data[compare_param].max():.2f}")
                    col4.metric("Std Dev", f"{compare_data[compare_param].std():.2f}")
            else:
                st.info("Not enough data points for comparison analysis.")
        else:
            st.info("Upload at least 2 reports for comparative analysis.")

st.markdown("---")

# Navigation
col_nav1, col_nav2, col_nav3 = st.columns([0.6, 0.2, 0.2])
with col_nav1:
    if st.button("⬅️ Back to Dashboard"):
        st.switch_page("pages/_dashboard.py")
with col_nav2:
    if st.button("➕ Upload New Report"):
        st.switch_page("pages/_evaluate_Report.py")
with col_nav3:
    if st.button("Logout", use_container_width=True, type="primary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.switch_page("report.py")