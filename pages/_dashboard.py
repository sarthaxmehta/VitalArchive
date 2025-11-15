import streamlit as st
import pandas as pd
import os
import re
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Health Dashboard",
    page_icon="🩺",
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
            transition: transform 0.2s;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
        }
        div[data-testid="stMetricLabel"] { font-size: 1.1rem; color: #A0AEC0; }
        h1, h2, h3 { color: #FFFFFF; }
        .stButton>button { width: 100%; }
        .risk-delta {
            color: #ff4b4b !important;
            font-size: 1rem;
            font-weight: 600;
            text-align: center;
        }
        .health-score {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            padding: 2rem;
            border-radius: 50%;
            width: 150px;
            height: 150px;
            margin: auto;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .score-excellent { background: linear-gradient(135deg, #00c853, #4caf50); color: white; }
        .score-good { background: linear-gradient(135deg, #4caf50, #8bc34a); color: white; }
        .score-fair { background: linear-gradient(135deg, #ffc107, #ff9800); color: white; }
        .score-poor { background: linear-gradient(135deg, #ff5722, #f44336); color: white; }
        .insight-card {
            background: linear-gradient(135deg, #1e3c72, #2a5298);
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
            border-left: 4px solid #4CAF50;
        }
        </style>
    """, unsafe_allow_html=True)

local_css()

VITAL_DATASET_FILE = 'vitaldataset.xlsx'
PATIENT_DATA_FILE = 'Common dataframe.xlsx'

@st.cache_data
def load_health_parameters(file_path):
    """Loads and parses the vital dataset to create a structured dictionary of health parameters."""
    try:
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
        st.error(f"Vital dataset file not found at: {file_path}")
        return {}
    except Exception as e:
        st.error(f"An error occurred while loading health parameters: {e}")
        return {}

@st.cache_data
def load_patient_data(file_path):
    """Loads the common patient dataframe and prepares it for analysis."""
    try:
        df = pd.read_excel(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Convert all parameter columns to numeric where possible
        for col in df.columns:
            if col not in ['ID', 'Date', 'Name']:
                df[col] = pd.to_numeric(df[col], errors='ignore')
        
        return df
    except FileNotFoundError:
        st.error(f"Patient data file not found at: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred while loading patient data: {e}")
        return pd.DataFrame()

HEALTH_PARAMETERS = load_health_parameters(VITAL_DATASET_FILE)
PATIENT_DATA = load_patient_data(PATIENT_DATA_FILE)

if not st.session_state.get('logged_in', False):
    st.error("🚫 Access Denied")
    st.info("You must be logged in to view the dashboard.")
    if st.button("Go to Login"):
        st.switch_page("report.py")
    st.stop()

def calculate_health_score(user_id):
    """Calculate an overall health score based on latest report."""
    user_df = PATIENT_DATA[PATIENT_DATA['ID'] == int(user_id)]
    if user_df.empty:
        return None, "No Data"

    latest_report = user_df.loc[user_df['Date'].idxmax()]
    total_params = 0
    normal_params = 0

    for organ, params in HEALTH_PARAMETERS.items():
        for param, values in params.items():
            if param in latest_report and pd.notna(latest_report[param]):
                user_value = pd.to_numeric(latest_report[param], errors='coerce')
                if pd.isna(user_value): continue

                total_params += 1
                lower, upper = values['range']
                if lower <= user_value <= upper:
                    normal_params += 1

    if total_params == 0:
        return None, "No Data"
    
    score = int((normal_params / total_params) * 100)
    
    if score >= 90:
        category = "Excellent"
    elif score >= 75:
        category = "Good"
    elif score >= 60:
        category = "Fair"
    else:
        category = "Needs Attention"
    
    return score, category

def get_latest_risk_factors(user_id):
    """Analyzes the latest report for a user and returns the top 3 most critical risk factors."""
    user_df = PATIENT_DATA[PATIENT_DATA['ID'] == int(user_id)]
    if user_df.empty:
        return []

    latest_report = user_df.loc[user_df['Date'].idxmax()]
    high_risk_factors = []

    for organ, params in HEALTH_PARAMETERS.items():
        for param, values in params.items():
            if param in latest_report and pd.notna(latest_report[param]):
                user_value = pd.to_numeric(latest_report[param], errors='coerce')
                if pd.isna(user_value): continue

                lower, upper = values['range']
                if not (lower <= user_value <= upper):
                    risk_score = abs((user_value - upper) / upper) if user_value > upper else abs((user_value - lower) / lower)
                    status = "High" if user_value > upper else "Low"
                    delta_text = f"{status} (Normal: {lower}-{upper})"
                    high_risk_factors.append({
                        "param": param, "value": user_value, "unit": values['unit'],
                        "delta_text": delta_text, "risk_score": risk_score, "organ": organ
                    })
    
    high_risk_factors.sort(key=lambda x: x['risk_score'], reverse=True)
    return high_risk_factors[:3]

def get_health_insights(user_id):
    """Generate personalized health insights based on data trends."""
    user_df = PATIENT_DATA[PATIENT_DATA['ID'] == int(user_id)]
    if len(user_df) < 2:
        return ["Upload more reports to get personalized health insights and trends!"]
    
    insights = []
    user_df_sorted = user_df.sort_values('Date')
    
    # Check for improving/worsening trends
    for organ, params in HEALTH_PARAMETERS.items():
        for param, values in params.items():
            if param in user_df.columns and user_df[param].notna().sum() >= 2:
                try:
                    # Convert to numeric, coercing errors to NaN
                    recent_values = pd.to_numeric(user_df_sorted[param], errors='coerce').dropna().tail(3)
                    
                    if len(recent_values) >= 2:
                        trend = recent_values.iloc[-1] - recent_values.iloc[0]
                        lower, upper = values['range']
                        latest_value = recent_values.iloc[-1]
                        
                        if trend > 0 and latest_value > upper:
                            insights.append(f"⚠️ {param} is trending upward and above normal range. Consider consulting your physician.")
                        elif trend < 0 and latest_value < lower:
                            insights.append(f"⚠️ {param} is trending downward and below normal range. Monitor closely.")
                        elif abs(trend) > (upper - lower) * 0.3:
                            direction = "increasing" if trend > 0 else "decreasing"
                            insights.append(f"📊 {param} is {direction} significantly. Keep tracking this parameter.")
                except (ValueError, TypeError, AttributeError):
                    # Skip parameters that can't be converted to numeric
                    continue
    
    # Check report frequency
    try:
        date_diff = (user_df['Date'].max() - user_df['Date'].min()).days
        if date_diff > 90 and len(user_df) < 3:
            insights.append("📅 Consider getting health checkups more frequently for better tracking.")
    except:
        pass
    
    if not insights:
        insights.append("✅ Your health parameters are stable. Keep up the good work!")
    
    return insights[:5]  # Return top 5 insights

def create_organ_health_radar(user_id):
    """Create a radar chart showing health status by organ system."""
    user_df = PATIENT_DATA[PATIENT_DATA['ID'] == int(user_id)]
    if user_df.empty:
        return None

    latest_report = user_df.loc[user_df['Date'].idxmax()]
    organ_scores = {}

    for organ, params in HEALTH_PARAMETERS.items():
        total = 0
        normal = 0
        for param, values in params.items():
            if param in latest_report and pd.notna(latest_report[param]):
                try:
                    user_value = pd.to_numeric(latest_report[param], errors='coerce')
                    if pd.isna(user_value): 
                        continue
                    
                    total += 1
                    lower, upper = values['range']
                    if lower <= user_value <= upper:
                        normal += 1
                except (ValueError, TypeError):
                    continue
        
        if total > 0:
            organ_scores[organ] = (normal / total) * 100

    if not organ_scores:
        return None

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=list(organ_scores.values()),
        theta=list(organ_scores.keys()),
        fill='toself',
        name='Health Score',
        line_color='#4CAF50'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        template='plotly_dark',
        title="Health Status by Organ System",
        height=400
    )

    return fig

# Main Dashboard UI
st.title(f"🩺 Health Dashboard - {st.session_state.get('name', 'User')}")

# Calculate overall health score
health_score, score_category = calculate_health_score(st.session_state.username)

# Top section with health score and quick stats
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    if health_score is not None:
        score_class = {
            "Excellent": "score-excellent",
            "Good": "score-good",
            "Fair": "score-fair",
            "Needs Attention": "score-poor"
        }.get(score_category, "score-fair")
        
        st.markdown(f"""
            <div class='health-score {score_class}'>
                {health_score}
            </div>
            <h3 style='text-align: center; margin-top: 1rem;'>{score_category}</h3>
        """, unsafe_allow_html=True)
    else:
        st.info("Upload your first report to see your health score!")

with col2:
    # Organ system health radar
    radar_fig = create_organ_health_radar(st.session_state.username)
    if radar_fig:
        st.plotly_chart(radar_fig, use_container_width=True)
    else:
        st.info("Radar chart will appear after uploading reports")

with col3:
    user_df = PATIENT_DATA[PATIENT_DATA['ID'] == int(st.session_state.username)]
    if not user_df.empty:
        st.metric("Total Reports", len(user_df))
        st.metric("Last Updated", user_df['Date'].max().strftime("%d %b %Y"))
        days_since = (datetime.now() - user_df['Date'].max()).days
        st.metric("Days Since Last Report", days_since)
    else:
        st.metric("Total Reports", 0)
        st.info("No reports yet")

st.markdown("---")

# Critical Risk Factors Section
st.header("⚠️ Top 3 Critical Risk Factors")
top_risk_factors = get_latest_risk_factors(st.session_state.username)

if not top_risk_factors:
    st.success("✅ No high-risk values found in your most recent report. Great job!")
else:
    cols = st.columns(len(top_risk_factors))
    for i, metric_data in enumerate(top_risk_factors):
        with cols[i]:
            st.metric(
                label=f"{metric_data['param']} ({metric_data['organ']})", 
                value=f"{metric_data['value']:.2f} {metric_data['unit']}"
            )
            st.markdown(f"<p class='risk-delta'>{metric_data['delta_text']}</p>", unsafe_allow_html=True)

st.markdown("---")

# Health Insights Section
st.header("💡 Personalized Health Insights")
insights = get_health_insights(st.session_state.username)

for insight in insights:
    st.markdown(f"<div class='insight-card'>{insight}</div>", unsafe_allow_html=True)

st.markdown("---")

# Historical Parameter Trends
st.header("📈 Historical Parameter Trends")
user_df = PATIENT_DATA[PATIENT_DATA['ID'] == int(st.session_state.username)]

if user_df.empty:
    st.info("No historical data found. Upload your first report to start tracking!")
    col_upload = st.columns([1, 2, 1])[1]
    with col_upload:
        if st.button("➕ Upload First Report", use_container_width=True, type="primary"):
            st.switch_page("pages/_evaluate_Report.py")
else:
    vital_types = {organ: list(params.keys()) for organ, params in HEALTH_PARAMETERS.items()}
    
    col_select1, col_select2 = st.columns(2)
    
    with col_select1:
        selected_organ = st.selectbox("Select an Organ/System", options=list(vital_types.keys()))
    
    params_in_type = vital_types.get(selected_organ, [])
    available_params = [p for p in params_in_type if p in user_df.columns and user_df[p].notna().any()]
    
    if not available_params:
        st.warning(f"No data available for '{selected_organ}' parameters in your reports.")
    else:
        with col_select2:
            view_mode = st.radio("View Mode", ["Grid View", "Stacked View"], horizontal=True)
        
        st.markdown("**Select parameters to display:**")
        selected_params = st.multiselect(
            "Parameters",
            options=available_params,
            default=available_params[:3] if len(available_params) >= 3 else available_params,
            label_visibility="collapsed"
        )
        
        st.markdown("---")

        if selected_params:
            if view_mode == "Grid View":
                NUM_COLUMNS = 3
                for i in range(0, len(selected_params), NUM_COLUMNS):
                    row_params = selected_params[i : i + NUM_COLUMNS]
                    cols = st.columns(NUM_COLUMNS)
                    
                    for j, param in enumerate(row_params):
                        with cols[j]:
                            plot_df = user_df[['Date', param]].dropna()
                            if not plot_df.empty:
                                # Get normal range for reference lines
                                param_info = HEALTH_PARAMETERS.get(selected_organ, {}).get(param, {})
                                lower, upper = param_info.get('range', (None, None))
                                unit = param_info.get('unit', '')
                                
                                fig = go.Figure()
                                
                                # Add actual values
                                fig.add_trace(go.Scatter(
                                    x=plot_df['Date'], 
                                    y=plot_df[param],
                                    mode='lines+markers',
                                    name=param,
                                    line=dict(color='#4CAF50', width=2),
                                    marker=dict(size=8)
                                ))
                                
                                # Add reference lines
                                if lower and upper:
                                    fig.add_hline(y=upper, line_dash="dash", line_color="#ff4b4b", 
                                                annotation_text="Upper", annotation_position="right")
                                    fig.add_hline(y=lower, line_dash="dash", line_color="#ffa500", 
                                                annotation_text="Lower", annotation_position="right")
                                
                                fig.update_layout(
                                    title=param,
                                    xaxis_title="Date", 
                                    yaxis_title=f"Value ({unit})",
                                    template='plotly_dark',
                                    height=300,
                                    showlegend=False
                                )
                                st.plotly_chart(fig, use_container_width=True)
            else:  # Stacked View
                for param in selected_params:
                    plot_df = user_df[['Date', param]].dropna()
                    if not plot_df.empty:
                        param_info = HEALTH_PARAMETERS.get(selected_organ, {}).get(param, {})
                        lower, upper = param_info.get('range', (None, None))
                        unit = param_info.get('unit', '')
                        
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=plot_df['Date'], 
                            y=plot_df[param],
                            mode='lines+markers',
                            name=param,
                            line=dict(color='#4CAF50', width=3),
                            marker=dict(size=10),
                            fill='tozeroy',
                            fillcolor='rgba(76, 175, 80, 0.1)'
                        ))
                        
                        if lower and upper:
                            fig.add_hline(y=upper, line_dash="dash", line_color="#ff4b4b", 
                                        annotation_text="Upper Normal", annotation_position="right")
                            fig.add_hline(y=lower, line_dash="dash", line_color="#ffa500", 
                                        annotation_text="Lower Normal", annotation_position="right")
                        
                        fig.update_layout(
                            title=f'{param} Trend Analysis',
                            xaxis_title="Date", 
                            yaxis_title=f"Value ({unit})",
                            template='plotly_dark',
                            height=400,
                            showlegend=False,
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Please select at least one parameter to display its trend.")

st.markdown("---")

# Quick Actions Section
st.header("Quick Actions")
col_act1, col_act2, col_act3, col_act4 = st.columns(4)

with col_act1:
    if st.button("📊 View Full History", use_container_width=True):
        st.switch_page("pages/_Report_History.py")

with col_act2:
    if st.button("📝 Analyze New Report", use_container_width=True):
        st.switch_page("pages/_evaluate_Report.py")

with col_act3:
    if st.button("📄 Export Data", use_container_width=True):
        if not user_df.empty:
            csv = user_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"health_data_{st.session_state.username}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No data to export")

with col_act4:
    if st.button("🚪 Logout", use_container_width=True, type="primary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.switch_page("report.py")