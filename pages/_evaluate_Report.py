import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import plotly.graph_objects as go
from datetime import datetime
from tempfile import NamedTemporaryFile
import medical_parser as lib  # <-- ADDED: Import your library

st.set_page_config(
    page_title="Analyze Health Report",
    page_icon="🔬",
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
        h1, h2 { color: #FFFFFF; }
        .stButton>button { width: 100%; }
        .risk-delta {
            color: #ff4b4b !important;
            font-size: 1rem;
            font-weight: 600;
        }
        .success-banner {
            background: linear-gradient(135deg, #00c853, #4caf50);
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
            color: white;
            font-size: 1.2rem;
            margin: 1rem 0;
        }
        .warning-banner {
            background: linear-gradient(135deg, #ff6f00, #ff9800);
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
            color: white;
            font-size: 1.2rem;
            margin: 1rem 0;
        }
        .analysis-card {
            background-color: #1E1E1E;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
            border-left: 4px solid #4CAF50;
        }
        .upload-section {
            background-color: #1a1a2e;
            padding: 2rem;
            border-radius: 15px;
            border: 2px dashed #4CAF50;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

local_css()


# Replace the initialization section in _evaluate_Report.py (around lines 38-55)

# --- Configuration ---
HEALTH_PARAMS_FILE = 'vitaldataset.xlsx'
COMMON_DATABASE_FILE = 'Common dataframe.xlsx'  # Changed to relative path

# Get the model path from secrets or use a default
try:
    NORMALIZATION_MODEL_PATH = st.secrets.get("NORMALIZATION_MODEL_PATH", "vitalarchive_model2")
except:
    NORMALIZATION_MODEL_PATH = "vitalarchive_model2"

# --- Model Initialization ---
@st.cache_resource
def initialize_models():
    """
    Loads and initializes the Gemini and SentenceTransformer models.
    """
    try:
        # Check if API key exists
        if "GEMINI_API_KEY" not in st.secrets:
            st.error("⚠️ Configuration Error: GEMINI_API_KEY not found in secrets.")
            st.info("""
            **To fix this:**
            1. Create a file `.streamlit/secrets.toml` in your project directory
            2. Add your API key:
               ```
               GEMINI_API_KEY = "your-api-key-here"
               ```
            3. Restart the Streamlit app
            """)
            return False
        
        api_key = st.secrets["GEMINI_API_KEY"]
        
        # Check if model path exists
        if not os.path.exists(NORMALIZATION_MODEL_PATH):
            st.error(f"⚠️ Model Path Error: Directory '{NORMALIZATION_MODEL_PATH}' not found.")
            st.info("""
            **To fix this:**
            1. Ensure the vitalarchive_model2 folder is in your project directory
            2. Or update the path in .streamlit/secrets.toml:
               ```
               NORMALIZATION_MODEL_PATH = "/path/to/your/model"
               ```
            """)
            return False
        
        # Initialize the models
        with st.spinner("Loading AI models... This may take a moment."):
            lib.initialize(
                api_key=api_key,
                normalization_model_path=NORMALIZATION_MODEL_PATH
            )
        
        st.success("✅ Models initialized successfully!")
        return True
        
    except Exception as e:
        st.error(f"❌ Error initializing models: {str(e)}")
        st.exception(e)  # This will show the full traceback
        return False
# --- Data Loading Functions ---
@st.cache_data
def load_health_parameters(file_path):
    """Loads and parses the vital dataset Excel file."""
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
        st.error(f"Dataset file not found at: {file_path}")
        return {}
    except Exception as e:
        st.error(f"An error occurred while loading the health parameter data: {e}")
        return {}

HEALTH_TEST_PARAMETERS = load_health_parameters(HEALTH_PARAMS_FILE)

# --- Login Check ---
if not st.session_state.get('logged_in', False):
    st.error("🚫 Access Denied")
    st.info("You must be logged in to analyze a report.")
    if st.button("Go to Login"):
        st.switch_page("report.py")
    st.stop()

# --- Analysis Functions ---
def analyze_report(df_single_row, test_type):
    """Analyzes a single row of a dataframe for high-risk values."""
    params = HEALTH_TEST_PARAMETERS.get(test_type)
    if not params:
        return None

    results = {}
    for param, values in params.items():
        if param in df_single_row.columns:
            try:
                user_value = pd.to_numeric(df_single_row[param].iloc[0])
            except (ValueError, TypeError):
                continue
            
            if pd.isna(user_value): continue
            
            lower, upper = values["range"]
            unit = values["unit"]
            status = "Normal"
            delta_text = ""
            deviation_percent = 0

            if user_value > upper:
                status = "High"
                deviation_percent = ((user_value - upper) / upper) * 100
                delta_text = f"↑ High by {deviation_percent:.1f}% (Normal: {lower}-{upper})"
            elif user_value < lower:
                status = "Low"
                deviation_percent = ((lower - user_value) / lower) * 100
                delta_text = f"↓ Low by {deviation_percent:.1f}% (Normal: {lower}-{upper})"
            else:
                delta_text = f"✓ Normal (Range: {lower}-{upper})"
            
            results[param] = {
                "value": user_value, 
                "status": status, 
                "unit": unit, 
                "delta_text": delta_text,
                "deviation": deviation_percent,
                "lower": lower,
                "upper": upper
            }
    
    return results

def create_comparison_gauge(param_name, value, lower, upper, unit):
    """Create a gauge chart showing parameter status."""
    # Determine the range for the gauge
    range_span = upper - lower
    gauge_min = lower - (range_span * 0.2)
    gauge_max = upper + (range_span * 0.2)
    
    # Determine color based on value
    if value < lower:
        color = "#ffa500"  # Orange for low
    elif value > upper:
        color = "#ff4b4b"  # Red for high
    else:
        color = "#4CAF50"  # Green for normal
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number", # Removed 'delta' as it's redundant with text
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': param_name, 'font': {'size': 16, 'color': 'white'}},
        number = {'suffix': f" {unit}", 'font': {'size': 20, 'color': 'white'}},
        gauge = {
            'axis': {'range': [gauge_min, gauge_max], 'tickcolor': "white"},
            'bar': {'color': color},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "white",
            'steps': [
                {'range': [gauge_min, lower], 'color': 'rgba(255, 165, 0, 0.3)'},
                {'range': [lower, upper], 'color': 'rgba(76, 175, 80, 0.3)'},
                {'range': [upper, gauge_max], 'color': 'rgba(255, 75, 75, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor = "rgba(0,0,0,0)",
        font = {'color': "white", 'family': "Arial"},
        height = 250,
        margin = dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

# --- Main UI ---
st.title(f"🔬 Health Report Analysis for {st.session_state.get('name', 'User')}")
st.markdown("---")

# Initialize models
models_loaded = initialize_models()

# Create tabs for different upload methods
tab1, tab2 = st.tabs(["📤 Upload PDF Report", "📋 Quick Manual Entry"])

# ==============================================================================
# TAB 1: PDF UPLOAD (This section is REPLACED)
# ==============================================================================
with tab1:
    if not models_loaded:
        st.error("PDF evaluation is disabled. Please check app configuration.")
    else:
        st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
        st.header("Upload PDF Health Report")
        st.info("Upload your medical report (PDF). The system will use AI to extract, normalize, and save the data to your health history.")
        
        uploaded_file = st.file_uploader(
            "Choose your PDF report file", 
            type="pdf",
            help="Upload a PDF file containing your health parameters"
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
        if uploaded_file is not None:
            st.info(f"File '{uploaded_file.name}' uploaded. Click 'Evaluate' to process.")
            
            if st.button("Evaluate PDF Report", type="primary", use_container_width=True):
                
                # Save to a temporary file to get a stable path
                with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    temp_pdf_path = tmp.name
                
                new_report_df = None  # Initialize variable
                try:
                    with st.spinner("Evaluating report... This may take a moment."):
                        # Call your library to get the path to the *new* CSV
                        output_csv_path = lib.valuate(temp_pdf_path)
                        
                        # Read the CSV created by the library
                        new_report_df = pd.read_csv(output_csv_path)
                        
                        # Add user ID and ensure Date exists
                        new_report_df['ID'] = st.session_state.get('username', 'unknown_user')
                        if 'Date' not in new_report_df.columns or pd.isna(new_report_df['Date'].iloc[0]):
                             new_report_df['Date'] = datetime.now()

                    # --- Save to common database (logic from your original script) ---
                    if os.path.exists(COMMON_DATABASE_FILE):
                        common_db_df = pd.read_excel(COMMON_DATABASE_FILE)
                        updated_db_df = pd.concat([common_db_df, new_report_df], ignore_index=True)
                    else:
                        updated_db_df = new_report_df

                    updated_db_df.to_excel(COMMON_DATABASE_FILE, index=False)
                    # --- End save logic ---
                    
                    st.markdown("<div class='success-banner'>✅ Report Successfully Uploaded, Parsed, and Saved!</div>", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"An error occurred during PDF evaluation: {e}")
                
                finally:
                    # Clean up the temporary file
                    if os.path.exists(temp_pdf_path):
                        os.remove(temp_pdf_path)
                    # Clean up the intermediate CSV if it exists
                    if 'output_csv_path' in locals() and os.path.exists(output_csv_path):
                        os.remove(output_csv_path)

                # --- Run Analysis (logic from your original script) ---
                if new_report_df is not None:
                    st.markdown("---")
                    st.header("📊 Comprehensive Report Analysis")
                    
                    col_test1, col_test2 = st.columns([2, 1])
                    with col_test1:
                        test_type = st.selectbox(
                            "Select Organ/System to Analyze", 
                            options=list(HEALTH_TEST_PARAMETERS.keys()),
                            help="Choose which organ system you want to analyze from your report"
                        )
                    
                    with col_test2:
                        view_style = st.radio("View Style", ["Detailed", "Compact"], horizontal=True)

                    analysis_results = analyze_report(new_report_df, test_type)
                    
                    if analysis_results:
                        high_risk_metrics = {param: result for param, result in analysis_results.items() if result['status'] != 'Normal'}
                        normal_metrics = {param: result for param, result in analysis_results.items() if result['status'] == 'Normal'}
                        
                        total_params = len(analysis_results)
                        normal_count = len(normal_metrics)
                        risk_count = len(high_risk_metrics)
                        health_percentage = (normal_count / total_params) * 100 if total_params > 0 else 0
                        
                        st.subheader("Analysis Summary")
                        col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
                        
                        col_sum1.metric("Total Parameters", total_params)
                        col_sum2.metric("Normal Values", normal_count, delta="✓")
                        col_sum3.metric("Needs Attention", risk_count, delta="⚠" if risk_count > 0 else "✓")
                        col_sum4.metric("Health Score", f"{health_percentage:.0f}%")
                        
                        st.markdown("---")
                        
                        if high_risk_metrics:
                            st.markdown("<div class='warning-banner'>⚠️ Parameters Requiring Attention</div>", unsafe_allow_html=True)
                            
                            if view_style == "Detailed":
                                num_cols = 3
                                risk_items = list(high_risk_metrics.items())
                                
                                for i in range(0, len(risk_items), num_cols):
                                    cols = st.columns(num_cols)
                                    for j in range(num_cols):
                                        if i + j < len(risk_items):
                                            param, result = risk_items[i + j]
                                            with cols[j]:
                                                gauge_fig = create_comparison_gauge(
                                                    param, 
                                                    result['value'], 
                                                    result['lower'], 
                                                    result['upper'], 
                                                    result['unit']
                                                )
                                                st.plotly_chart(gauge_fig, use_container_width=True)
                                                st.markdown(f"<p class='risk-delta'>{result['delta_text']}</p>", unsafe_allow_html=True)
                            else: # Compact view
                                METRICS_PER_ROW = 4
                                params_to_display = list(high_risk_metrics.items())
                                
                                for i in range(0, len(params_to_display), METRICS_PER_ROW):
                                    chunk = params_to_display[i:i + METRICS_PER_ROW]
                                    cols = st.columns(METRICS_PER_ROW)
                                    
                                    for j, (param, result) in enumerate(chunk):
                                        with cols[j]:
                                            st.metric(
                                                label=param,
                                                value=f"{result['value']:.2f} {result['unit']}",
                                            )
                                            st.markdown(f"<p class='risk-delta'>{result['delta_text']}</p>", unsafe_allow_html=True)
                            
                            st.markdown("---")
                            st.subheader("💡 Recommendations")
                            st.markdown("<div class='analysis-card'>", unsafe_allow_html=True)
                            st.markdown(f"- **Immediate Action:** You have {risk_count} parameter(s) outside normal range")
                            st.markdown("- **Next Steps:** Consult with your healthcare provider about these values")
                            st.markdown("- **Monitoring:** Schedule a follow-up test to track these parameters")
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        else:
                            st.markdown("<div class='success-banner'>✅ Excellent! All parameters are within normal range</div>", unsafe_allow_html=True)
                        
                        if normal_metrics:
                            with st.expander(f"✅ View Normal Parameters ({len(normal_metrics)} parameters)", expanded=False):
                                cols = st.columns(4)
                                for idx, (param, result) in enumerate(normal_metrics.items()):
                                    with cols[idx % 4]:
                                        st.metric(
                                            label=param,
                                            value=f"{result['value']:.2f} {result['unit']}",
                                            delta="Normal"
                                        )
                    else:
                        st.error(f"Could not find the required columns for '{test_type}' in your uploaded file.")
                        st.info(f"**Required columns for {test_type}:** {', '.join(HEALTH_TEST_PARAMETERS.get(test_type, {}).keys())}")
                        st.dataframe(new_report_df) # Show the user what was parsed

# ==============================================================================
# TAB 2: MANUAL ENTRY (This section is UNCHANGED)
# ==============================================================================
with tab2:
    st.header("Quick Manual Entry")
    st.info("Enter health parameters manually for quick analysis")
    
    with st.form("manual_entry_form"):
        # Select organ system
        selected_organ = st.selectbox("Select Organ/System", options=list(HEALTH_TEST_PARAMETERS.keys()))
        
        # Get parameters for selected organ
        available_params = HEALTH_TEST_PARAMETERS.get(selected_organ, {})
        
        if available_params:
            st.markdown(f"**Enter values for {selected_organ} parameters:**")
            
            # Create input fields for each parameter
            manual_data = {}
            num_cols = 2
            param_list = list(available_params.items())
            
            for i in range(0, len(param_list), num_cols):
                cols = st.columns(num_cols)
                for j in range(num_cols):
                    if i + j < len(param_list):
                        param_name, param_info = param_list[i + j]
                        with cols[j]:
                            value = st.number_input(
                                f"{param_name} ({param_info['unit']})",
                                min_value=0.0,
                                format="%.2f",
                                help=f"Normal range: {param_info['range'][0]}-{param_info['range'][1]} {param_info['unit']}"
                            )
                            manual_data[param_name] = value
            
            # Entry date
            entry_date = st.date_input("Report Date", value=datetime.now())
            
            # Submit button
            submitted = st.form_submit_button("Analyze Manual Entry", use_container_width=True, type="primary")
            
            if submitted:
                # Create dataframe from manual entry
                manual_df = pd.DataFrame([manual_data])
                manual_df['ID'] = st.session_state.get('username')
                manual_df['Date'] = entry_date
                
                # Save to database
                if os.path.exists(COMMON_DATABASE_FILE):
                    common_db_df = pd.read_excel(COMMON_DATABASE_FILE)
                    updated_db_df = pd.concat([common_db_df, manual_df], ignore_index=True)
                else:
                    updated_db_df = manual_df

                updated_db_df.to_excel(COMMON_DATABASE_FILE, index=False)
                
                st.success("✅ Manual entry saved successfully!")
                
                # Analyze the manual entry
                st.markdown("---")
                st.subheader("Analysis Results")
                
                analysis_results = analyze_report(manual_df, selected_organ)
                
                if analysis_results:
                    high_risk = {k: v for k, v in analysis_results.items() if v['status'] != 'Normal'}
                    normal = {k: v for k, v in analysis_results.items() if v['status'] == 'Normal'}
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Parameters", len(analysis_results))
                    col2.metric("Normal", len(normal))
                    col3.metric("Needs Attention", len(high_risk))
                    
                    if high_risk:
                        st.warning("⚠️ Parameters Requiring Attention")
                        for param, result in high_risk.items():
                            st.markdown(f"**{param}:** {result['value']:.2f} {result['unit']} - {result['delta_text']}")
                    else:
                        st.success("✅ All parameters are within normal range!")
        else:
            st.warning("No parameters available for the selected organ system.")

# --- Additional Resources (UNCHANGED) ---
st.markdown("---")
st.header("📚 Additional Resources")

col_res1, col_res2 = st.columns(2)

with col_res1:
    st.markdown("<div class='analysis-card'>", unsafe_allow_html=True)
    st.markdown("### 📖 Understanding Your Results")
    st.markdown("""
    - **Normal Range:** Your values fall within the healthy range
    - **High Values:** Indicates levels above the recommended upper limit
    - **Low Values:** Indicates levels below the recommended lower limit
    - **Deviation %:** Shows how far your value is from the normal range
    """)
    st.markdown("</div>", unsafe_allow_html=True)

with col_res2:
    st.markdown("<div class='analysis-card'>", unsafe_allow_html=True)
    st.markdown("### 💡 Next Steps")
    st.markdown("""
    - Review all parameters marked as requiring attention
    - Consult your healthcare provider for abnormal values
    - Track trends by uploading reports regularly
    - Maintain a healthy lifestyle and follow medical advice
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Navigation (UNCHANGED) ---
st.markdown("---")
col_nav1, col_nav2, col_nav3, col_nav4 = st.columns(4)

with col_nav1:
    if st.button("⬅️ Back to Dashboard", use_container_width=True):
        st.switch_page("pages/_dashboard.py")

with col_nav2:
    if st.button("📊 View History", use_container_width=True):
        st.switch_page("pages/_Report_History.py")

with col_nav3:
    if st.button("🔄 Refresh Page", use_container_width=True):
        st.rerun()

with col_nav4:
    if st.button("🚪 Logout", use_container_width=True, type="primary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.switch_page("report.py")