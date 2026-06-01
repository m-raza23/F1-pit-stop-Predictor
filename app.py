import streamlit as st
import requests
import pandas as pd
import time

# 1. Page Configuration & Styling
st.set_page_config(
    page_title="ApexStrategy // Pit Wall Telemetry",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply a dark, high-contrast racing theme
st.markdown("""
    <style>
    .main { background-color: #0f1115; color: #f0f2f6; }
    .stNumberInput, .stSlider, .stSelectbox { background-color: #1a1c23 !important; }
    h1, h2, h3 { font-family: 'Courier New', Courier, monospace; font-weight: bold; }
    .metric-card {
        background-color: #161920;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #e10600; /* F1 Red */
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

page = st.sidebar.radio("Navigation", ["Dashboard", "About"], index=0)
st.sidebar.markdown("---")

if page == "About":
    st.title("🏎️ APEXSTRATEGY // About")
    st.subheader("F1 strategy dashboard with a FastAPI inference backend")
    st.markdown("---")
    st.markdown(
        "**ApexStrategy** is a live telemetry dashboard designed for F1 pit wall strategy.")
    st.markdown(
        "It combines a Streamlit front-end with a FastAPI model server to predict pit stop probability in real time.")
    st.markdown("### What this project includes")
    st.markdown(
        "- `app.py` — Streamlit dashboard UI and telemetry controls\n"
        "- `main.py` — FastAPI prediction API with ensemble inference\n"
        "- `f1_production_assets.pkl` — saved model artifacts and encoders\n"
        "- `requirements.txt` — Python package dependencies"
    )
    st.markdown("### How to run")
    st.markdown(
        "1. Start the backend: `python main.py`\n"
        "2. Run the dashboard: `streamlit run app.py`\n"
        "3. Use the sidebar to select a driver, race, tyres, and live telemetry values."
    )
    st.markdown("### Why this project")
    st.markdown(
        "ApexStrategy simulates a pit wall decision engine using ensemble machine learning models and"
        " a clean racing-themed UI for strategy review and training." 
    )
    st.markdown("### GitHub Ready")
    st.markdown(
        "This repository is ready for GitHub with a local git history and standard Python ignores. "
        "Add a remote and push your code to publish it." 
    )
    st.stop()

st.title("🏎️ APEXSTRATEGY // Real-Time Pit Wall Telemetry")
st.subheader("Ensemble Predictive Inference Engine for F1 Race Strategy")
st.markdown("---")

# 2. Sidebar Configuration & Inputs
st.sidebar.header("🕹️ Race Control Center")

# Fetch maps from local storage file to populate dropdown selections
try:
    import joblib
    assets = joblib.load('f1_production_assets.pkl')
    drivers_list = sorted(list(assets['driver_map'].keys()))
    races_list = sorted(list(assets['race_map'].keys()))
    compounds_list = sorted(list(assets['compound_map'].keys()))
except:
    drivers_list = ["VER", "HAM", "LEC", "NOR", "SAI", "ALO", "PIA", "RUS"]
    races_list = ["Bahrain Grand Prix", "Monaco Grand Prix", "Silverstone", "Monza"]
    compounds_list = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]

selected_driver = st.sidebar.selectbox("Select Driver Focus", drivers_list)
selected_race = st.sidebar.selectbox("Select Grand Prix Location", races_list)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Live Telemetry Inputs")

current_stint = st.sidebar.number_input("Current Stint Number", min_value=1, max_value=4, value=1)
current_position = st.sidebar.slider("Current Track Position", 1, 20, 3)
position_change = st.sidebar.slider("Position Change (Last 3 Laps)", -5, 5, 0)

st.sidebar.markdown("---")
# Simulation Controls
run_sim = st.sidebar.button("🚀 Start Live Lap Simulation Loop", use_container_width=True)

# 3. Main Dashboard Layout Space
col_left, col_right = st.columns([2, 1])

with col_left:
    st.header("⏱️ Active Stint Telemetry")
    
    # Grid of Sliders for dynamic manual tweaking
    c1, c2 = st.columns(2)
    with c1:
        selected_compound = st.selectbox("Current Tyre Compound", compounds_list, index=1)
        tyre_life = st.slider("Tyre Age (Laps run)", 0.0, 50.0, 12.0)
        cumulative_deg = st.slider("Cumulative Tyre Degradation", 0.0, 5.0, 0.8, step=0.1)
    with c2:
        race_progress = st.slider("Race Progress (0.0 to 1.0)", 0.0, 1.0, 0.32, step=0.01)
        lap_delta = st.slider("Lap Time Delta (Seconds vs. Personal Average)", -2.0, 4.0, 0.3, step=0.1)
        lap_time_s = st.slider("Last Raw Lap Time (Seconds)", 60.0, 100.0, 78.5, step=0.1)

with col_right:
    st.header("🧠 Strategy Prediction")
    
    # Pack parameters into payload
    payload = {
        "Driver": selected_driver,
        "Race": selected_race,
        "Compound": selected_compound,
        "TyreLife": tyre_life,
        "LapTime_Delta": lap_delta,
        "Cumulative_Degradation": cumulative_deg,
        "RaceProgress": race_progress,
        "Stint": current_stint,
        "Position": float(current_position),
        "Position_Change": float(position_change),
        "LapTime_s": lap_time_s
    }
    
    # API Communication Trigger
    API_URL = "http://127.0.0.1:8000/predict"
    
    if not run_sim:
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                result = response.json()
                prob = result['pit_stop_probability']
                
                # Dynamic Alert Banner Color shifting based on Risk Thresholds
                if prob < 0.30:
                    color = "#28a745" # Green (Stay Out)
                    status_text = "STAY OUT // STINT STABLE"
                elif prob < 0.70:
                    color = "#ffc107" # Yellow (Prepare Window)
                    status_text = "MONITORING // PIT WINDOW OPENING"
                else:
                    color = "#dc3545" # Red (Box This Lap)
                    status_text = "⚠️ BOX THIS LAP // CRITICAL DEGRADATION"
                
                st.markdown(f"""
                    <div class="metric-card" style="border-left-color: {color};">
                        <h3 style="color: {color}; margin: 0;">{status_text}</h3>
                        <p style="margin: 10px 0 0 0; font-size: 14px; color: #aaa;">PIT PROBABILITY TARGET NEXT LAP</p>
                        <h1 style="font-size: 48px; margin: 0; color: #fff;">{prob*100:.1f}%</h1>
                    </div>
                """, unsafe_allow_html=True)
                
                # Breakdown Sub-metrics
                st.subheader("🤖 Sub-Model Breakdown")
                breakdown = result['model_breakdown']
                st.progress(breakdown['MetaStack'], text=f"Meta-Model Classifier: {breakdown['MetaStack']*100:.1f}%")
                st.progress(breakdown['LightGBM'], text=f"LightGBM Pipeline: {breakdown['LightGBM']*100:.1f}%")
                st.progress(breakdown['CatBoost'], text=f"CatBoost Pipeline: {breakdown['CatBoost']*100:.1f}%")
                st.progress(breakdown['XGBoost'], text=f"XGBoost Pipeline: {breakdown['XGBoost']*100:.1f}%")
                
            else:
                st.error(f"Backend Server Error code: {response.status_code}")
        except Exception as e:
            st.warning("Could not establish a connection to the FastAPI Backend Server. Verify main.py is up and running on port 8000.")

# 4. Live Simulation Loop Execution Mode
if run_sim:
    st.markdown("---")
    st.header("📺 Live Pit-Wall Feed Simulation")
    sim_status = st.empty()
    chart_holder = st.empty()
    
    # Mocking sequential lap progression data
    sim_data = []
    base_progress = race_progress
    base_tyre_life = tyre_life
    base_deg = cumulative_deg
    
    for current_lap in range(1, 11):
        # Scale parameters incrementally lap over lap
        base_progress += 0.015
        base_tyre_life += 1.0
        base_deg += 0.15
        if base_tyre_life > 18:
            lap_delta += 0.25
        
        payload["RaceProgress"] = base_progress
        payload["TyreLife"] = base_tyre_life
        payload["Cumulative_Degradation"] = base_deg
        payload["LapTime_Delta"] = lap_delta
        
        try:
            res = requests.post(API_URL, json=payload).json()
            p_val = res['pit_stop_probability']
            sim_data.append({"Simulated Lap": current_lap, "Pit Probability": p_val * 100})
            
            sim_status.metric(
                label=f"🔄 Processing Race Simulation Lap {current_lap}/10...",
                value=f"{p_val*100:.1f}% Pit Risk",
                delta=f"+{base_tyre_life - tyre_life} Laps on Compound"
            )
            
            # Update visualization dataframe graph dynamically
            df_chart = pd.DataFrame(sim_data).set_index("Simulated Lap")
            chart_holder.line_chart(df_chart, y="Pit Probability", color="#e10600")
            
            time.sleep(0.8) # Quick delay pause to make the animation fluid
        except:
            st.error("Simulation broken. Lost contact with Backend Server API.")
            break
    st.balloons()