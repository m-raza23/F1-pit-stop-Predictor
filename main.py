from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings('ignore')

# 1. Initialize FastAPI Server Application
app = FastAPI(
    title="ApexStrategy AI Core",
    description="Production Ensemble Inference Engine for F1 Pit Stop Window Analysis"
)

# 2. Load the Production Artifacts Once on Startup
print("⏳ Booting engine and loading ensemble artifacts...")
try:
    artifacts = joblib.load('f1_production_assets.pkl')
    print("🚀 All models, encoders, and weights loaded successfully into memory!")
except Exception as e:
    raise RuntimeError(f"Failed to load f1_production_assets.pkl. Did you run the training script? Error: {str(e)}")

# Extract dictionaries and models for quick global usage
DRIVERS_MAP = artifacts['driver_map']
RACES_MAP = artifacts['race_map']
COMPOUND_MAP = artifacts['compound_map']
COMPOUND_LIFE_MAP = {0: 20, 1: 30, 2: 40, 3: 25, 4: 35}

# 3. Define the Incoming Data Schema using Pydantic
class LapDataInput(BaseModel):
    Driver: str
    Race: str
    Compound: str
    TyreLife: float
    LapTime_Delta: float
    Cumulative_Degradation: float
    RaceProgress: float
    Stint: int
    Position: float
    Position_Change: float
    LapTime_s: float  # Map this from user's 'LapTime (s)' format

# 4. Define the Local Live Preprocessing Function
def preprocess_single_lap(input_data: LapDataInput):
    """ Transforms raw live dashboard inputs into the exact 46 features required by the models """
    
    # Safely convert incoming text labels using our saved training maps
    encoded_driver = DRIVERS_MAP.get(input_data.Driver, -1)
    encoded_race = RACES_MAP.get(input_data.Race, -1)
    encoded_compound = COMPOUND_MAP.get(input_data.Compound, -1)
    
    # Build dictionary matching training features base structures
    row = {
        'Driver': encoded_driver,
        'Race': encoded_race,
        'Compound': encoded_compound,
        'TyreLife': input_data.TyreLife,
        'LapTime_Delta': input_data.LapTime_Delta,
        'Cumulative_Degradation': input_data.Cumulative_Degradation,
        'RaceProgress': input_data.RaceProgress,
        'Stint': input_data.Stint,
        'Position': input_data.Position,
        'Position_Change': input_data.Position_Change,
        'LapTime (s)': input_data.LapTime_s,
        'LapNumber': 15 # Anchor placeholder for parsing bounds
    }
    
    # Calculate rolling derivations locally for single item simulation
    compound_life = COMPOUND_LIFE_MAP.get(encoded_compound, 30)
    tyre_life_pct = row['TyreLife'] / compound_life
    
    row['tyre_life_pct'] = tyre_life_pct
    row['tyre_life_remaining'] = 1 - tyre_life_pct
    row['tyre_critical'] = int(tyre_life_pct > 0.80)
    row['tyre_danger_zone'] = int(tyre_life_pct > 0.90)
    row['expected_life'] = compound_life
    row['laps_until_overdue'] = compound_life - row['TyreLife']
    
    # Without access to a true full dataframe profile during single calculations, 
    # we evaluate current window dynamics explicitly based on raw progress
    row['rolling_deg_2'] = row['LapTime_Delta']
    row['rolling_deg_3'] = row['LapTime_Delta']
    row['rolling_deg_5'] = row['LapTime_Delta']
    row['rolling_deg_std'] = 0.0
    row['deg_acceleration'] = 0.0
    row['deg_trend'] = 0.0
    row['cumulative_deg_rate'] = row['Cumulative_Degradation'] / (row['TyreLife'] + 1)
    
    row['in_first_window'] = int(0.25 <= row['RaceProgress'] <= 0.40)
    row['in_second_window'] = int(0.55 <= row['RaceProgress'] <= 0.70)
    row['too_late_to_pit'] = int(row['RaceProgress'] > 0.90)
    row['in_any_pit_window'] = int(row['in_first_window'] or row['in_second_window'])
    row['is_first_stint'] = int(row['Stint'] == 1)
    row['is_second_stint'] = int(row['Stint'] == 2)
    row['race_progress_sq'] = row['RaceProgress'] ** 2
    row['lap_in_race_pct'] = 0.5
    
    row['is_frontrunner'] = int(row['Position'] <= 5)
    row['is_backmarker'] = int(row['Position'] >= 16)
    row['in_midfield'] = int(6 <= row['Position'] <= 15)
    row['losing_positions'] = int(row['Position_Change'] < -1)
    row['gaining_positions'] = int(row['Position_Change'] > 1)
    row['rolling_pos_change'] = row['Position_Change']
    row['laptime_normalized'] = 0.0 # Standard mean anchor profile
    
    row['tyre_x_window'] = row['tyre_life_pct'] * row['in_any_pit_window']
    row['tyre_x_deg'] = row['tyre_life_pct'] * row['rolling_deg_3']
    row['critical_x_window'] = row['tyre_critical'] * row['in_any_pit_window']
    row['stint_x_tyre'] = row['Stint'] * row['tyre_life_pct']
    row['progress_x_tyre'] = row['RaceProgress'] * row['tyre_life_pct']
    
    # Map Target Encodings dynamically using target_encoder artifact
    df_temp = pd.DataFrame([row])
    encoded_te = artifacts['target_encoder'].transform(df_temp[['Driver', 'Race', 'Compound']])
    df_temp['Driver_te'] = encoded_te['Driver'].values[0]
    df_temp['Race_te'] = encoded_te['Race'].values[0]
    df_temp['Compound_te'] = encoded_te['Compound'].values[0]
    
    # Enforce correct ordering of all 46 features expected by predictors
    return df_temp[artifacts['features_list']]

# 5. Core Prediction Endpoint Handler
@app.post("/predict")
def get_prediction(payload: LapDataInput):
    try:
        # Step A: Format data
        processed_features = preprocess_single_lap(payload)
        
        # Step B: Gather Probabilities from Level-1 folds
        preds_xgb = np.mean([m.predict_proba(processed_features)[:, 1] for m in artifacts['models']['xgb']])
        preds_lgb = np.mean([m.predict_proba(processed_features)[:, 1] for m in artifacts['models']['lgb']])
        preds_cat = np.mean([m.predict_proba(processed_features)[:, 1] for m in artifacts['models']['cat']])
        
        # Step C: Pack Meta Features and pass through Stacking Logistic Regression
        meta_features = np.array([[preds_xgb, preds_lgb, preds_cat]])
        pairs = np.array([[preds_xgb * preds_lgb, preds_xgb * preds_cat, preds_lgb * preds_cat]])
        meta_combined = np.column_stack([meta_features, pairs])
        
        meta_scaled = artifacts['scaler'].transform(meta_combined)
        preds_meta = artifacts['meta_model'].predict_proba(meta_scaled)[:, 1][0]
        
        # Step D: Apply Optimal Weighted Blending formula
        w = artifacts['blend_weights']
        final_probability = (
            (preds_meta * w['meta']) +
            (preds_lgb * w['lgb']) +
            (preds_cat * w['cat']) +
            (preds_xgb * w['xgb'])
        )
        
        return {
            "status": "success",
            "pit_stop_probability": round(float(final_probability), 4),
            "model_breakdown": {
                "MetaStack": round(float(preds_meta), 4),
                "LightGBM": round(float(preds_lgb), 4),
                "CatBoost": round(float(preds_cat), 4),
                "XGBoost": round(float(preds_xgb), 4)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Failure: {str(e)}")

@app.get("/")
def home():
    return {"message": "ApexStrategy Model Server is Online"}