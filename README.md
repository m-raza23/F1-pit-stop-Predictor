# ApexStrategy

ApexStrategy is a small F1 strategy dashboard project built with Streamlit and FastAPI.

## What this project does

- `app.py` is the Streamlit user interface.
  - It shows a racing dashboard with live telemetry inputs.
  - The sidebar lets you select driver, race, tyre compound, and race status values.
  - It sends input data to the backend and displays pit stop probability and model breakdown.
  - It includes an `About` page describing the project and how to use it.
  - It can also simulate a 10-lap live feed loop when the simulation button is pressed.

- `main.py` is the FastAPI backend.
  - It loads saved production artifacts from `f1_production_assets.pkl`.
  - It defines a `/predict` endpoint that accepts live telemetry data.
  - It preprocesses dashboard inputs into the feature set expected by the machine learning models.
  - It runs ensemble inference from XGBoost, LightGBM, CatBoost, and a meta model.
  - It returns a final pit stop probability and sub-model breakdown.

- `f1_production_assets.pkl` contains the saved model artifacts, encoders, and feature list.
- `requirements.txt` lists the Python packages needed for Streamlit and FastAPI.
- `.gitignore` ignores Python virtual environment files and other generated artifacts.

## How to run

1. Activate your virtual environment.
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Start the backend server:
   ```powershell
   python main.py
   ```
4. Open the Streamlit dashboard:
   ```powershell
   streamlit run app.py
   ```

## What happens when you use it

- The Streamlit app collects telemetry values from the sidebar.
- It sends those values to the FastAPI backend.
- The backend preprocesses the values into model features and runs the ensemble.
- The dashboard displays a pit stop probability, colored status message, and model breakdown.
- If the simulation loop is started, the app makes repeated backend requests to simulate multiple laps.

## Notes

- Make sure `main.py` is running before using `app.py`.
- If the backend is not available, the dashboard will show a warning instead of predictions.
- The `About` page is available from the sidebar navigation.
