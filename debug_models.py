import joblib
import dill
import os
import sys

try:
    print(f"CWD: {os.getcwd()}")
    print("Loading scalers.pkl...")
    scalers = joblib.load("scalers.pkl")
    print("Scalers loaded.")
    
    print("Loading xgb_model.pkl...")
    xgb_model = joblib.load("xgb_model.pkl")
    print("XGB model loaded.")
    
    print("Loading lime_explainer.dill...")
    with open("lime_explainer.dill", "rb") as f:
        bundle = dill.load(f)
    print("LIME explainer loaded.")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
