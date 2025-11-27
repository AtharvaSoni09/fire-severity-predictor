import os
import joblib
import dill
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
import io
import base64

app = Flask(__name__)

# Global variables for models
scalers = None
xgb_model = None
explainer = None
predict_fn = None
std_scaler = None
pwr_scaler = None

POWER_COLS = ["precip_in", "pop_density"]
STANDARD_COLS = ["temp_max_F", "humidity_pct", "windspeed_mph", "ndvi", "slope"]

def load_models():
    global scalers, xgb_model, explainer, predict_fn, std_scaler, pwr_scaler
    try:
        print("Loading scalers...")
        scalers = joblib.load("scalers.pkl")
        std_scaler = scalers["standard_scaler"]
        pwr_scaler = scalers["power_scaler"]
        print("Scalers loaded.")

        print("Loading XGBoost model...")
        xgb_model = joblib.load("xgb_model.pkl")
        print("XGBoost model loaded.")

        print("Loading LIME explainer...")
        try:
            # Instead of loading the pickled explainer which causes crashes, we initialize a new one.
            # We need the training data statistics. Since we don't have the full training data,
            # we will use the scaler's mean and scale to approximate the training data distribution
            # or just initialize it with a representative sample if possible.
            
            # Better approach: The user wants a working app. The pickled explainer is broken.
            # We will try to load it, but if it fails (or we know it crashes), we create a new one.
            # Since we don't have X_train, we can't perfectly recreate it.
            # HOWEVER, we can try to unpickle ONLY the training data if it's in the bundle?
            # The bundle had "explainer" and "predict_fn".
            
            # Let's try to create a fresh explainer using the scaler's stats to generate dummy training data
            # This is a hack but better than a crash.
            
            import lime.lime_tabular
            
            # Reconstruct training data summary from scalers
            # std_scaler.mean_ is the mean, std_scaler.scale_ is the std dev
            means = std_scaler.mean_
            stds = std_scaler.scale_
            
            # Generate dummy training data (e.g. 100 samples)
            # STANDARD_COLS = ["temp_max_F", "humidity_pct", "windspeed_mph", "ndvi", "slope"]
            # POWER_COLS = ["precip_in", "pop_density"]
            
            # We need to be careful about column order. The explainer expects a specific order.
            # The original code had:
            # self.standard_cols = ["temp_max_F", "humidity_pct", "windspeed_mph", "ndvi","slope"]
            # self.power_cols = ["precip_in", "pop_density"]
            # And input_df was created with specific order.
            
            # Let's assume the order used in robust_predict_fn:
            # ["temp_max_F", "humidity_pct", "precip_in", "windspeed_mph", "ndvi", "pop_density", "slope"]
            
            # We will create a dummy training set to initialize LIME
            # Since the model expects scaled data (which is approximately standard normal),
            # we generate random data with mean 0 and std 1.
            # Use a fixed seed to ensure consistency across server restarts
            np.random.seed(42)
            dummy_train_data = np.random.normal(0, 1, (1000, 7))
            feature_names = ["temp_max_F", "humidity_pct", "precip_in", "windspeed_mph", "ndvi", "pop_density", "slope"]
            
            explainer = lime.lime_tabular.LimeTabularExplainer(
                dummy_train_data,
                feature_names=feature_names,
                class_names=['fire_severity'],
                mode='regression'
            )
            predict_fn = None # We will use robust_predict_fn defined in predict()
            print("Created fresh LIME explainer.")
            
        except Exception as e:
            print(f"WARNING: Could not create LIME explainer. Explanations will be disabled. Error: {e}")
            explainer = None

    except Exception as e:
        print(f"CRITICAL ERROR loading models: {e}")
        # We don't exit here to allow the server to start and show errors in logs
        pass

# Load on startup
load_models()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global std_scaler, pwr_scaler, xgb_model, explainer, predict_fn
    
    if not std_scaler or not xgb_model:
        return jsonify({"success": False, "error": "Models not loaded correctly. Check server logs."}), 500

    try:
        data = request.json
        
        # Extract inputs
        inputs = {
            "temp_max_F": float(data.get("temp_max_F", 75)),
            "humidity_pct": float(data.get("humidity_pct", 50)),
            "windspeed_mph": float(data.get("windspeed_mph", 6)),
            "precip_in": float(data.get("precip_in", 0)),
            "ndvi": float(data.get("ndvi", 3000)),
            "pop_density": float(data.get("pop_density", 200)),
            "slope": float(data.get("slope", 600))
        }
        
        # Create DataFrame
        input_df = pd.DataFrame([inputs])
        
        # Enforce correct column order matching the training data
        cols_order = ["temp_max_F", "humidity_pct", "precip_in", "windspeed_mph", "ndvi", "pop_density", "slope"]
        input_df = input_df[cols_order]
        
        # Scale inputs
        input_df[STANDARD_COLS] = std_scaler.transform(input_df[STANDARD_COLS])
        input_df[POWER_COLS] = pwr_scaler.transform(input_df[POWER_COLS])
        
        # Predict
        log_prediction = float(xgb_model.predict(input_df)[0])
        prediction_acres = float(10**log_prediction)
        
        with open("debug_log.txt", "a") as f:
            f.write(f"Prediction made: {prediction_acres}\n")
        
        lime_plot_url = ""
        lime_list = []

        # LIME Explanation (Only if loaded)
        if explainer:
            with open("debug_log.txt", "a") as f:
                f.write("Explainer exists. Starting explanation...\n")
            try:
                instance = input_df.iloc[0].values
                with open("debug_log.txt", "a") as f:
                    f.write(f"Instance shape: {instance.shape}\n")
                
                # Define robust predict function for LIME
                # We use this instead of the pickled one to ensure compatibility with our DataFrame structure
                def robust_predict_fn(X):
                    # X is a numpy array of shape (n_samples, n_features)
                    # Convert back to DataFrame with correct column names
                    cols_order = ["temp_max_F", "humidity_pct", "precip_in", "windspeed_mph", "ndvi", "pop_density", "slope"]
                    df = pd.DataFrame(X, columns=cols_order)
                    return xgb_model.predict(df)

                def dummy_predict_fn(X):
                    # Return random predictions to test if LIME itself works
                    return np.array([5.0] * X.shape[0])

                with open("debug_log.txt", "a") as f:
                    f.write(f"Explainer mode: {explainer.mode}\n")

                # Try with dummy first to see if it crashes
                # explanation = explainer.explain_instance(instance, dummy_predict_fn, num_samples=500)
                
                # If dummy works, we know explainer is fine. Let's try robust again but maybe with even fewer samples?
                # Or maybe the issue is threading? XGBoost is not thread safe?
                # Let's try robust_predict_fn again but catch everything
                
                # Use random_state to ensure consistent explanations for the same input
                explanation = explainer.explain_instance(instance, robust_predict_fn, num_samples=100, random_state=42)
                
                with open("debug_log.txt", "a") as f:
                    f.write("Explanation generated.\n")
                
                # Generate Plot
                fig = explanation.as_pyplot_figure()
                with open("debug_log.txt", "a") as f:
                    f.write("Figure created.\n")
                
                input_text = "\n".join(f"{col}: {val}" for col, val in inputs.items())
                plt.text(
                    x=plt.xlim()[1] + 0.01 * plt.xlim()[1],
                    y=0.5 * plt.ylim()[1],
                    s=input_text,
                    fontsize=9,
                    verticalalignment='top',
                    horizontalalignment='left'
                )
                plt.subplots_adjust(right=0.75)
                plt.title(f"Predicted: {log_prediction:.2f} (Log), Acres: {prediction_acres:.2f}")
                
                img = io.BytesIO()
                plt.savefig(img, format='png', bbox_inches='tight')
                img.seek(0)
                lime_plot_url = base64.b64encode(img.getvalue()).decode()
                plt.close(fig)
                
                with open("debug_log.txt", "a") as f:
                    f.write("Plot saved.\n")

                # Convert LIME list items to native types
                raw_lime_list = explanation.as_list()
                lime_list = [(str(k), float(v)) for k, v in raw_lime_list]
                
                with open("debug_log.txt", "a") as f:
                    f.write("List converted.\n")
                
            except Exception as e:
                print(f"Error generating LIME explanation: {e}")
                with open("lime_error.log", "w") as f:
                    f.write(str(e))
                    import traceback
                    traceback.print_exc(file=f)
                # Continue without explanation
        else:
             with open("debug_log.txt", "a") as f:
                f.write("Explainer is None.\n")
        
        return jsonify({
            "success": True,
            "prediction_acres": prediction_acres,
            "prediction_log": log_prediction,
            "lime_plot": lime_plot_url,
            "lime_data": lime_list
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
