from flask import Flask, request, jsonify
import pandas as pd
from scripts.utils import load_input_features, load_scalers_encoders, load_mlflow_model
from scripts.input_transformer import feature_engineering_pipeline

# Initialize Flask app
app = Flask(__name__)

# Load feature names and scaled numerical column names
FEATURES_PATH = "./feature_store/fraud_features.pkl"
SCALED_NUMERICAL_PATH = "./feature_store/scaled_numerical_features.pkl"
features, scaled_features = load_input_features(feature_path=FEATURES_PATH, scaled_numerical_path=SCALED_NUMERICAL_PATH)

# Load encoder and scaler
ENCODER_PATH = "./scalers/categorical_encoder.pkl"
SCALER_PATH = "./scalers/fraud_scaler.pkl"
scaler, encoder = load_scalers_encoders(scaler_path=SCALER_PATH, encoder_path=ENCODER_PATH)

# Load MLflow model
MODEL_PATH = "./model/artifacts"
model = load_mlflow_model(parent_folder=MODEL_PATH)

# Define prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.get_json()

    input_df = pd.DataFrame([data], columns=features)

    # Apply feature engineering or transformation pipeline
    transformed_df = feature_engineering_pipeline(
        data=input_df,
        scaler=scaler,
        encoder=encoder,
        scaled_columns=scaled_features
    )
    
    # Make prediction using the model
    prediction = model.predict(transformed_df.to_numpy())

    # result of fraud
    fraud = 1 if prediction[0][0] > 0.5 else 0

    # Return the prediction result as JSON
    return jsonify({
        "fraud": fraud
    })

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
