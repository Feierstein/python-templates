import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.xgboost.estimator import XGBoost
import boto3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib  # To save the scaler
import os

# 1. Set up the SageMaker session and role
sagemaker_session = sagemaker.Session()
bucket = sagemaker_session.default_bucket()  # Or specify your own bucket
role = sagemaker.get_execution_role()

# 2. Load and prepare the data
from sklearn.datasets import load_boston
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.DataFrame(boston.target, columns=["PRICE"])

# Split the data into training and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Fit and transform the training data using StandardScaler (or other transformations)
scaler = StandardScaler()

# Apply fit_transform to the training data
X_train_scaled = scaler.fit_transform(X_train)

# Apply transform to the validation data (important: do not use fit_transform here)
X_val_scaled = scaler.transform(X_val)

# 4. Save the transformation object for later use
scaler_filename = 'scaler.pkl'
joblib.dump(scaler, scaler_filename)

# Save the scaler file to S3 so it can be used later during inference
scaler_s3_path = sagemaker_session.upload_data(scaler_filename, bucket=bucket, key_prefix='xgboost-model/scaler')

# Combine the training data with target
train_data = pd.concat([pd.DataFrame(y_train.values), pd.DataFrame(X_train_scaled)], axis=1)
val_data = pd.concat([pd.DataFrame(y_val.values), pd.DataFrame(X_val_scaled)], axis=1)

# Save to CSV (without headers, since XGBoost expects it this way)
train_data.to_csv('train.csv', index=False, header=False)
val_data.to_csv('validation.csv', index=False, header=False)

# Upload the transformed data to S3
train_s3_path = sagemaker_session.upload_data('train.csv', bucket=bucket, key_prefix='xgboost-model')
val_s3_path = sagemaker_session.upload_data('validation.csv', bucket=bucket, key_prefix='xgboost-model')

# 5. Train the XGBoost model using SageMaker
container = sagemaker.image_uris.retrieve("xgboost", boto3.Session().region_name, "1.5-1")

xgb = XGBoost(
    entry_point=None,  # Not using custom code
    framework_version="1.5-1",
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    output_path=f"s3://{bucket}/xgboost-output",
    sagemaker_session=sagemaker_session,
    hyperparameters={
        "max_depth": 5,
        "eta": 0.2,
        "objective": "reg:squarederror",
        "num_round": 100
    }
)

xgb.fit({
    "train": TrainingInput(train_s3_path, content_type="csv"),
    "validation": TrainingInput(val_s3_path, content_type="csv")
})

# 6. Deploy the trained model to a SageMaker endpoint
predictor = xgb.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.xlarge",
    endpoint_name="xgboost-boston-endpoint"
)

# 7. Use the transformation during inference (deploy the model with the transformation)
# Create a custom inference script that loads both the model and the scaler, and applies the transformation to new data.

# Custom inference script to be placed in a separate file (e.g., 'inference.py')
"""
import os
import joblib
import numpy as np
from sagemaker_inference import content_types, default_inference_handler

class MyXGBoostInferenceHandler(default_inference_handler.DefaultInferenceHandler):
    def __init__(self):
        super().__init__()
        self.scaler = None

    def load_model(self, model_dir):
        # Load the trained model
        model = super().load_model(model_dir)

        # Load the scaler from the S3 path or model directory
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        self.scaler = joblib.load(scaler_path)

        return model

    def transform_fn(self, model, data, input_content_type, output_content_type):
        # Parse the input data (assuming CSV format)
        if input_content_type == content_types.CSV:
            data = np.array([float(x) for x in data.split(",")]).reshape(1, -1)
        
        # Apply the scaler transformation
        transformed_data = self.scaler.transform(data)
        
        # Make predictions
        predictions = model.predict(transformed_data)

        return str(predictions), output_content_type

"""

# Ensure the model and scaler are deployed together