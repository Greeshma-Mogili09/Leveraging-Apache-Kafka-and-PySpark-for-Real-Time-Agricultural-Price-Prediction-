import os
import pickle
import pandas as pd
import numpy as np
from kafka import KafkaProducer, KafkaConsumer
import json

class PandasPredictor:
    def __init__(self, model_dir):
        # Load model artifacts
        with open(os.path.join(model_dir, r"C:\Users\mgree\Downloads\Bigdata_project\real_time_prediction\spark\models\crop_price_model.pkl"), "rb") as f:
            self.model = pickle.load(f)
        with open(os.path.join(model_dir, r"C:\Users\mgree\Downloads\Bigdata_project\real_time_prediction\spark\models\crop_price_scaler.pkl"), "rb") as f:
            self.scaler = pickle.load(f)
        with open(os.path.join(model_dir, r"C:\Users\mgree\Downloads\Bigdata_project\real_time_prediction\spark\models\train_cols.pkl"), "rb") as f:
            self.train_cols = pickle.load(f)
        
        # Kafka producer for predictions
        self.producer = KafkaProducer(
            bootstrap_servers='localhost:29092',
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

    def process_message(self, message):
        try:
            data = json.loads(message.value.decode('utf-8'))
            pdf = pd.DataFrame([data])
            
            # Feature engineering
            month_map = {month: idx+1 for idx, month in enumerate([
                'January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December'
            ])}
            pdf['Month_sin'] = np.sin(2 * np.pi * pdf["Months"].map(month_map) / 12)
            pdf['Month_cos'] = np.cos(2 * np.pi * pdf["Months"].map(month_map) / 12)
            
            # Prepare features
            numeric_cols = ['Month_sin', 'Month_cos'] + [f"Y{year}" for year in range(1991, 2025)]
            categorical_cols = ['Area', 'Item', 'Months']
            encoded = pd.get_dummies(pdf[categorical_cols], prefix=categorical_cols)
            final_df = pd.concat([pdf[numeric_cols], encoded], axis=1)
            
            # Align columns with training data
            for col in self.train_cols:
                if col not in final_df.columns:
                    final_df[col] = 0
            final_df = final_df[self.train_cols]
            
            # Scale features
            if self.scaler:
                num_cols = [c for c in self.scaler.feature_names_in_ if c in final_df.columns]
                final_df[num_cols] = self.scaler.transform(final_df[num_cols])
            
            # Predict
            prediction = self.model.predict(final_df)[0]
            result = {
                "Area": data["Area"],
                "Item": data["Item"],
                "Months": data["Months"],
                "Prediction": float(prediction)
            }
            
            self.producer.send('crop_price_predictions', result)
            print(f"Prediction sent: {result}")
            
        except Exception as e:
            print(f"Error processing message: {str(e)}")

    def run(self):
        consumer = KafkaConsumer(
            'crop_price_data',
            bootstrap_servers='localhost:29092',
            auto_offset_reset='earliest',
            group_id='pandas_predictor'
        )
        print("Pandas predictor started. Waiting for messages...")
        for message in consumer:
            self.process_message(message)

if __name__ == "__main__":
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    PandasPredictor(model_dir).run()