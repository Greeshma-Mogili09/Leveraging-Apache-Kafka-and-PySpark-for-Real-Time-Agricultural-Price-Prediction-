from kafka import KafkaProducer
import json
import time
import pandas as pd

# Load dataset
data = pd.read_csv(r"C:\Users\mgree\Downloads\Bigdata_project\real_time_prediction\data\Prices_E_All_Data.csv")
monthly_data = data[data["Months"].isin([
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
])]

# Initialize producer
producer = KafkaProducer(
    bootstrap_servers='localhost:29092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    api_version=(2, 8, 1)
)

# Send messages
for _, row in monthly_data.iterrows():
    # Modify the message dictionary to include all expected years
    message = {
    "Area": row["Area"],
    "Item": row["Item"],
    "Months": row["Months"],
    **{f"Y{year}": row.get(f"Y{year}", 0) for year in range(1991, 2025)}  # All years 1991-2024
}
    producer.send('crop_price_data', message)
    print(f"Sent: {row['Area']} - {row['Item']}")
    time.sleep(0.5)  # Throttle messages

producer.flush()