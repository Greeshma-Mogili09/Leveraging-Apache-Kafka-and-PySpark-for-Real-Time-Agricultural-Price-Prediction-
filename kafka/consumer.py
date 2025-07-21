from kafka import KafkaConsumer
import json

def create_consumer(topic_name):
    """Helper function to create a configured consumer"""
    return KafkaConsumer(
        topic_name,
        bootstrap_servers='localhost:29092',
        auto_offset_reset='earliest',
        group_id='crop_price_analysis_group',  # Added consumer group ID
        enable_auto_commit=True,
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        consumer_timeout_ms=10000  # Timeout after 10 seconds of no messages
    )

def consume_messages():
    print("Listening to Kafka topics...")

    # Create consumer for raw data
    data_consumer = create_consumer('crop_price_data')

    # Create consumer for predictions
    predictions_consumer = create_consumer('crop_price_predictions')

    print("Listening for messages. Press Ctrl+C to exit...")

    try:
        while True:
            # Check for raw data messages
            data_messages = data_consumer.poll(timeout_ms=1000)
            for _, messages in data_messages.items():
                for message in messages:
                    print(f"\n[RAW DATA] Received at {message.timestamp}:")
                    print(json.dumps(message.value, indent=2))

            # Check for prediction messages
            prediction_messages = predictions_consumer.poll(timeout_ms=1000)
            for _, messages in prediction_messages.items():
                for message in messages:
                    print(f"\n[PREDICTION] Received at {message.timestamp}:")
                    print(json.dumps(message.value, indent=2))

    except KeyboardInterrupt:
        print("\nShutting down consumers...")
    finally:
        data_consumer.close()
        predictions_consumer.close()
        print("Consumers closed properly.")

if __name__ == "__main__":
    consume_messages()