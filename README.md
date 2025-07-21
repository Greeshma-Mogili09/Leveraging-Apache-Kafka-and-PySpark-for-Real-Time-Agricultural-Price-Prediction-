# Leveraging-Apache-Kafka-and-PySpark-for-Real-Time-Agricultural-Price-Prediction-
This project implements a scalable big data pipeline for agricultural commodity price forecasting using Apache Kafka (for real-time data streaming) and PySpark MLlib (for distributed machine learning). The system processes 650K+ records from the FAOSTAT dataset, integrating batch and streaming workflows to predict prices with 85% R² accuracy (RMSE: $12.45, 9.2% of mean price). Key features include:

Hybrid Architecture: Kafka simulates live market data feeds, while PySpark handles feature engineering (lag variables, cyclical encoding) and model training (Gradient Boosted Trees).

Performance: Achieves 180ms latency in streaming predictions, with automated drift detection and Slack alerts for anomalies.

Impact: Provides actionable insights for farmers and policymakers, addressing volatility in agricultural markets.

Tech Stack: Python, PySpark, Kafka, Docker, Spark Structured Streaming, Scikit-learn.

