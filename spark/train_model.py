import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def clean_target_variable(df, target_col):
    """Clean and normalize the target variable with robust outlier handling"""
    df = df.copy()
    
    # Calculate robust bounds using median and IQR
    median = df[target_col].median()
    q1 = df[target_col].quantile(0.25)
    q3 = df[target_col].quantile(0.75)
    iqr = q3 - q1
    
    # Define bounds (more conservative than 1.5*IQR)
    lower_bound = max(0, q1 - 3 * iqr)  # Prices can't be negative
    upper_bound = q3 + 3 * iqr
    
    # Filter outliers
    clean_df = df[(df[target_col] >= lower_bound) & 
                  (df[target_col] <= upper_bound)].copy()
    
    # Log transform to handle skewness
    clean_df[f"{target_col}_log"] = np.log1p(clean_df[target_col])
    
    return clean_df, f"{target_col}_log"

def create_valid_lag_features(df, valid_years, lag, group_cols=["Area", "Item", "Months"]):
    """Creates lagged price features only for valid year combinations"""
    df = df.copy()
    for year in valid_years:
        year_col = f"Y{year}"
        if year_col in df.columns:
            # Only create lag if the base year has data
            lag_col = f"{year_col}_lag_{lag}"
            df[lag_col] = df.groupby(group_cols)[year_col].shift(lag)
            
            # Remove the lag column if it's all NA
            if df[lag_col].isna().all():
                df.drop(columns=[lag_col], inplace=True)
    return df

def analyze_target_variable(y, title="Price Distribution"):
    """Enhanced target variable analysis with diagnostic plots"""
    print("\n=== Target Variable Analysis ===")
    print(y.describe())
    
    plt.figure(figsize=(15, 5))
    
    # Original scale
    plt.subplot(1, 3, 1)
    plt.hist(y.dropna(), bins=50)
    plt.title(f"Original {title}")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    
    # Log scale
    plt.subplot(1, 3, 2)
    plt.hist(np.log1p(y.dropna()), bins=50)
    plt.title(f"Log-Transformed {title}")
    plt.xlabel("Log(Price)")
    
    # Boxplot
    plt.subplot(1, 3, 3)
    plt.boxplot(y.dropna())
    plt.title("Price Boxplot")
    
    plt.tight_layout()
    plt.show()

def preprocess_features(X):
    """Enhanced feature preprocessing with robust missing value handling"""
    X = X.copy()
    
    # More robust missing value handling
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    
    # For numeric columns - use median imputation only if less than 50% missing
    for col in numeric_cols:
        if X[col].isna().mean() > 0.5:  # If more than 50% missing
            X.drop(col, axis=1, inplace=True)
        elif X[col].isna().any():
            imputer = SimpleImputer(strategy="median")
            X[col] = imputer.fit_transform(X[[col]])
    
    # For categoricals - create 'missing' category
    for col in categorical_cols:
        if X[col].isna().any():
            X[col] = X[col].fillna('missing')
    
    # One-hot encode with drop_first to reduce dimensionality
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dummy_na=False)
    
    return X

def train_evaluate_model(df, features, target):
    """Main model training and evaluation function"""
    # Clean and transform target variable
    df, target = clean_target_variable(df, target)
    
    # Drop rows where target is NA
    df = df.dropna(subset=[target]).copy()
    
    if len(df) == 0:
        raise ValueError("No data available after cleaning")
    
    analyze_target_variable(np.expm1(df[target]), "Cleaned Target Variable")

    X = df[features].copy()
    y = df[target].copy()

    # Preprocess features
    X_processed = preprocess_features(X)
    
    # Only keep columns that have data after preprocessing
    valid_cols = [col for col in X_processed.columns if not X_processed[col].isna().all()]
    X_processed = X_processed[valid_cols]
    
    # Scale numeric features
    numeric_cols = X_processed.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        scaler = StandardScaler()
        X_processed[numeric_cols] = scaler.fit_transform(X_processed[numeric_cols])
    else:
        scaler = None

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Predictions (converting back from log scale)
    y_pred = np.expm1(model.predict(X_test))
    y_test = np.expm1(y_test)
    
    # Round predictions to 2 decimal places
    y_pred = np.round(y_pred, 2)
    
    # Evaluation metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mean_price = y_test.mean()
    
    print("\n=== Model Evaluation ===")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R² Score: {r2:.2f}")
    print(f"Mean Price: {mean_price:.2f}")
    print(f"RMSE as % of Mean: {(rmse/mean_price)*100:.2f}%")
    
    # Diagnostic plots
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs Predicted Prices")
    
    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Predicted Prices")
    plt.ylabel("Residuals")
    plt.title("Residual Analysis")
    
    plt.tight_layout()
    plt.show()

    return model, scaler, X_processed.columns.tolist()

def main():
    # Load data
    data = pd.read_csv(
        r"C:\Users\mgree\Downloads\Bigdata_project\real_time_prediction\data\Prices_E_All_Data.csv"
    )

    # Filter to only monthly data
    valid_months = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]
    data = data[data["Months"].isin(valid_months)].copy()

    # Create month features
    month_map = {month: idx+1 for idx, month in enumerate(valid_months)}
    month_data = pd.DataFrame({
        "Month_sin": np.sin(2 * np.pi * data["Months"].map(month_map) / 12),
        "Month_cos": np.cos(2 * np.pi * data["Months"].map(month_map) / 12)
    })
    data = pd.concat([data, month_data], axis=1)

    # Find years with actual data
    year_cols = [col for col in data.columns  
                  if col.startswith("Y") and len(col) == 5  
                  and not col.endswith("F")]
    
    valid_years = []
    for col in year_cols:
        if data[col].notna().any():
            valid_years.append(int(col[1:]))
    
    print(f"\nYears with data: {sorted(valid_years)}")
    
    # Use most recent year with data as target
    target_year = sorted(valid_years, reverse=True)[0]
    target = f"Y{target_year}"
    print(f"Using {target} as target variable")

    # Save target year information
    os.makedirs("models", exist_ok=True)
    with open("models/target_year.pkl", "wb") as f:
        pickle.dump(target_year, f)

    # Initial analysis of raw target
    analyze_target_variable(data[target], "Raw Target Variable")

    # Create lag features only for valid year combinations
    feature_years = [year for year in valid_years if year != target_year]
    data = data.sort_values(["Area", "Item", "Months"])
    
    # Only create lags for years where we have sufficient history
    for lag in [1, 2, 3]:
        data = create_valid_lag_features(data, feature_years, lag)

    # Select features - only use columns we know have data
    base_features = ["Area", "Item", "Months", "Month_sin", "Month_cos"]
    year_features = [f"Y{year}" for year in feature_years]
    lag_features = [col for col in data.columns  
                    if "lag" in col and any(f"Y{year}_" in col for year in feature_years)]
    
    features = base_features + year_features + lag_features

    # Train model with enhanced preprocessing
    try:
        print("\nStarting model training with enhanced preprocessing...")
        model, scaler, train_cols = train_evaluate_model(data, features, target)
        
        # Save artifacts
        with open("models/crop_price_model.pkl", "wb") as f:
            pickle.dump(model, f)
        if scaler is not None:
            with open("models/crop_price_scaler.pkl", "wb") as f:
                pickle.dump(scaler, f)
        with open("models/train_cols.pkl", "wb") as f:
            pickle.dump(train_cols, f)
        
        print("\n=== Model Saved Successfully ===")
        print(f"Model saved to: models/crop_price_model.pkl")
        print(f"Scaler saved to: models/crop_price_scaler.pkl")
        print(f"Features saved to: models/train_cols.pkl")
        print(f"Target year saved to: models/target_year.pkl")
        
    except Exception as e:
        print(f"\n!!! Model training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()