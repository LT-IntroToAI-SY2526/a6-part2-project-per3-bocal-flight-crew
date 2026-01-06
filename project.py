import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

DATA_FILE = 'Data_Train.csv'

def load_and_explore_data(filename):
    data = pd.read_csv(filename)
    
    data.columns = data.columns.str.strip()

    print("=== Ticket Price Data ===")
    print("\nFirst 5 rows:")
    print(data.head())

    print(f"\nDataset shape: {data.shape[0]} rows, {data.shape[1]} columns")
    
    print("\nBasic statistics:")
    print(data.describe())
    
    print("\nMissing values:")
    print(data.isnull().sum())
    
    return data


def preprocess_data(data):
    """Convert categorical/time/duration columns into numeric values."""

    # Encode Airline as category codes
    data['Airline'] = data['Airline'].astype('category').cat.codes

    # Convert Dep_Time to hour only
    data['Dep_Time'] = pd.to_datetime(data['Dep_Time']).dt.hour

    # Convert Duration to minutes
    def duration_to_minutes(x):
        parts = x.split()
        hours = int(parts[0][:-1])
        minutes = int(parts[1][:-1]) if len(parts) > 1 else 0
        return hours * 60 + minutes

    data['Duration'] = data['Duration'].apply(duration_to_minutes)

    return data


def visualize_data(data):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Flight Features vs Price', fontsize=16, fontweight='bold')
    
    axes[0, 0].scatter(data['Airline'], data['Price'], alpha=0.6)
    axes[0, 0].set_title('Airline vs Price')

    axes[1, 0].scatter(data['Dep_Time'], data['Price'], alpha=0.6)
    axes[1, 0].set_title('Departure Time vs Price')

    axes[0, 1].scatter(data['Duration'], data['Price'], alpha=0.6)
    axes[0, 1].set_title('Duration vs Price')

    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig('Flight_features.png', dpi=300)
    print("\n✓ Feature plots saved as 'Flight_features.png'")
    plt.show()


def prepare_and_split_data(data):
    feature_columns = ['Airline', 'Duration', 'Dep_Time']
    X = data[feature_columns]
    y = data['Price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\n=== Data Split ===")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    return X_train, X_test, y_train, y_test, feature_columns


def train_model(X_train, y_train, feature_names):
    model = LinearRegression()
    model.fit(X_train, y_train)

    print("\n=== Model Training Complete ===")
    print(f"Intercept: {model.intercept_:.2f}")

    print("\nCoefficients:")
    for name, coef in zip(feature_names, model.coef_):
        print(f"{name}: {coef:.2f}")

    return model


def evaluate_model(model, X_test, y_test, feature_names):
    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print("\n=== Model Performance ===")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.2f}")

    print("\n=== Feature Importance ===")
    importance = sorted(
        zip(feature_names, np.abs(model.coef_)),
        key=lambda x: x[1],
        reverse=True
    )
    for name, score in importance:
        print(f"{name}: {score:.2f}")

    return predictions


def make_prediction(model):
    print("\n=== Example Prediction ===")

    sample = pd.DataFrame({
        'Airline': [3],     # Example airline code
        'Duration': [180],  # 3 hours
        'Dep_Time': [14]    # 2 PM
    })

    predicted_price = model.predict(sample)[0]
    print(f"Predicted ticket price: {predicted_price:.2f}")


if __name__ == "__main__":
    data = load_and_explore_data(DATA_FILE)
    data = preprocess_data(data)
    visualize_data(data)

    X_train, X_test, y_train, y_test, feature_names = prepare_and_split_data(data)

    model = train_model(X_train, y_train, feature_names)

    predictions = evaluate_model(model, X_test, y_test, feature_names)

    make_prediction(model)

    print("\n=== PROJECT COMPLETE ===")
