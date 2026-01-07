import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


def load_and_explore_data(filename):
    data = pd.read_csv(filename)
    data.columns = data.columns.str.strip()

    print("=== Flight Price Data ===")
    print("\nFirst 5 rows:")
    print(data.head())

    print(f"\nDataset shape: {data.shape[0]} rows, {data.shape[1]} columns")

    print("\nBasic statistics:")
    print(data.describe())

    print("\nColumn names:")
    print(list(data.columns))

    return data


def preprocess_data(data):
    # Encode Airline as numbers
    data['Airline'] = data['Airline'].astype('category').cat.codes

    # Convert departure time to hour
    data['Dep_Time'] = pd.to_datetime(data['Dep_Time']).dt.hour

    # Convert duration to minutes
    def duration_to_minutes(x):
        hours = 0
        minutes = 0
        if 'h' in x:
            hours = int(x.split('h')[0])
        if 'm' in x:
            minutes = int(x.split('m')[0].split()[-1])
        return hours * 60 + minutes

    data['Duration'] = data['Duration'].apply(duration_to_minutes)

    return data


def visualize_features(data):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Flight Features vs Price', fontsize=16, fontweight='bold')

    axes[0, 0].scatter(data['Airline'], data['Price'], alpha=0.6)
    axes[0, 0].set_title('Airline vs Price')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].scatter(data['Dep_Time'], data['Price'], alpha=0.6)
    axes[0, 1].set_title('Departure Time vs Price')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].scatter(data['Duration'], data['Price'], alpha=0.6)
    axes[1, 0].set_title('Duration vs Price')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig('flight_features.png', dpi=300)
    print("\n✓ Feature plots saved as 'flight_features.png'")
    plt.show()


def prepare_features(data):
    feature_columns = ['Airline', 'Dep_Time', 'Duration']
    X = data[feature_columns]
    y = data['Price']

    print("\n=== Feature Preparation ===")
    print(f"Features (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")
    print(f"Feature columns: {feature_columns}")

    return X, y


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\n=== Data Split ===")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    return X_train, X_test, y_train, y_test


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

    for i, (name, score) in enumerate(importance, 1):
        print(f"{i}. {name}: {score:.2f}")

    return predictions


def compare_predictions(y_test, predictions, num_examples=5):
    print("\n=== Prediction Examples ===")
    print(f"{'Actual Price':<15} {'Predicted Price':<18} {'Error':<12}")
    print("-" * 55)

    for i in range(min(num_examples, len(y_test))):
        actual = y_test.iloc[i]
        predicted = predictions[i]
        error = actual - predicted

        print(f"${actual:<14.2f} ${predicted:<17.2f} ${error:<10.2f}")


def make_prediction(model, airline, dep_time, duration):
    sample = pd.DataFrame(
        [[airline, dep_time, duration]],
        columns=['Airline', 'Dep_Time', 'Duration']
    )

    predicted_price = model.predict(sample)[0]

    print("\n=== New Prediction ===")
    print(f"Airline code: {airline}")
    print(f"Departure hour: {dep_time}")
    print(f"Duration: {duration} minutes")
    print(f"Predicted price: ${predicted_price:,.2f}")

    return predicted_price


if __name__ == "__main__":
    print("=" * 70)
    print("FLIGHT PRICE PREDICTION PROJECT")
    print("=" * 70)

    data = load_and_explore_data('Data_Train.csv')
    data = preprocess_data(data)

    visualize_features(data)

    X, y = prepare_features(data)

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train, X.columns)

    predictions = evaluate_model(model, X_test, y_test, X.columns)

    compare_predictions(y_test, predictions, num_examples=10)

    make_prediction(model, airline=3, dep_time=14, duration=180)

    print("\n" + "=" * 70)
    print("✓ Project complete! Check saved plots.")
    print("=" * 70)
