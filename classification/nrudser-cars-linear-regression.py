import pandas as pd
from sklearn.linear_model import LinearRegression

def predict(cars_df):
    """Perform linear regression: price ~ enginesize."""
    # Create predictor dataframe (only 'enginesize' column)
    cars_predictors_df = cars_df[["enginesize"]]
    # Create response series (only 'price' column)
    cars_response_series = cars_df["price"]

    # Fit linear regression
    algorithm = LinearRegression()
    model = algorithm.fit(cars_predictors_df, cars_response_series)
    
    # Print R^2 value
    r2 = model.score(cars_predictors_df, cars_response_series)
    print(f"R^2 value: {r2:.4f}")

    prediction = model.predict(cars_predictors_df)

    # Print actual vs predicted values side by side
    print("Actual Price vs Predicted Price")
    for actual, pred in zip(cars_response_series, prediction):
        print(f"Actual: {actual:.2f} | Predicted: {pred:.2f}")


def main():
    """Main function for cars-dataset analysis."""
    # Load the CSV file
    cars_dataset_df = pd.read_csv('cars.csv')
    # Perform linear regression analysis
    predict(cars_dataset_df)


if __name__ == "__main__":
    main()
