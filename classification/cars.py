def main():
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    
    df = pd.read_csv('cars.csv')
    print(df.head())  # Display first few rows
    
    # Drop rows with missing values in enginesize or price
    df.dropna(subset=['enginesize', 'price'], inplace=True)
    
    # Perform linear regression: enginesize as predictor, price as response
    X = df[['enginesize']]
    y = df['price']
    
    model = LinearRegression()
    model.fit(X, y)
    
    print(f"Linear Regression Results:")
    print(f"Coefficient (slope): {model.coef_[0]}")
    print(f"Intercept: {model.intercept_}")
    print(f"R-squared: {model.score(X, y)}")

if __name__ == "__main__":
    main()