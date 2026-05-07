import pandas as pd
import sklearn.linear_model as lm


def analyze(cherry_tree_df):
    print("Analyze function is running.")
    predictors_df = cherry_tree_df[['Diam','Height']]
    response_df = cherry_tree_df['Volume']

    algorithm = lm.LinearRegression()
    model = algorithm.fit(predictors_df, response_df)
    prediction = model.predict(predictors_df)

    # Print actual vs predicted values side by side
    print("Actual Volume vs Predicted Volume:")
    for actual, pred in zip(response_df, prediction):
        print(f"Actual: {actual:.2f} | Predicted: {pred:.2f}")


def main():
    cherrytree_df = pd.read_csv("CherryTree.csv")
    analyze(cherrytree_df)


if __name__ == "__main__":
    main()
