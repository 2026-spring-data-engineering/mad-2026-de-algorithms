import numpy as np
import pandas as pd
import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import seaborn as sns


def analyze(cherry_tree_df):
    print("Analyze function is running.")
    predictors_df = cherry_tree_df[['Diam','Height']]  # [['Diam','Height']]
    response_df = cherry_tree_df['Volume']

    # Split the data into training and testing sets
    predictors_training_df, predictors_testing_df, \
        response_training_df, response_testing_df = train_test_split(
            predictors_df, response_df, test_size=0.25, random_state=0
    )

    algorithm = lm.LinearRegression()
    model = algorithm.fit(predictors_training_df, response_training_df)

    # Print R^2 value on test set
    r2 = model.score(predictors_training_df, response_training_df)
    print(f"R^2 value (test set): {r2:.4f}")

    prediction = model.predict(predictors_testing_df)

    # Print actual vs predicted values side by side for test set
    print("Actual Volume vs Predicted Volume (Test Set):")
    for actual, pred in zip(response_testing_df, prediction):
        print(f"Actual: {actual:.2f} | Predicted: {pred:.2f}")
    
    mse = metrics.mean_squared_error(response_testing_df, prediction)
    rmse = np.sqrt(mse)
    print(f'RMSE: {rmse}')

    # 3D scatter plot with best-fit plane
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    x = cherry_tree_df['Diam']
    y = cherry_tree_df['Height']
    z = cherry_tree_df['Volume']
    ax.scatter(x, y, z, color='b', label='Data Points')

    # Fit plane using the model trained on all data
    predictors_all = cherry_tree_df[['Diam', 'Height']]
    model_all = lm.LinearRegression().fit(predictors_all, z)
    # Create grid to plot the plane
    x_surf, y_surf = np.meshgrid(
        np.linspace(x.min(), x.max(), 20),
        np.linspace(y.min(), y.max(), 20)
    )
    z_surf = model_all.intercept_ + model_all.coef_[0] * x_surf + model_all.coef_[1] * y_surf
    ax.plot_surface(x_surf, y_surf, z_surf, color='r', alpha=0.5, label='Best-fit Plane')

    ax.set_xlabel('Diameter')
    ax.set_ylabel('Height')
    ax.set_zlabel('Volume')
    ax.set_title('3D Scatter Plot with Best-fit Plane')
    plt.legend()
    plt.show()


def main():
    cherrytree_df = pd.read_csv("CherryTree.csv")
    analyze(cherrytree_df)


if __name__ == "__main__":
    main()
