import numpy as np
import pandas as pd
import sklearn.model_selection as ms
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

def show_prediction_results(header, actual_data, predictions):
    '''Show the results of the prediction.'''
    # Print the actual data and the predictions
    print("Inspect the contents of actual:")                 
    print(actual_data.values.T)     
    print("Inspect the contents of predictions:")
    print(predictions)      # Problem 4

    np_prediction = np.round(predictions).astype(int)
    np_actual = np.array(actual_data.astype(int))
    result = np_actual == np_prediction

    num_correct_predictions = np.count_nonzero(result)
    num_incorrect_predictions = np.count_nonzero(result == False)
    print('{0}: \n{1} correct, {2} incorrect, accuracy: {3}'    # Problem 5
        .format(header, num_correct_predictions, num_incorrect_predictions,
        (num_correct_predictions/(num_correct_predictions+num_incorrect_predictions))))
    return (num_correct_predictions/(num_correct_predictions+num_incorrect_predictions))    


def plot_confusion_matrix(actual, predicted):
    """Plot the confusion matrix with actual on x-axis and predicted on y-axis."""
    cm = metrics.confusion_matrix(actual, predicted, labels=[1, 0])
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['True', 'False'])
    ax.set_yticklabels(['True', 'False'])
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Confusion Matrix')
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, cm[i, j], ha='center', va='center', color='black')
    plt.show()


def analyze(titanic_df):
    """Analyze the titanic dataframe."""
    # Create predictors using "Pclass", "Age", "SibSp", "Parch", "Fare" as the predictors
    titanic_predictors_df = titanic_df[["Pclass", "Age", "SibSp", "Parch", "Fare"]]     # Problem 3
    # Create response (only 'Survived' column)
    titanic_response_df = titanic_df["Survived"]    # Problem 3

    (titanic_predictors_training_df, titanic_predictors_testing_df,
     titanic_response_training_df, titanic_response_testing_df) = \
        ms.train_test_split(titanic_predictors_df, titanic_response_df,
                            random_state=1)     # Problem 3

    # Train a logistic regression model on the training set.
    #algorithm = LogisticRegression(max_iter=100000)

    # Switch to the random forest classifier
    algorithm = RandomForestClassifier()
    model = algorithm.fit(titanic_predictors_training_df, titanic_response_training_df)
    predictions = model.predict(titanic_predictors_testing_df)

    show_prediction_results('Logistic regression with all specified independent vars', titanic_response_testing_df, predictions)
    accuracy = metrics.accuracy_score(titanic_response_testing_df, predictions)
    print('Sklearn accuracy: {0}'.format(accuracy))     # Problem 5

    # Plot confusion matrix
    plot_confusion_matrix(titanic_response_testing_df, predictions)


def main():
    """Main function for Titanic-Dataset analysis."""
    # Load the CSV file
    titanic_dataset_df = pd.read_csv('Titanic-Dataset.csv')     # Problem 1
    # Delete all rows that have invalid values ("na")
    titanic_dataset_cleaned_df = titanic_dataset_df.dropna()    # Problem 2
    # Call the analyze function
    analyze(titanic_dataset_cleaned_df)


if __name__ == "__main__":
    main()
