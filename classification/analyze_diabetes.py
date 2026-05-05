from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.model_selection as ms
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def show_confusion_matrix(actual, predicted):
    """Display a confusion matrix with TP in upper left and TN in bottom right, axes switched."""
    cm = confusion_matrix(actual, predicted)
    # Switch axes: x-axis = true label, y-axis = predicted label
    cm_switched = cm.T
    # Flip both axes to put TP in upper left and TN in bottom right
    cm_flipped = np.flipud(np.fliplr(cm_switched))
    plt.imshow(cm_flipped, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (TP upper left, TN bottom right)')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Positive', 'Negative'])
    plt.yticks(tick_marks, ['Positive', 'Negative'])
    plt.xlabel('True label')
    plt.ylabel('Predicted label')
    # Annotate cells
    thresh = cm_flipped.max() / 2.
    for i in range(cm_flipped.shape[0]):
        for j in range(cm_flipped.shape[1]):
            plt.text(j, i, format(cm_flipped[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm_flipped[i, j] > thresh else "black")
    plt.tight_layout()
    plt.show()


def show_prediction_results(actual_data, predictions):
    print("Actual:")
    print(actual_data.values.T)
    print("Predictions:")
    print(predictions)

    np_prediction = np.round(predictions).astype(int)
    np_actual = np.array(actual_data).astype(int)
    result = np_actual == np_prediction

    num_correct_predictions = np.count_nonzero(result)
    num_incorrect_predictions = np.count_nonzero(result == False)
    print('{0} correct, {1} incorrect, accuracy: {2}.'
        .format(num_correct_predictions, num_incorrect_predictions,
        (num_correct_predictions/(num_correct_predictions+num_incorrect_predictions))))
    return (num_correct_predictions/(num_correct_predictions+num_incorrect_predictions))


def analyze(diabetes_df):
    """Analyze the diabetes dataframe."""
    # Create predictors (all columns except 'Outcome')
    diabetes_predictors_df = diabetes_df.drop(columns=["Outcome"])
    # Create response (only 'Outcome' column)
    diabetes_response_df = diabetes_df[["Outcome"]]
    # For demonstration, print the shapes
    print("Predictors shape:", diabetes_predictors_df.shape)
    print("Response shape:", diabetes_response_df.shape)
    # ...existing code...
    print("Predictors columns:", diabetes_predictors_df.columns)
    print("Response columns:", diabetes_response_df.columns)

    diabetes_predictors_df = diabetes_df[["BloodPressure", "Age"]]

    (diabetes_predictors_training_df, diabetes_predictors_testing_df,
     diabetes_response_training_df, diabetes_response_testing_df) = \
        ms.train_test_split(diabetes_predictors_df, diabetes_response_df)

    # # Train a logistic regression model on all the data.
    # algorithm = LogisticRegression(max_iter=1000)
    # model = algorithm.fit(diabetes_predictors_df, diabetes_response_df.values.ravel())
    # print("Model trained. Coefficients:", model.coef_)
    # print("Intercept:", model.intercept_)
    
    # # Train a logistic regression model on the training set.
    # algorithm = LogisticRegression(max_iter=1000)
    # Switch to the random forest classifier.
    algorithm = RandomForestClassifier()
    model = algorithm.fit(diabetes_predictors_training_df, diabetes_response_training_df.values.ravel())
    predictions = model.predict(diabetes_predictors_testing_df)

    show_prediction_results(diabetes_response_testing_df, predictions)
    show_confusion_matrix(diabetes_response_testing_df.values.ravel(), predictions)


def main():
    """Main function for diabetes data analysis."""
    # Load the CSV file
    diabetes_df = pd.read_csv("pa_diabetes.csv")
    # Call the analyze function
    analyze(diabetes_df)


if __name__ == "__main__":
    main()
