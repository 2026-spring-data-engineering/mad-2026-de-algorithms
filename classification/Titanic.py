import numpy as np
import pandas as pd
import sklearn.model_selection as ms
import sklearn.linear_model as lm
import sklearn.metrics as metrics
import imblearn.over_sampling as ios
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier


def show_prediction_results(header, prediction, actual_data):
    np_prediction = np.round(prediction).astype(int)
    np_actual = np.array(actual_data).astype(int)
    result = np_actual == np_prediction

    num_correct_predictions = np.count_nonzero(result)
    num_incorrect_predictions = np.count_nonzero(result == False)
    print('{0}: {1} correct, {2} incorrect, accuracy: {3}.'
          .format(header, num_correct_predictions, num_incorrect_predictions,
          (num_correct_predictions/(num_correct_predictions+num_incorrect_predictions))))
    return (num_correct_predictions/(num_correct_predictions+num_incorrect_predictions))



def predict(titanic_df):
    titanic_df = titanic_df.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis='columns')
    all_independent_vars = titanic_df.columns.drop('Survived').values.tolist()

    titanic_predictors_df = titanic_df[all_independent_vars]
    titanic_response_df = titanic_df['Survived']

    (titanic_predictors_training_df, titanic_predictors_testing_df,
     titanic_response_training_df, titanic_response_testing_df) \
     = ms.train_test_split(titanic_predictors_df, titanic_response_df,
    test_size = 0.2) #, random_state=1)

    # algorithm = lm.LogisticRegression(max_iter=100000)
    algorithm = RandomForestClassifier()
    model = algorithm.fit(titanic_predictors_training_df, titanic_response_training_df)
    prediction = model.predict(titanic_predictors_testing_df)

    show_prediction_results('Logistic regression with all independent vars', prediction, titanic_response_testing_df)
    accuracy = metrics.accuracy_score(titanic_response_testing_df, prediction)
    print('Sklearn accuracy: {0}'.format(accuracy))


def main():
    titanic_df = pd.read_csv('Titanic-Dataset.csv')
    titanic_df = titanic_df.dropna()

    predict(titanic_df)


if __name__ == '__main__':
    main()