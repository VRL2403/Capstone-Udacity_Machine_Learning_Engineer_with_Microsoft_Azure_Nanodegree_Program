from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.core import Workspace, Dataset
from sklearn.preprocessing import StandardScaler

run = Run.get_context()
ws = run.experiment.workspace


def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0,
                        help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--data', type=str, help="Loading dataset")
    parser.add_argument('--max_iter', type=int, default=100,
                        help="Maximum number of iterations to converge")

    args = parser.parse_args()

    # split data to train and test sets
    dataset = Dataset.get_by_name(ws, name='diabetes')
    dataset = dataset.to_pandas_dataframe()

    # Scaling data
    scaler = StandardScaler()
    scaler.fit(dataset.drop('Outcome', axis=1))
    scaler_features = scaler.transform(dataset.drop('Outcome', axis=1))
    df_feat = pd.DataFrame(scaler_features, columns=dataset.columns[:-1])
    # appending the outcome feature
    df_feat['Outcome'] = dataset['Outcome'].astype(int)
    dataset = df_feat.copy()

    x = dataset.drop(columns=['Outcome'])
    y = dataset['Outcome']
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(
        C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)

    os.makedirs('outputs', exist_ok=True)

    joblib.dump(model, 'outputs/model.joblib')

    run.log("Accuracy", np.float(accuracy))


if __name__ == '__main__':
    main()
