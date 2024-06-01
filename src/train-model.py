# import libraries
import mlflow
import glob
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

def main(args):
    mlflow.autolog()

    df = get_data(args.training_data)

    X_train, X_test, y_train, y_test = split_data(df,args.target_feature)

    model = train_model(args.algorithm, X_train, X_test, y_train, y_test)

    eval_model(model, X_test, y_test)

def get_data(path):
    df = pd.read_csv(path)

    print(f'Modeling with {df.shape[1]} columns and {df.shape[0]} rows of data')
    
    return df

def split_data(df,target_feature):
    print("Splitting data...")
    X, y = df.drop(target_feature,axis=1), np.ravel(df[target_feature])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True, random_state=99)

    return X_train, X_test, y_train, y_test

def train_model(algorithm,X_train, X_test, y_train, y_test):
    print("Training model...")
    if algorithm == "gradient-boosting":
        model = GradientBoostingRegressor()
    if algorithm == "random-forest":
        model = RandomForestRegressor()
    else:
        model = LinearRegression()
    
    model.fit(X_train, y_train)

    mlflow.sklearn.save_model(model, args.model_output)

    return model


def eval_model(model, X_test, y_test):

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_pred, y_test)
    mlflow.log_param('MAE',mae)
    print('MAE:', mae)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--training_data", dest='training_data',
                        type=str)
    parser.add_argument("--algorithm", dest='algorithm',
                        type=str, default='linear-regression')
    parser.add_argument("--target_feature", dest='target_feature',
                        type=str, default='CPC')
    parser.add_argument("--model_output", dest='model_output',
                        type=str)

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    print("\n\n")
    print("*" * 60)

    args = parse_args()

    main(args)

    print("*" * 60)
    print("\n\n")
