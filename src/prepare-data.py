# import libraries
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import RobustScaler

def main(args):
    df = get_data(args.input_data)

    cleaned_data = clean_data(df)

    imputed_data = impute(cleaned_data)

    scaled_data = scale_data(imputed_data)

    output_df = scaled_data.to_csv((Path(args.output_data)), index = False)

def get_data(path):
    df = pd.read_csv(path)

    print(f'Preparing {df.shape[1]} columns and {df.shape[0]} rows of data')
    print(df.dtypes)

    df = df.drop("keyword",axis=1)
    
    return df

def clean_data(df):
    zero_mask = (df['lower_bid']==df['upper_bid'])|(df['lower_bid']==0)
    df = df[~zero_mask]
    lower_mask = (df['lower_bid']>df['upper_bid'])|(df['lower_bid']>df['CPC'])|(df['upper_bid']<df['CPC'])
    df = df[~lower_mask]

    return df

def impute(df):
    for column in df.columns:
        if df[column].dtype in ['float64', 'int64']:
            fill_value = df[column].median()
        else:
            fill_value = df[column].mode()[0]
        
        df[column].fillna(fill_value, inplace=True)
    
    return df

def scale_data(df):
    scaler = RobustScaler()
    num_cols = df.select_dtypes(['float64', 'int64']).columns.to_list()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--input_data", dest='input_data',
                        type=str)
    parser.add_argument("--output_data", dest='output_data',
                        type=str)

    # parse args
    args = parser.parse_args()

    # return args
    return args


if __name__ == "__main__":
    print("\n\n")
    print("*" * 60)

    args = parse_args()

    main(args)

    print("*" * 60)
    print("\n\n")
