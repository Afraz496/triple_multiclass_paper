import sys
import os
sys.path.append('src/')
import train
import test
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.decomposition import PCA
import logging
import pickle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import math
import time
from imblearn.over_sampling import RandomOverSampler
# Import the ML Models
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost
# TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import load_model
import yaml

with open('config/config.yml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

model_names = config['model_names']
predictor_column = config['predictor_column']
remapped_feature_names = config['feature_names']
def format_data(df):
    """
    Formats the total visits data to X, y
    """
    y = df[predictor_column]
    y = y.to_numpy()
    col_names = df.columns.tolist()
    col_names.remove(predictor_column)
    X = df[col_names]

    return X,y

def create_output_dirs(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def config_logger(output_dir):
    logger = logging.getLogger("Stanford Multiclass pipeline")
    logger.setLevel(logging.DEBUG)
    # create handlers
    fh = logging.FileHandler(os.path.join(output_dir, 'train_test_log.txt'))
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', default='results', help='Output dir for results')
    parser.add_argument('-d', '--data', default='', help='Path to CSV file of data, see README for format.')
    parser.add_argument('-f', '--filename', default= '', help='Path to CSV file of labels, see README for format.')
    parser.add_argument('-p', '--preprocessor', default = '', help = 'Path to Preprocessor pickle file for your current dataset')
    parser.add_argument('-models' , '--ml_models', default = '', help = 'Path to trained pickle Machine Learning Models')
    parser.add_argument('-t', '--train_size', default=0.7, help='Control the training data size, see README for format.')
    parser.add_argument('-remap', '--remap_file', default=None, help='Remap the Labels for Plots.')
    parser.add_argument('-s', '--remap_shap', default=None, help='Remap SHAP Feature names')
    return parser.parse_args()

def main():
    args = _parse_args()
    create_output_dirs(args.output_dir)
    logger = config_logger(args.output_dir)

    logger.info("Beginning Analysis")

    output_dir = args.output_dir

    # Load the preprocessor
    with open(args.preprocessor, 'rb') as f:
        preprocessor = pickle.load(f)

    try:
        all_data = pd.read_csv(args.data, index_col=False, delimiter=',', encoding='utf-8')
    except:
        all_data = pd.read_excel(args.data, index_col=False)

    train_size = float(args.train_size)  # Convert train_size argument to float
    train_data, test_data = train_test_split(all_data, train_size=train_size, random_state=42)

    # print some stats on data
    logger.info("Total samples: {}".format(len(all_data)))
    logger.info("    Train samples: {}".format(len(train_data)))
    logger.info("    Test samples: {}".format(len(test_data)))
    
    # Split into X_train and y_train
    X_train, y_train = format_data(train_data)
    X_test, y_test = format_data(test_data)

    from sklearn.preprocessing import LabelEncoder

# Check if y_train is already numerical
    if y_train.dtype.kind in 'bifc':  # b=bool, i=int, f=float, c=complex
        print("y_train is already numerical")
    else:
        print("y_train contains non-numerical values. Encoding now...")
        encoder = LabelEncoder()
        y_train = encoder.fit_transform(y_train)
        y_test = encoder.transform(y_test)

    df_numerical_features = X_train.select_dtypes(include='number')
    df_categorical_features = X_train.select_dtypes(include='object')

    # preprocess
    features = X_train.columns.to_list()
    numerical_features = df_numerical_features.columns.to_list()
    categorical_features = df_categorical_features.columns.to_list()
    norm_X_train = preprocessor.transform(X_train[features])
    norm_X_test = preprocessor.transform(X_test[features])
    features_processed = preprocessor.get_feature_names_out()
    
    # Calculate the number of unique classes
    n_classes = len(np.unique(y_train))

    all_models = []
    # Load the ML models    
    for root, dirs, files in os.walk(args.ml_models):
        for file in files:
            model = load_model(os.path.join(root, file))
            all_models.append(model)

    ml_dict = dict(zip(model_names, all_models))    
    
    # Get the original class names
    original_class_names = encoder.inverse_transform(range(n_classes))
    for name, model in ml_dict.items():
        test.plot_roc_neural_network(name, model, norm_X_test, y_test, n_classes, original_class_names, args.output_dir, filename=name+'_test_roc.png')
    
    if remapped_feature_names:
        test.shap_summary_neural_networks(ml_dict, remapped_feature_names, original_class_names, norm_X_test, args.output_dir)
    else:
        test.shap_summary_neural_networks(ml_dict, features_processed, original_class_names, norm_X_test, args.output_dir)
    
if __name__ == "__main__":
    main()
