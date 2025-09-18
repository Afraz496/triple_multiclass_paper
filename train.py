SEED = 2022
#!/usr/bin/env python
# Train and test
import sys
import os
sys.path.append('src/')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.decomposition import PCA
import logging
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import (
    TimeSeriesSplit,
    cross_val_score,
    cross_validate,
    train_test_split,
)
import time
# Import the ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import LSTM
from sklearn.svm import SVC
import yaml

with open('config/config.yml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

model_names = config['model_names']
predictor_column = config['predictor_column']

logger = logging.getLogger('Stanford Mulitclass pipeline')

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def fit_processor(X_train, numeric_features, categorical_features, output_dir):
    """
    Applies Simple Imputer to Categorical Features
    Applies One Hot Encoding to Categorical Features
    Applies Quantile Scaling to Numeric Features
    Returns and writes pickle file of the complete preprocessor

    Parameters
    ----------
    X_train : numpy
        Training Data in NumPy format
    numeric_features : list[string]
        List of Numeric Features
    categorical_features: list[string]
        List of Categorical Features
    output_dir: string
        Output directory to write the preprocessor to
    
    Returns
    -------
    preprocessor : sklearn.Preprocessor
        sklearn preprocessor fit on the training set
    """
    pipe_num = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler',  QuantileTransformer(output_distribution = 'normal', random_state=SEED))
    ])
    pipe_cat = Pipeline([
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ('impute', SimpleImputer(strategy='constant', fill_value=0))
        
    ]) 
    preprocessor = ColumnTransformer([
        ('num', pipe_num, numeric_features),
        ('cat', pipe_cat, categorical_features)
    ])
    logger.info("Preprocessing X_train")
    norm_X_train = preprocessor.fit(X_train)
    with open(os.path.join(output_dir, 'processor.pkl'),'wb') as f:
        pickle.dump(preprocessor, f)
    return preprocessor



def build_lstm_network(timesteps, features, output_classes):
    model = Sequential()
    model.add(LSTM(100, input_shape=(timesteps, features), return_sequences=True))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_lstm_network(X_train, y_train, timesteps, features, output_dir):
    output_classes = len(np.unique(y_train))
    X_train = X_train.reshape((-1, timesteps, features))  # Ensure the shape is correctly defined
    logger.info("Training LSTM Network")
    lstm_model = build_lstm_network(timesteps, features, output_classes)
    lstm_model.fit(X_train, y_train, epochs=10, batch_size=32)
    lstm_model.save(os.path.join(output_dir, 'models/lstm_model.h5'))
    return lstm_model


def train_svm(X_train, y_train, output_dir):
    logger.info("Training Support Vector Machine")
    svm_model = SVC(kernel='rbf', C=1.0, random_state=SEED)
    svm_model.fit(X_train, y_train)
    with open(os.path.join(output_dir, 'models/svm_model.pkl'), 'wb') as f:
        pickle.dump(svm_model, f)
    return svm_model


def build_neural_network(input_shape, output_classes):
    """
    Parameters
    ----------
        input_shape : int
            Dimensions for the input to the neural network. Note: Use X_train dimensions
        output_classes : int
            Number of classes to output the neural network over
    Returns
    -------
        model : keras.model
            Neural network built on specified params
    """
    model = Sequential()
    model.add(Dense(128, input_shape=(input_shape,), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(output_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_neural_networks(X_train, y_train, output_dir):
    """
    Parameters
    ----------
        X_train : np.array
            Training Data
        y_train : np.array
            Training prediction
        output_dir : String
            Output directory filepath
    Returns
    --------
        nn_model1, nn_model2, nn_model3 : (keras.model, keras.model, keras.model)
            Tuple of 3 neural network models
    """
    input_shape = X_train.shape[1]
    output_classes = len(np.unique(y_train))  # Assuming y_train is encoded properly

    # Build and train neural networks
    logger.info("Training First Neural Network")
    nn_model1 = build_neural_network(input_shape, output_classes)
    nn_model1.fit(X_train, y_train, epochs=10, batch_size=32)

    logger.info("Training Second Neural Network")
    nn_model2 = build_neural_network(input_shape, output_classes)
    nn_model2.fit(X_train, y_train, epochs=15, batch_size=64)

    logger.info("Training Third Neural Network")
    nn_model3 = build_neural_network(input_shape, output_classes)
    nn_model3.fit(X_train, y_train, epochs=20, batch_size=128)

     # Build and train an LSTM network
    logger.info("Training LSTM Neural Network")
    X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))  # Reshape for LSTM, assuming X_train is not sequence data
    lstm_model = train_lstm_network(X_train, y_train, 1, X_train.shape[1], output_dir)
    lstm_model.fit(X_train_lstm, y_train, epochs=10, batch_size=32)

    # Train SVM model
    logger.info("Training Support Vector Machine")
    svm_model = SVC(kernel='rbf', C=1.0, random_state=SEED)
    svm_model.fit(X_train, y_train)
    
    # Save models
    logger.info("Saving Neural Networks to specified output folder")
    nn_model1.save(os.path.join(output_dir, 'models/nn_model1.h5'))
    nn_model2.save(os.path.join(output_dir, 'models/nn_model2.h5'))
    nn_model3.save(os.path.join(output_dir, 'models/nn_model3.h5'))
    lstm_model.save(os.path.join(output_dir, 'models/lstm_model.h5'))
    with open(os.path.join(output_dir, 'models/svm_model.pkl'), 'wb') as f:
        pickle.dump(svm_model, f)

    return nn_model1, nn_model2, nn_model3, lstm_model, svm_model
