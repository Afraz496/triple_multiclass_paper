import sys
import os
sys.path.append('src/')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.lines as mlines
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.dates as mdates
import shap  # package used to calculate Shap values
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
import train_test
import random
import delong
import yaml
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
from scipy.stats import sem, t
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

logger = logging.getLogger('Stanford Multiclass pipeline')

with open('config/config.yml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

model_names = config['model_names']
predictor_column = config['predictor_column']
shap_sampler = config['shap_sampler']
remapped_feature_names = config['feature_names']

#-----MATLAB PLOTS CUSTOMISATION (COLORBLIND) ----#
RGB_val = 255

color01= (0,107,164)  # Blue wanted
color04= (255,128,14)  # red wanted
Colors = [color01, color04]

# Creating a blue red palette transition for graphics
Colors= [(R/RGB_val,G/RGB_val,B/RGB_val) for idx, (R,G,B) in enumerate(Colors)]
n = 256

# Start of the creation of the gradient
Color01= ListedColormap(Colors[0], name='Color01', N=None)
Color04= ListedColormap(Colors[1], name='Color04', N=None)
top = cm.get_cmap(Color01,128)
bottom = cm.get_cmap(Color04,128)
newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                        bottom(np.linspace(0, 1, 128))))

mymin0 = newcolors[0][0]
mymin1 = newcolors[0][1]
mymin2 = newcolors[0][2]
mymin3 = newcolors[0][3]
mymax0 = newcolors[255][0]
mymax1 = newcolors[255][1]
mymax2 = newcolors[255][2]
mymax3 = newcolors[255][3]

GradientBlueRed= [np.linspace(mymin0, mymax0,  n),
                    np.linspace(mymin1, mymax1,  n),
                    np.linspace(mymin2, mymax2,  n),
                    np.linspace(mymin3, mymax3,  n)]
GradientBlueRed_res =np.transpose(GradientBlueRed)
newcmp = ListedColormap(GradientBlueRed_res, name='tableau-colorblind-10-first2')


def get_color_blind_friendly_colors(n_classes):
    """
    Generate a list of color-blind friendly colors.
    Parameters
    ----------
        n_classes : int
            Number of Classes
    Returns
    -------
        colors : List[String]
            List of colorblind friendly colors
    """
    # Predefined color-blind friendly color palette (can be expanded)
    palette = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', 
               '#984ea3', '#999999', '#e41a1c', '#dede00']
    color_cycle = cycle(palette)
    colors = [next(color_cycle) for _ in range(n_classes)]
    return colors

def calculate_multiclass_roc_confidence_interval(y_true, y_scores, n_bootstraps, confidence_interval):
    """
    Calculate multiclass roc CIs
    
    Parameters
    ----------
        y_true : np.array
            True prediction values
        y_scores : np.array
            Predicted prediction values
        n_bootstraps : int
            Number of samples to bootstrap over
        confidence_interval : double
            Confidence interval to take, see README (default = 0.95)
    Returns
    -------
        roc_intervals : List[tuple]
            List of each class's CIs in tuples (CI_Lower, CI_Upper)
    """
    n_classes = y_true.shape[1]
    roc_intervals = []

    for i in range(n_classes):
        # Calculate confidence intervals for ROC curve
        confidence_lower, confidence_upper = calculate_roc_confidence_interval(
            y_true[:, i], y_scores[:, i], n_bootstraps, confidence_interval
        )
        roc_intervals.append((confidence_lower, confidence_upper))

    return roc_intervals

def calculate_roc_confidence_interval(y_true, y_scores, n_bootstraps, confidence_interval):
    """
    Calculate the ROC CI for a single class
    Parameters
    ----------
        y_true : np.array
            True prediction values
        y_scores : np.array
            Predicted prediction values
        n_bootstraps : int
            Number of samples to bootstrap
        confidence_interval : double
            Confidence interval to take, see README (default = 0.95)
    Returns
    -------
        confidence_lower, confidence_upper : tuple(double, double)
            Lower CI, Upper CI
    """
    bootstrapped_scores = []

    for _ in range(n_bootstraps):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        bootstrap_y = y_true[indices]
        bootstrap_scores = y_scores[indices]
        fpr_bs, tpr_bs, _ = roc_curve(bootstrap_y, bootstrap_scores)
        roc_auc_bs = auc(fpr_bs, tpr_bs)
        bootstrapped_scores.append(roc_auc_bs)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    confidence_lower = sorted_scores[int(0.5 * n_bootstraps * (1 - confidence_interval))]
    confidence_upper = sorted_scores[int(0.5 * n_bootstraps * (1 + confidence_interval))]

    return confidence_lower, confidence_upper

def plot_roc_neural_network(name, model, X_test, y_test, n_classes, class_names, output_dir, filename='test-roc.png', confidence_interval=0.95, n_bootstraps=1000):
    """
    Parameters
    ----------
        name : String
            Name of the ML Model
        model : keras.model
            Keras model
        X_test : np.array
            Test set on features
        y_test : np.array
            Test set on prediction
        n_classes : int
            Number of prediction classes
        class_names : list
            List of original class names
        output_dir : String
            Output directory
        filename : String
            Output filename
        confidence_interval : float, optional
            Confidence interval for ROC curve, default is 0.95
        n_bootstraps : int, optional
            Number of bootstraps for confidence interval estimation, default is 1000
    Returns
    -------
        <None>
    """
    print(name)
    # Binarize the y_test in a one-vs-all fashion
    y_test_binarized = label_binarize(y_test, classes=range(n_classes))

    if name == 'LSTM':
        # Reshape input data for LSTM model
        X_test_reshaped = np.expand_dims(X_test, axis=1)
    else:
        X_test_reshaped = X_test
    plt.figure(figsize=(8, 6))

    # Get color-blind friendly colors
    colors = get_color_blind_friendly_colors(n_classes)

    # Predict probabilities
    # Predict probabilities for neural network models
    if name != 'SVM':
        y_pred_prob = model.predict(X_test_reshaped)
    else:  # Handle SVM model
        decision_scores = model.decision_function(X_test_reshaped)
    

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    roc_intervals = dict()

    for i in range(n_classes):
        if name != 'SVM':
            fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_prob[:, i])
        else:  # Handle SVM model
            fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], decision_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Calculate confidence intervals for ROC curve for neural network models
    if name != 'SVM':
        roc_intervals = calculate_multiclass_roc_confidence_interval(y_test_binarized, y_pred_prob, n_bootstraps, confidence_interval)

    # Plot ROC curve for each class
    for i, color in zip(range(n_classes), colors):
        if name != 'SVM':
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'class: {class_names[i]}, model: {name} (area = {roc_auc[i]:0.2f}, CI = [{roc_intervals[i][0]:0.2f}-{roc_intervals[i][1]:0.2f}])')
        else:  # Handle SVM model
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'class: {class_names[i]}, model: {name} (area = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC')
    plt.legend(loc="lower right", prop={'size': 7})
    plt.savefig(os.path.join(output_dir, filename))

def plot_roc_neural_networks(ml_dict, X_test, y_test, n_classes, class_names, output_dir, filename='test-roc.png'):
    """
    Parameters
    ----------
        ml_dict : dict{String}{object}
            Dictionary of ML Models
        X_test : np.array
            Test set on features
        y_test : np.array
            Test set on prediction
        n_classes : int
            Number of prediction classes
        class_names : list
            List of original class names
        output_dir : String
            Output directory
        filename : String
            Output filename
    Returns
    -------
        <None>
    """
    # Binarize the y_test in a one-vs-all fashion
    y_test_binarized = label_binarize(y_test, classes=range(n_classes))

    plt.figure(figsize=(8, 6))

    # Get color-blind friendly colors
    colors = get_color_blind_friendly_colors(n_classes)

    for name, model in ml_dict.items():
        print(name)
        print(model)
        # Predict probabilities
        if isinstance(model, Sequential):  # Neural network model
            X_test_reshaped = X_test.reshape(-1, 1, 27)  # Assuming 27 features
            y_pred_prob = model.predict(X_test_reshaped)
        else:  # For other models, no reshaping needed
            X_test_reshaped = X_test
            y_pred_prob = model.predict_proba(X_test_reshaped)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot ROC curve for each class
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'ROC curve of class {class_names[i]} for model {name} (area = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC for Neural Networks')
    plt.legend(loc="lower right", prop={'size': 7})
    plt.savefig(os.path.join(output_dir, filename))

def plot_roc(ml_models, norm_X_test, y_test, output_dir, filename='test-roc.png'):
    """ Plots an ROC AUC Curve with 95% Confidence Intervals
    Arguments:
        ml_models {Dict} -- Model Name as Key, Pickle ML models as Value
        norm_X_test {NumPy Array} -- Preprocessed X test xalues
        y_test {NumPy Array} -- y test values
        output_dir {String} -- Output Path to save ROC Curve
    """
    plt.figure(figsize=(6,6))
    for name, model in ml_models.items():
        y_pred_prob = model.predict_proba(norm_X_test)
        fpr, tpr, a = delong.get_prediction_stats(y_test, y_pred_prob[:,1])
        ci = delong.compute_stats(0.95,y_pred_prob[:,1],y_test)
        plt.plot(fpr, tpr, lw=2, label= name + ' ROC curve - (area = %(a)0.2f) (%(left)0.2f, %(right)0.2f)' % {'a':a, 'left': ci[0], 'right': ci[1]})
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Linear ROC - Test Set')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, filename))

def MSE(y_true,y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return mse

def shap_values_to_list(shap_values, class_names):
    shap_as_list=[]
    for i in range(len(class_names)):
        shap_as_list.append(shap_values[:,:,i])
    return shap_as_list

def shap_summary_neural_networks(models, features_processed, class_names, X_test, output_dir, shap_sampler=100, remapped_feature_names=None):
    """
    Generate SHAP summary plots for neural networks and save them.
    Parameters
    ----------
        models : dict
            Dictionary of model names to trained Keras models.
        features_processed : list of str
            List of preprocessed feature names.
        class_names : list of str
            Names of classes for the model outputs.
        X_test : np.ndarray
            Test data features.
        output_dir : str
            Output directory filepath.
        shap_sampler : int, optional
            Number of samples to use for SHAP background data.
        remapped_feature_names : list of str, optional
            Remapped feature names to use instead of features_processed.
    Returns
    -------
        None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Use remapped feature names if provided
    if remapped_feature_names:
        features_to_use = remapped_feature_names
    else:
        features_to_use = features_processed
        
    # Define custom colors for each class
    colorblind_colors = {
        "Negative": (0, 107, 164),  # Blue
        "Influenza A": (255, 128, 14),  # Orange
        "SARS-CoV-2": (171, 171, 171),  # Grey
        "RSV": (89, 89, 89)  # Dark Grey
    }
    # Normalize colors
    colorblind_colors = {k: (r/255, g/255, b/255) for k, (r, g, b) in colorblind_colors.items()}
    
    for name, model in models.items():
        print(f"Processing SHAP values for model: {name}")
        try:
            X_test_sample = shap.sample(X_test, shap_sampler)
            
            if hasattr(model, 'predict_proba'):
                explainer = shap.KernelExplainer(model.predict_proba, X_test_sample)
            else:
                explainer = shap.KernelExplainer(model.predict, X_test_sample)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_test_sample)
            
            # Initialize shap_as_list
            shap_as_list = []
            
            if name == "SVM":
                # For SVM, shap_values might be 2-dimensional
                if isinstance(shap_values, list):
                    shap_as_list = shap_values
                else:
                    if shap_values.ndim == 2:
                        shap_as_list = [shap_values]  # Wrap in a list for consistency
                    else:
                        shap_as_list = shap_values_to_list(shap_values, class_names)
                        
            elif "NNet" in name or "nn_model" in name:
                # For neural networks, shap_values might be a list of arrays
                if isinstance(shap_values, list) and len(shap_values) == len(class_names):
                    shap_as_list = shap_values
                elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                    # Convert 3D array to list format
                    shap_as_list = shap_values_to_list(shap_values, class_names)
                else:
                    print(f"Warning: Unexpected SHAP values shape for neural network {name}: {type(shap_values)}")
                    if isinstance(shap_values, np.ndarray):
                        print(f"SHAP values shape: {shap_values.shape}")
                    # Try to handle as list anyway
                    if isinstance(shap_values, list):
                        shap_as_list = shap_values
                    else:
                        shap_as_list = [shap_values]
            else:
                # Default handling for other model types
                if isinstance(shap_values, list):
                    shap_as_list = shap_values
                else:
                    shap_as_list = [shap_values]
            
            # Ensure we have valid shap_as_list
            if not shap_as_list:
                print(f"Error: No valid SHAP values for model {name}")
                continue
                
            # Plot
            plt.figure(figsize=(20, 20))
            shap.summary_plot(shap_as_list, features=X_test_sample, feature_names=features_to_use, 
                              class_names=class_names, show=False, plot_type='bar')
            
            plt.title(f'SHAP Summary Plot of {name}')
            plt.xlabel('Mean SHAP Value')
            
            # Manually set the colors for the bars
            ax = plt.gca()
            patches = ax.patches
            if patches:  # Only try to color if we have patches
                try:
                    # Get the legend to map colors
                    legend = ax.get_legend()
                    if legend:
                        for i, (patch, class_name) in enumerate(zip(patches, class_names)):
                            if class_name in colorblind_colors:
                                patch.set_color(colorblind_colors[class_name])
                except Exception as color_error:
                    print(f"Warning: Could not apply custom colors: {color_error}")
            
            plt.savefig(os.path.join(output_dir, f'SHAP_summary_{name}.png'), bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error processing SHAP values for model {name}: {e}")
            import traceback
            traceback.print_exc()
            continue

def renamed_shap_summary_neural_networks(models, features_processed, X_test, output_dir, class_names):
    """
    Parameters
    ----------
        models : dict{String}{keras.model}
            Dictionary of ML Models
        features_processed : List[String]
            List of preprocessed features
        X_test : np.array
            Test data features
        output_dir : String
            Output Directory filepath
        class_names : List[String]
            List of class names
    Returns
    -------
        <None>
    """
    for name, model in models.items():
        explainer = shap.KernelExplainer(model.predict, X_test)
        shap_values = explainer.shap_values(X_test)
        colors = get_color_blind_friendly_colors(len(class_names))

        plt.figure(figsize=(20, 20))
        for i, class_name in enumerate(class_names):
            values = np.array(shap_values[i])
            mean_shap = np.abs(values).mean(0)
            sorted_feature_indices = np.argsort(-mean_shap)
            plt.barh(np.arange(len(sorted_feature_indices)) + i * 0.1, mean_shap[sorted_feature_indices], 
                     color=colors[i], height=0.1, label=class_name)
        
        plt.yticks(np.arange(len(sorted_feature_indices)), np.array(features_processed)[sorted_feature_indices])
        plt.title(f'SHAP summary plot for {name}')
        plt.xlabel('SHAP Value (impact on model output)')

        # Create custom legend
        legend_handles = [mlines.Line2D([], [], color=colors[i], marker='o', linestyle='None',
                            markersize=10, label=class_names[i]) for i in range(len(class_names))]
        plt.legend(handles=legend_handles, title="Classes")

        plt.savefig(os.path.join(output_dir, f'SHAP_summary_{name}.png'), bbox_inches='tight')