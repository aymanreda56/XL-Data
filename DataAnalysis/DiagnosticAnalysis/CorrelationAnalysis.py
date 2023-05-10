from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import ChiSquareTest
import scipy.stats as ss
from pyspark.ml import Pipeline

import sys; sys.path.append("../../")
from DataPreprocessing.DataPreprocessing import encode_categ_features


def numerical_corr(df):
    '''
    Calculate the correlation between all numerical columns
    '''
    num_corr = df.toPandas().corr(numeric_only=True)

    plt.figure(figsize=(20,10))
    sns.heatmap(num_corr, annot=True, cmap='coolwarm')
    sns.set(font_scale=1.4)
    plt.title(f"Correlation matrix between Numerical features")
    plt.show()


def max_fn(a,b):
    return a if a>b else b

def min_fn(a,b):
    return a if a<b else b

def cramers_v(df, col1, col2):
    """ calculate Cramers V statistic for categorical-categorical association.
    """
    contingency_matrix = df.crosstab(col1, col2)
    contingency_matrix = contingency_matrix.toPandas().drop(col1+'_'+col2, axis=1)
    chi2 = ss.chi2_contingency(contingency_matrix)[0]
    n = contingency_matrix.values.sum()  
    phi2 = chi2 / n
    r, k = contingency_matrix.shape
    phi2corr = max_fn(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min_fn((kcorr-1), (rcorr-1)))


def categorical_corr(df):
    '''
    plot a correlation matrix for the categorical features in the dataset
    ''' 
    # Get the categorical features
    categorical_features = df.columns

    # Encode the categorical features
    df_disc = encode_categ_features(df, categorical_features)

    # Calculate the Cramer's V statistic for each pair of categorical features
    corr = np.zeros((len(categorical_features), len(categorical_features)))
    for i in range(len(categorical_features)):
        for j in range(len(categorical_features)):
            # matrix is symmetric, so don't need to calculate the lower triangle, place it with the value of the upper triangle
            if j < i:
                corr[i,j] = corr[j,i]
            else:
                corr[i,j] = cramers_v(df_disc, categorical_features[i], categorical_features[j])

    # Now plot the correlation matrix
    plt.figure(figsize=(20,20))
    plt.style.use('dark_background')
    sns.set(font_scale=1.4)
    sns.heatmap(corr,
                xticklabels=categorical_features,
                yticklabels=categorical_features,
                annot=True, cmap='coolwarm')
    plt.title('Categorical Correlation Matrix')
    plt.show()