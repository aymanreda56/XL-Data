from pyspark.sql.functions import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as ss
from scipy.stats import f_oneway

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
            elif i == j:
                corr[i,j] = 1
            else:
                corr[i,j] = cramers_v(df_disc, categorical_features[i], categorical_features[j])

    # Now plot the correlation matrix
    plt.figure(figsize=(20,10))
    sns.set(font_scale=1.4)
    sns.heatmap(corr,
                xticklabels=categorical_features,
                yticklabels=categorical_features,
                annot=True, cmap='coolwarm')
    plt.title('Categorical Correlation Matrix')
    plt.show()

    
def correlation_ratio(df, col1, col2):
    categories = df.select(col1).distinct().collect()
    values = df.select(col2).rdd.flatMap(lambda x: x).collect()

    group_variances = 0
    for category in categories:
        group_values = df.filter(col(col1) == category[col1]).select(col(col2)).rdd.flatMap(lambda x: x).collect()
        group_mean = np.mean(group_values)
        group_variances += len(group_values) * (group_mean - np.mean(values))**2

    total_variance = np.var(values) * (len(values) - 1)

    return (group_variances / total_variance)**.5

def mix_correlation_matrix(df):
    '''
    plot a correlation matrix for the categorical and continuous features in the dataset
    '''
    categ_features = [column for column, dtype in df.dtypes if dtype == 'string']
    num_features = [column for column, dtype in df.dtypes if dtype != 'string']
    
    corr = np.zeros((len(categ_features), len(num_features)))
    for i in range(len(categ_features)):
        for j in range(len(num_features)):
            corr[i, j] = correlation_ratio(df, categ_features[i], num_features[j])
    
    # now plot the correlation matrix
    plt.figure(figsize=(20,10))
    sns.set(font_scale=1.4)
    sns.heatmap(corr,
                xticklabels=num_features,
                yticklabels=categ_features,
                annot=True, cmap='coolwarm')
    plt.title('Mixed Correlation Matrix')
    plt.show()


def ANOVA_test(df, cat_feature, num_feature, cats=[]):
    '''
    This function computes the ANOVA test between a categorical feature and a numerical feature
    can take df, categorical feature and numerical feature as input
    and optional list of  specific categories of the categorical feature
    '''
    if len(cats)==0:
        cats = df[cat_feature].unique()
    # Compute ANOVA test
    groups = []
    for category in cats:
        groups.append(df[df[cat_feature]==category] [num_feature])
    f_value, p_value = f_oneway(*groups)

    # Print the F-value and p-value
    print("P-value: ", p_value)
    return p_value