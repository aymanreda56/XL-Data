from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import matplotlib.pyplot as plt
import seaborn as sns
 

def numerical_corr(df):
    '''
    Calculate the correlation between all numerical columns
    '''
    num_corr = df.toPandas().corr(numeric_only=True)

    plt.figure(figsize=(8,6))
    sns.heatmap(num_corr, annot=True, cmap='coolwarm')
    plt.title(f"Correlation matrix between features")
    plt.show()


def numerical_col_corr(df, col1, col2):
    '''
    Calculate the correlation between two numerical columns
    '''
    df.select(corr(col1,col2)).show()
    draw_num_corr(df, col1, col2)


def draw_num_corr(df, col1, col2):
    '''
    Draw the correlation between two numerical columns
    '''
    # take a sample of 50 rows to represent the correlation
    df_sample = df.toPandas().sample(n=50, random_state=42)

    plt.rcParams['axes.grid'] = False
    plt.rcParams['figure.dpi'] = 300
    plt.style.use('dark_background')
    plt.figure()
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.scatter(df_sample[col1], df_sample[col2])
    plt.title("Correlation between " + col1 + " and " + col2)
    plt.show()
