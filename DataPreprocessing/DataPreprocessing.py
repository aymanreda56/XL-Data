import os
import sys; sys.path.append("../")
import pandas as pd
import pyspark
from pyspark.ml.feature import Imputer, StringIndexer

def split_data():
    '''
    Split the dataset into train, validation and test set with ratio 60:20:20
    '''
    df = pd.read_csv('../Dataset/Google-Playstore.csv')
    df = df.sample(frac=1).reset_index(drop=True)

    train = df[:int(0.6*len(df))]
    val = df[int(0.6*len(df)):int(0.8*len(df))]
    test = df[int(0.8*len(df)):]

    train.to_csv('../Dataset/train.csv', index=False)
    val.to_csv('../Dataset/val.csv', index=False)
    test.to_csv('../Dataset/test.csv', index=False)


def read_data(spark, kind='train', features='all', encode=False, drop_cols=[]):
    
    '''
    Read the dataset and return a dataframe 


    kind: 'train', 'val', 'test', 'all' ----> all is the default
    features: 'all', 'Categorical', 'Numerical' ----> all is the default
    encode: True, False (encode categorical features) 
    drop_cols: list of columns to drop
    
    TODO: return x_data and y_data instead of the whole dataframe
    '''
    dir= os.path.dirname(os.path.realpath(__file__))

    if kind == 'train':     path= os.path.join(dir, '../Dataset/train.csv')
    elif kind == 'val':     path= os.path.join(dir, '../Dataset/val.csv')
    elif kind == 'test':    path= os.path.join(dir, '../Dataset/test.csv')
    else:                   path= os.path.join(dir, '../Dataset/Google-Playstore.csv')


    df = spark.read.csv(path, header=True, inferSchema= True)
    
    numerical_cols= ["Rating", "Rating Count", "Minimum Installs", "Maximum Installs","Price"]

    # extract the categorical fetaures only 
    if features=='Categorical':
        df= df.select([column for column in df.columns if column not in numerical_cols])
        
    
    # extract the numerical fetaures only
    elif features=='Numerical':
        df= df.select(numerical_cols)
        
        df= df.withColumn("Rating", df["Rating"].cast("float"))
        df= df.withColumn("Rating Count", df["Rating Count"].cast("int"))
        df= df.withColumn("Minimum Installs", df["Minimum Installs"].cast("int"))
        df= df.withColumn("Maximum Installs", df["Maximum Installs"].cast("int"))
        df= df.withColumn("Price", df["Price"].cast("float"))

   
    # encode the categorical features 
    if encode :
        cols_to_drop=[]
        for col in df.columns:
            if col not in numerical_cols:
                indexer = StringIndexer(inputCol=col, outputCol=col+"_index")
                df = indexer.setHandleInvalid("keep").fit(df).transform(df) 
                cols_to_drop.append(col)

        df = df.drop(*cols_to_drop)

    # drop useless columns 
    if len(drop_cols) > 0:
        df = df.drop(*drop_cols) 

    return df


def missing_values(df, treatment='drop', cols=[]):
    '''
    Dealing with the missing values in the dataset
    '''
    
    print(f'Total Number of rows : {df.count()}')
    
    # get #rows with missing values 
    for col in df.columns:
        df = df.withColumn(col, df[col].cast("string"))
    df_missing = df.filter(df[col].isNull()).count()
    print(f'Number of rows with missing values: {df_missing}')

    if treatment=='drop':
        df = df.na.drop()
        print(f'Number of rows after dropping: {df.count()}') 
        return df

    imputer = Imputer(inputCols=cols, outputCols=["{}_imputed".format(c) for c in cols])

    if treatment=='mean':
        imputer.setStrategy("mean")

    elif treatment=='median':
        imputer.setStrategy("median")

    elif treatment=='mode':
        imputer.setStrategy("mode")        

    elif treatment== 'interpolate':
        imputer.setStrategy("interpolate")

    df = imputer.fit(df).transform(df)
    return df


def get_info(df):
    '''
    Get the info of the dataset
    '''
    print(f'Number of rows: {df.count()}, Number of columns: {len(df.columns)}')
    print(f'Available features: {df.columns}')

    df.describe().show()


def detect_outliers(df, col):
    '''
    Detect #outliers in a certain column
    '''

    # calculate interquartile range (IQR)
    q1 = df.approxQuantile(col, [0.25], 0.0)[0]
    q3 = df.approxQuantile(col, [0.75], 0.0)[0]

    iqr= q3 - q1

    # calculate the lower and upper bound
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    # get the number of outliers 
    outliers = df.filter((df[col] < lower_bound) | (df[col] > upper_bound))
    
    # outliers_index = outliers.index
    num_outliers = outliers.count()

    print(f'Number of outliers in {col}: {num_outliers}')


