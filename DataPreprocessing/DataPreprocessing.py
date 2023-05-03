import os
import sys; sys.path.append("../")
from sklearn.model_selection import train_test_split
import pandas as pd
import pyspark
from pyspark.ml.feature import Imputer, StringIndexer


def split_data():
    '''
    Split the dataset into train, validation and test set with ratio 60:20:20
    '''
    df = pd.read_csv('../Dataset/Google-Playstore.csv')

    train_test,val=  train_test_split(df, test_size=0.2)
    train,test= train_test_split(train_test, test_size=0.25)

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

    # elif treatment== 'interpolate':
    #     imputer.setStrategy("interpolate")

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
    
    outliers_index = outliers.index
    num_outliers = outliers.count()

    print(f'Number of outliers in {col}: {num_outliers}')
    return outliers_index
    

def remove_outliers(df, col):
    outliers_index= detect_outliers(df, col)
    df = df.drop(outliers_index)
    return df


#--------------------------------------------------------------------------
          
def remove_commas(df):
    '''
    Remove commas from a column to make only commas be for separating columns.
    Mainly for RDD purposes.

    '''
    for col in df.columns:  
        col_type= df[col].dtypes

        if col_type!='object':
            df[col] = df[col].astype(str)

        if col=='Installs':
            df[col] = df[col].str.replace(',', '')
        else:
            
            df[col] = df[col].str.replace(',', ' ')

        df[col]= df[col].astype(col_type)

    return df

    

def delimiter_to_comma(file_name='Google-Playstore'):
    df= pd.read_csv('../Dataset/'+file_name+'.csv',index_col=False,)
    df_new= remove_commas(df)
    df_new.to_csv('../Dataset/'+file_name+'-RDD'+'.csv', index=False)


# def replace_delimiters(delimiter, spark=None, kind='Google-Playstore'):
#     if spark!=None:
#         df= read_data(spark, kind=kind)
#         df.write.options(header=True, delimiter=delimiter).csv('../Dataset/'+kind+'RDD')
#     else: 
#         dir= os.path.dirname(os.path.realpath(__file__))
#         path= os.path.join(dir, '../Dataset/'+kind+'.csv')
#         new_path= os.path.join(dir, '../Dataset/'+kind+'RDD.xlsx')

#         df= pd.read_csv(path)
#         df.to_csv(new_path, index=False, sep=delimiter)

