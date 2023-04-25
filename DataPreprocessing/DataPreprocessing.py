import os
import sys; sys.path.append("../")
import pandas as pd


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


def read_data(kind='train', features='all', encode=None, drop_cols=[]):
    
    '''
    Read the dataset and return a dataframe 
    TODO: return x_data and y_data instead of the whole dataframe
    '''
    dir= os.path.dirname(os.path.realpath(__file__))

    if kind == 'train':     path= os.path.join(dir, '../Dataset/train.csv')
    elif kind == 'val':     path= os.path.join(dir, '../Dataset/val.csv')
    elif kind == 'test':    path= os.path.join(dir, '../Dataset/test.csv')
    else:                   path= os.path.join(dir, '../Dataset/Google-Playstore.csv')

    df = pd.read_csv(path)
    
    # drop useless columns
    if len(drop_cols) > 0:
        df = df.drop(drop_cols, axis=1) 
    
    # extract the categorical fetaures only 
    if features==' Categorical':
        categ_features =[col for col in df.columns if type(df.iloc[0, df.columns.get_loc(col)]) == str] 
        df= df[categ_features] 
    
    # extract the numerical fetaures only
    elif features==' Numerical':
        num_features = [col for col in df.columns if type(df.iloc[0, df.columns.get_loc(col)]) != str] 
        df= df[num_features]
        for col in df.columns:
            df[col] = df[col].astype(float)

    # encode the categorical features
    if encode == 'label':
        for col in df.columns:
            if type(df.iloc[0, df.columns.get_loc(col)]) == str:
                df[col] = df[col].astype('category')
                df[col] = pd.factorize(df[col])[0]

    elif encode == 'oneHot':
        for col in df.columns:
            if type(df.iloc[0, df.columns.get_loc(col)]) == str:
                df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)
                df = df.drop(col, axis=1)


    return df


def missing_values(df, treatment='drop'):
    '''
    Dealing with the missing values in the dataset
    '''
    
    print(f'Total Number of rows : {len(df)}')
    
    # get #rows with missing values 
    print(f'Number of rows with missing values: {df.isnull().any(axis=1).sum()}')

    if treatment=='drop':
        df = df.dropna()
        print(f'Number of rows after dropping: {len(df)}') 

    elif treatment=='mean':
        for col in df.columns:
            df[col] = df[col].fillna(df[col].mean())

    elif treatment=='median':
        for col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    elif treatment=='mode':
        for col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    elif treatment== 'interpolate':
        df = df.interpolate(method='linear', axis=0).ffill().bfill()

    return df

def get_info(df):
    '''
    Get the info of the dataset
    '''
    print(f'Number of rows: {len(df)}, Number of columns: {len(df.columns)}')
    print(f'Available Features: {df.columns.tolist()}')


def detect_outliers(df, col):
    '''
    Detect #outliers in a certain column
    '''

    # calculate interquartile range (IQR)
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)

    iqr = q3 - q1 

    # calculate the lower and upper bound
    lower_bound = q1 - (1.5 * iqr) 
    upper_bound = q3 + (1.5 * iqr)

    # get the number of outliers
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    # outliers_index = outliers.index
    num_outliers = len(outliers)

    print(f'Number of outliers in {col}: {num_outliers}')


