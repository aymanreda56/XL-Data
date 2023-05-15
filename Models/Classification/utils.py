import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from pyspark.sql.functions import floor, col
from pyspark.ml.feature import VectorAssembler,StringIndexer, IndexToString

def encode(df, columns):
    for col in columns:
        categoryEncoder = StringIndexer(inputCol=col,outputCol=col+"_enc", handleInvalid='keep').fit(df)
        df = categoryEncoder.transform(df)
        df = df.withColumn(col+"_enc", df[col+"_enc"].cast('int'))
    return df

def decode(df, columns):
    #to decode back the encoded column
    for col in columns:
        converter = IndexToString(inputCol=col,outputCol=col[:-4]+'orig')
        df = converter.transform(df)
    return df

def floor_col (df, c):
    df = df.withColumn(c, floor(c).cast('int'))
    return df

def read_data(label_column='Rating', inputDf=None):
    if(inputDf):
        df= inputDf
    else:
        df= pd.read_csv('../../Dataset/Preprocessed_data.csv', on_bad_lines='skip')

    # Prepare the label column
    if label_column == 'Rating':
        df['Rating'] = df['Rating'].str.replace('"', '')
        df.dropna(inplace=True, subset=['Rating'])
        df['Rating'] = df['Rating'].astype(float).apply(np.floor).astype(int)
        df = df[df['Rating'] <= 5]

    # Prepare the features
    df['Free'] = df['Free'].astype(int)
    df = df[df['Ad Supported'].isin(['True', 'False'])]

    # encode the categorical columns
    disc_feats = [feat for feat in df.columns if type(df.iloc[0, df.columns.get_loc(feat)]) == str]
    for feat in disc_feats:
        if type(df.iloc[0, df.columns.get_loc(feat)]) == str:
            df[feat] = df[feat].map(df[feat].value_counts())/len(df)

    return df 

def hyperparameter_search(hyperparameters,clf, X_train, y_train):
    opt_params = {}

    random_search = RandomizedSearchCV(estimator=clf, param_distributions=hyperparameters, n_iter=30,\
                                    scoring='accuracy', n_jobs=4, cv=5, random_state=42)
    random_search.fit(X_train, y_train)

    opt_params = random_search.best_params_

    opt_params['accuracy'] = random_search.best_score_

    return opt_params
