import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV


def read_data(label_column='Rating'):
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
