import os
import sys; sys.path.append("../")
import pandas as pd
import pyspark
from pyspark.ml.feature import Imputer, StringIndexer
from pyspark.sql.functions import regexp_replace, isnan, when, count, col,mode, split
import matplotlib.pyplot as plt
import seaborn as sns


#  ======================================= Generic functions  =======================================

def split_spark_df(df):
    '''
    Split the dataset into train, validation and test set with ratio 60:20:20
    '''
    train_test,val=  df.randomSplit([0.8,0.2])
    train,test= train_test.randomSplit([0.75,0.25])

    train.toPandas().to_csv('../Dataset/train.csv', index=False)
    val.toPandas().to_csv('../Dataset/val.csv', index=False)
    test.toPandas().to_csv('../Dataset/test.csv', index=False)

    return train,val,test


def read_data(spark, file_name='all', features='all', encode=False, useless_cols=[]):
    
    '''
    Read the dataset and return a dataframe 
    TODO: return x_data and y_data instead of the whole dataframe for the model to be trained

    file_name: 'train', 'val', 'test', 'all' ----> all is the default
    features: 'all', 'Categorical', 'Numerical' ----> all is the default
    encode: True, False (encode categorical features)     
    useless_cols: list of columns to be removed from the dataset (default: empty list)

    '''
    dir= os.path.dirname(os.path.realpath(__file__))

    if file_name == 'train':     path= os.path.join(dir, '../Dataset/train.csv')
    elif file_name == 'val':     path= os.path.join(dir, '../Dataset/val.csv')
    elif file_name == 'test':    path= os.path.join(dir, '../Dataset/test.csv')
    else:                        path= os.path.join(dir, '../Dataset/Google-Playstore.csv')

    # colab_path = '/content/drive/MyDrive/Google-Playstore.csv'
    df = spark.read.csv(path, header=True, inferSchema= True)
    
    numerical_cols= ["Rating", "Rating Count", "Minimum Installs", "Maximum Installs","Price"]

    # cast the numerical columns to their correct type
    df= df.withColumn("Rating", df["Rating"].cast("float"))
    df= df.withColumn("Rating Count", df["Rating Count"].cast("int"))
    df= df.withColumn("Minimum Installs", df["Minimum Installs"].cast("int"))
    df= df.withColumn("Maximum Installs", df["Maximum Installs"].cast("int"))
    df= df.withColumn("Price", df["Price"].cast("float"))

    # extract the categorical fetaures only 
    if features=='Categorical':
        df= df.select([column for column in df.columns if column not in numerical_cols])
        
    # extract the numerical fetaures only
    elif features=='Numerical':
      df= df.select(numerical_cols)  

    # encode the categorical features 
    if encode:
        df= encode_categ_features(df,numerical_cols)
        
    # remove the useless columns
    if len(useless_cols) > 0:
        df = remove_useless_col(df, useless_cols)

    return df


def get_info(df):
    '''
    Get the info of the dataset
    '''
    print(f'Number of rows: {df.count()}, Number of columns: {len(df.columns)}')    
    df.describe().show()


def remove_useless_col(df, cols=[]):
    '''
    Remove the useless columns from the dataset
    '''
    df = df.drop(*cols) 
    return df

def encode_categ_features(df, cols=[]):
    '''
    Encode the categorical features to numerical features
    '''
    if len(cols)==0:
        numerical_cols= ["Rating", "Rating Count", "Minimum Installs", "Maximum Installs","Price"]
        cols= [column for column in df.columns if column not in numerical_cols]

    cols_to_drop=[]

    for col_name in df.columns:
        if col_name not in cols:
            indexer = StringIndexer(inputCol=col_name, outputCol=col_name+"_index")
            df = indexer.setHandleInvalid("keep").fit(df).transform(df) 
            cols_to_drop.append(col_name)

    df = remove_useless_col(df,cols_to_drop)
    return df

# ======================================= Data Cleaning =======================================
# ---------------------------------------- Outliers ----------------------------------------

def detect_outliers(df):
    '''
    Detect outliers in all numerical columns
    Detect rows with outliers in any of its columns and if it has more than 1 outlier in any of its columns
    then it will be considered as an outlier row and can be removed 
    '''
    # Select only the numerical columns
    df_num = df

    # get the total number of rows in the DataFrame
    total_rows = df_num.count()
    numerical_columns = ["Rating", "Rating Count", "Minimum Installs", "Maximum Installs", "Price"]

    for col in numerical_columns:
        # calculate interquartile range (IQR)
        q1 = df_num.approxQuantile(col, [0.25], 0.0)[0]
        q3 = df_num.approxQuantile(col, [0.75], 0.0)[0]
        iqr = q3 - q1

        # calculate the lower and upper bound
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        # get the number of outliers
        outliers = df_num.filter((df_num[col] < lower_bound) | (df_num[col] > upper_bound))
        num_outliers = outliers.count()

        # calculate the percentage of outliers
        percent_outliers = (num_outliers / total_rows) * 100

        print(f'Number of outliers in {col}: {num_outliers} ({percent_outliers:.2f}%)')

        # create a new column to flag the outliers
        is_outlier_col = f"is_outlier_{col}"
        df_num = df_num.withColumn(is_outlier_col, when((df_num[col] < lower_bound) | (df_num[col] > upper_bound), 1).otherwise(0))

    # Select the outlier columns
    selected_columns = [column for column in df_num.columns if column.startswith("is_outlier")]

    # Add up the outlier columns to create a new column for total outliers
    new_df = df_num.withColumn("total_outliers", sum(df_num[col] for col in selected_columns))

    # Filter out rows with more than 1 total outlier
    new_df = new_df.filter(new_df["total_outliers"] <= 1)

    print(f'Number of rows before removing outliers: {total_rows}')
    print(f'Number of rows after removing those having more than 1 outlier in its columns: {new_df.count()}')

    # Drop the extra columns created above
    new_df = new_df.drop(*selected_columns).drop("total_outliers")

    return new_df


def boxplot_for_outliers(df, new_df):
    # Select only the numerical columns
    df_num = df.select(["Rating", "Rating Count", "Minimum Installs", "Maximum Installs", "Price"])
    new_df_num = new_df.select(["Rating", "Rating Count", "Minimum Installs", "Maximum Installs", "Price"])

    # Create a list of the numerical columns
    num_cols = df_num.columns

   # create a grid of subplots 
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 10))
    plt.style.use("dark_background")

    # plot each column in a boxplot for both the original and the new DataFrame
    for i, col_name in enumerate(num_cols):
        # make the boxplot vertically stacked

        sns.boxplot(y=col_name, data=df_num.toPandas(),  ax=axes[0,i])
        sns.boxplot(y=col_name,data=new_df_num.toPandas(), ax=axes[1,i])

        # give each subplot a title
        axes[0,i].set_title(f'Original {col_name}')
        axes[1,i].set_title(f'New {col_name}')

    plt.tight_layout()
    plt.show()


# def remove_outliers(original_df,df_with_no_outliers):
#     '''
#     Remove the outliers from the dataset
#    '''
#     common_cols = list(set(original_df.columns) & set(df_with_no_outliers.columns))

#     new_df = df_with_no_outliers.join(original_df, on=common_cols, how='inner')    

#     return new_df


#---------------------------------------  Missing Values --------------------------------------- 

def show_nulls(df):
    '''
    Show the number of null values in each column
    '''

    df_miss = df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]).toPandas()
    df_miss = df_miss.transpose()
    df_miss.columns = ['count']
    df_miss = df_miss.sort_values(by='count', ascending=False)

    # get the percentage of null values in each column
    df_miss['percentage'] = df_miss['count']/df.count()*100

    print(df_miss)


def handle_missing_values(df, handling_method='drop', cols=[]):
    '''
    Handling the missing values in the dataset
        '''
    
    print(f'Total Number of rows : {df.count()}')
    
    if handling_method=='drop':
        if len(cols)>0:
            df = df.na.drop(subset=cols)
        
        print(f'Number of rows after dropping nulls: {df.count()}') 
        return df
    
    if handling_method=='mode':
        for col_name in cols:
            mode_value = df.select(mode(col_name)).collect()[0][0]
            df = df.fillna(mode_value, subset=[col_name])
        return df

    imputer = Imputer(inputCols=cols, outputCols=["{}_imputed".format(c) for c in cols])

    if handling_method=='mean':
        imputer.setStrategy("mean")

    elif handling_method=='median':
        imputer.setStrategy("median")
    

    df = imputer.fit(df).transform(df)

    # replace the value of the original columns with the imputed columns
    for col_name in cols:
        df = df.withColumn(col_name, when(col(col_name).isNull(), col(col_name + '_imputed')).otherwise(col(col_name)))
        df = df.drop(col_name + '_imputed')
        
    return df


# ======================================= Explore some columns =======================================
#---------------------------------------  Currency  ---------------------------------------


def currency_col(df):
    '''
    Explore the currency column
    '''
    if 'Currency' not in df.columns:
        return
    
    currency_col = df.select('Currency')
    currency_counts = currency_col.groupBy('Currency').count().sort('count', ascending=False)
    currency_counts.show()


# ---------------------------------------  Size  ---------------------------------------

def check_values_in_size_col(df):
    '''
    We have a value in the size column called 'Varies with device', we need to check how many apps have this value
    '''
    df_size= df.filter(df.Size == 'Varies with device').count()
    print(f'Percentage of apps with size "Varies with device": {(df_size/df.count())*100} %')


def convert_size_to_bytes(df):
    '''
    Since size is in G/M/K, we can convert it to be totally numerical for ease of analysis
    '''

    df = df.filter(df.Size != 'Varies with device')

    # remove the 'G', 'M' and 'k' from the values
    df = df.withColumn('Size', regexp_replace('Size', 'G', '000000000'))
    df = df.withColumn('Size', regexp_replace('Size', 'M', '000000'))
    df = df.withColumn('Size', regexp_replace('Size', 'k', '000'))   

    print("Converted all sizes to Bytes.")

    return df


# ---------------------------------------  Scraped time  ---------------------------------------

def check_scraped_time(df):
    '''
    Check the unique dates in the scrapped time column, so that if most of them have been scrapped in the same day,
    we can drop the scrapped time column
    '''

    scraped_time_col = df.select('Scraped Time')

    # 2021-06-16 01:37:34 --> split by space --> 2021-06-16    
    scraped_time_col = scraped_time_col.withColumn('Scraped Time', split(col('Scraped Time'), ' ').getItem(0))

    # get the unique dates
    scraped_time_col = scraped_time_col.groupBy('Scraped Time').count().sort('count', ascending=False)

    scraped_time_col.show()


#======================================== RDD ========================================
          
def remove_commas(df):
    '''
    Remove commas from a column to make only commas be for separating columns.
    Mainly for RDD purposes.

    '''
    for col in df.columns:  
        col_type= df[col].dtypes
        if col_type=='bool':
            continue

        if col_type!='object':
            df[col] = df[col].astype(str)

        if col=='Installs':
            df[col] = df[col].str.replace(',', '')
        else:
            
            df[col] = df[col].str.replace(',', ' ')

        df[col]= df[col].astype(col_type)

    return df


def delimiter_to_comma(file_name='Google-Playstore',raw=False):
    '''
    Handle the delimiter in the dataset to be used in RDD

    raw: if True, use the raw data, else use the processed data
    '''
    if raw: 
        df= pd.read_csv('../Dataset/'+file_name+'.csv',index_col=False,)

    df_new= remove_commas(df)
    df_new.to_csv('../Dataset/'+file_name+'-RDD'+'.csv', index=False)


#======================================== Main Function ========================================

def process_data(spark, file_name='all', features='all', encode=False, useless_cols=[], rdd=False):
    '''
    To be used in the next modules
    '''

    df= read_data(spark, file_name=file_name, features=features, encode=encode, useless_cols=useless_cols)

    print('Detecting outliers...')
    df= detect_outliers(df)

    print("Removing useless columns...")
    drop_cols= ['Developer Website','Privacy Policy','Currency','Scraped time'] 
    df= remove_useless_col(df,drop_cols)

    print("Handling missing values...")
    uninteresting_cols= ['Minimum Android','Size','Minimum Installs','Installs','Developer Email',\
                   'Developer Id','Price','Ad Supported','In App Purchases']
    df=handle_missing_values(df,cols=uninteresting_cols)

    interesting_num_cols=['Rating','Rating Count','Maximum Installs']
    df= handle_missing_values(df, handling_method='mean', cols=interesting_num_cols)

    interesting_cat_cols=['Released']
    df= handle_missing_values(df, handling_method='mode', cols=interesting_cat_cols) 

    print("Converting size to bytes...")
    df= convert_size_to_bytes(df)
    
    if rdd:
        df= df.toPandas()
        delimiter_to_comma(file_name=file_name,raw=False)

    return df