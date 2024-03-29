import os
import sys; sys.path.append("../")
import pandas as pd
import pyspark
from pyspark.ml.feature import Imputer, StringIndexer,IndexToString
from pyspark.sql.functions import regexp_replace, isnan, when, count, col,mode, split
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql.types import BooleanType
from pyspark.sql import SparkSession
import os

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


def read_data(spark, file_name=None, features='all', useless_cols=[], cols_to_encode=[], absolute_csv_path=False):
    
    '''
    Read the dataset and return a dataframe 

    features: 'all', 'Categorical', 'Numerical' ----> all is the default     
    useless_cols: list of columns to be removed from the dataset (default: empty list)
    cols_to_encode: list of Categorical columns to be encoded (default: empty list)

    '''
    if(absolute_csv_path):
        df = spark.read.csv(file_name, header=True, inferSchema= True)
    else:
        dir= os.path.dirname(os.path.realpath(__file__))
        if file_name!=None: path= os.path.join(dir, '../Dataset/'+file_name+'.csv')
        else:               path= os.path.join(dir, '../Dataset/Preprocessed_data.csv')

        df = spark.read.csv(path, header=True, inferSchema= True)
    
    # cast the numerical columns to their correct type
    df= df.withColumn("Rating", df["Rating"].cast("float"))
    df= df.withColumn("Rating Count", df["Rating Count"].cast("int"))
    df= df.withColumn("Minimum Installs", df["Minimum Installs"].cast("int"))
    df= df.withColumn("Maximum Installs", df["Maximum Installs"].cast("int"))
    df= df.withColumn("Price", df["Price"].cast("float"))

    # remove the comma and the plus sign from the Installs column and cast it to int
    df = df.withColumn("Installs", when(col("Installs").contains(","), regexp_replace("Installs", ",", ""))\
        .otherwise(col("Installs")))\
        .withColumn("Installs", regexp_replace("Installs", "\\+", ""))\
        .withColumn("Installs", col("Installs").cast("int"))

    # cast boolean columns to string (for encoding purposes)
    df= df.withColumn("Editors Choice", df["Editors Choice"].cast("string"))

    # Numerical columns
    numerical_cols = [column for column, dtype in df.dtypes if dtype != 'string']

    # extract the categorical fetaures only 
    if features=='Categorical':
        df= df.select([column for column in df.columns if column not in numerical_cols])
        
    # extract the numerical fetaures only
    elif features=='Numerical':
      df= df.select(numerical_cols)  

    # encode the categorical features 
    if len(cols_to_encode)>0:
        df= encode_categ_features(df,cols_to_encode)
        
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

def binarize_col(df, cols=[]):
    '''
    takes columns and turns them to binary, missing values are put as False, filled values are put as True
    '''
    for c in cols:
        df = df.withColumn("new_column", when(col(c).isNull(), False).otherwise(True))
        df = df.drop(c)
        df = df.withColumnRenamed("new_column", c)
         
    return df

def convert_to_bool(df, cols=[]):
    '''
    takes boolean columns but in string format and converts them to true python boolean
    '''
    df = convert_binary_pyspark(df, cols=cols)
    for c in cols:
        df = df.withColumn(c, df[c].cast(BooleanType))
    return df


# function to convert binary columns to numeric with 0 and 1
def convert_binary_pyspark(df, cols):    
    binary_cols = cols
    for c in binary_cols:
        df = df.withColumn("new_column", when(col(c)=='False', False).otherwise(True))
        df = df.drop(c)
        df = df.withColumnRenamed("new_column", c)
        #df = df.withColumn(c, df[c].cast(BooleanType))
    return df




def encode_categ_features(df, cols_to_encode=[]):
    '''
    Encode the categorical features to numerical features
    '''
        
    for column in cols_to_encode: 
        encoder= StringIndexer(inputCol=column, outputCol=column+"_index", handleInvalid='skip').fit(df)
        df= encoder.transform(df)
        df= remove_useless_col(df,[column])
        df= df.withColumnRenamed(column+"_index", column)

    return df

def decode_num_features(df,cols_to_decode=[]):
    '''
    Decode the encoded numerical features back to categorical features
    '''

    for column in cols_to_decode:
        decoder= IndexToString(inputCol=column, outputCol=column+"_index")
        df = decoder.transform(df)
        df=remove_useless_col(df,[column])
        df= df.withColumnRenamed(column+"_index", column)

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
    numerical_columns = [column for column, dtype in df.dtypes if dtype != 'string']

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
    df_num = df.select([column for column, dtype in df.dtypes if dtype != 'string'])
    new_df_num = new_df.select([column for column, dtype in new_df.dtypes if dtype != 'string'])

    # Create a list of the numerical columns
    num_cols = df_num.columns

    # Create a figure with number of rows equal to number of columns in the dataframe and 2 columns 
    fig, ax = plt.subplots(nrows=len(num_cols), ncols=2, figsize=(15, 50))

    # Loop over the columns and create a boxplot for each one, where the first column is the original dataframe and the second column is the dataframe without outliers
    for i, column in enumerate(num_cols):
       # Convert the PySpark column to a Pandas Series
        df_series = df_num.select(column).toPandas()[column]
        new_df_series = new_df_num.select(column).toPandas()[column]

        # Plot the boxplot using the Pandas Series
        sns.boxplot(data=df_series, ax=ax[i, 0])
        sns.boxplot(data=new_df_series, ax=ax[i, 1])
        
        ax[i, 0].set_title(f'Original {column}')
        ax[i, 1].set_title(f'Without outliers {column}')

    plt.show()

# function to convert binary columns to numeric with 0 and 1
def convert_binary(df, cols= None):    
    df.dropna(inplace=True)
    if(cols):
        binary_cols = cols
    else:
        binary_cols = ['Ad Supported', 'In App Purchases', 'Free', 'Editors Choice']
    for col in binary_cols:
        # check if the column is exist in the data frame
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('True', '1')
            df[col] = df[col].astype(str).str.replace('False', '0')
            # drop any value that is not 0 or 1
            df = df[df[col].isin(['0', '1'])]
    return df

def convert_to_numeric(df):
    ''' function to clean numeric columns from any strings and convert them to numeric'''
    df = df.sample(frac=0.15)
    df.dropna(inplace=True)
    # remove comma from all the columns
    df['Installs'] = df['Installs'].str.replace(',', '').str.replace('+', '').astype(float) 
    df['Rating'] = df['Rating'].astype(float)
    df['Size'] = df['Size'].str.replace(',', '').astype(float) /1000000
    df['Minimum Android'] = df['Minimum Android'].str.replace(',', '').str.replace('Varies with device', '0.0')
    # remove  "and up" from  Minimum Android column
    df['Minimum Android'] = df['Minimum Android'].str.replace(' and up', '')
    return df

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

    # convert the column to float
    df = df.withColumn('Size', df['Size'].cast("float")) 

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

def convert_Last_Updated_to_Year(df):
    '''
    converts the column "Last Updated" to year only (reduction of dimensionality)
    '''
    #Last_Updated_col = df.select('Last Updated')

    # May 21, 2020 --> split by space --> May-21,-2020    
    df = df.withColumn('Last Updated', split(col('Last Updated'), ' ').getItem(2).cast('int'))
    df=handle_missing_values(df,cols=['Last Updated'])
    # get the unique dates
    another_df = df.groupBy('Last Updated').count().sort('count', ascending=False)

    another_df.show()
    
    return df


#======================================== RDD ========================================
          
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


def delimiter_to_comma(file_name='Preprocessed_data',raw=False):
    '''
    Handle the delimiter in the dataset to be used in RDD

    raw: if True, use the raw data, else use the processed data
    '''
    if raw: 
        df= pd.read_csv('../Dataset/'+"Google-Playstore"+'.csv',index_col=False, on_bad_lines='skip')
    else: 
        df= pd.read_csv('../Dataset/'+file_name+'.csv',index_col=False, on_bad_lines='skip')

    df_new= remove_commas(df)
    df_new.to_csv('../Dataset/'+file_name+'_RDD'+'.csv', index=False)


#======================================== Main Function ========================================

def process_data(spark, file_name='all', features='all', cols_to_encode=[], useless_cols=[], rdd=False):
    '''
    To be used in the next modules
    '''

    df= read_data(spark, file_name=file_name, features=features,cols_to_encode=cols_to_encode, useless_cols=useless_cols)

    print("Removing useless columns...")
    drop_cols= ['Developer Website','Privacy Policy','Currency','Scraped time'] 
    df= remove_useless_col(df,drop_cols)

    print("Converting size to bytes...")
    df= convert_size_to_bytes(df)

    print('Detecting outliers...')
    df= detect_outliers(df)

    print("Handling missing values...")
    uninteresting_cols= ['Minimum Android','Size','Minimum Installs','Installs','Developer Email',\
                   'Developer Id','Price','Ad Supported','In App Purchases']
    df=handle_missing_values(df,cols=uninteresting_cols)

    interesting_num_cols=['Rating','Rating Count','Maximum Installs']
    df= handle_missing_values(df, handling_method='mean', cols=interesting_num_cols)

    interesting_cat_cols=['Released']
    df= handle_missing_values(df, handling_method='mode', cols=interesting_cat_cols) 
    
    if rdd:
        df= df.toPandas()
        delimiter_to_comma(file_name=file_name,raw=False)

    return df