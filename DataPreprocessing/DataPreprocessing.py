import os
import sys; sys.path.append("../")
from sklearn.model_selection import train_test_split
import pandas as pd
import pyspark
from pyspark.ml.feature import Imputer, StringIndexer
from pyspark.sql.functions import regexp_replace, isnan, when, count, col,row_number, monotonically_increasing_id
from pyspark.sql.window import Window

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
    if encode :
        cols_to_drop=[]
        for col in df.columns:
            if col not in numerical_cols:
                indexer = StringIndexer(inputCol=col, outputCol=col+"_index")
                df = indexer.setHandleInvalid("keep").fit(df).transform(df) 
                cols_to_drop.append(col)

        df = df.drop(*cols_to_drop)

    # remove the useless columns
    if len(useless_cols) > 0:
        df = remove_useless_col(df, useless_cols)

    return df


def get_info(df):
    '''
    Get the info of the dataset
    '''
    print(f'Number of rows: {df.count()}, Number of columns: {len(df.columns)}')
    # print(f'Available features: {df.columns}')
    
    df.describe().show()


def remove_useless_col(df, cols=[]):
    '''
    Remove the useless columns from the dataset
    '''
    df = df.drop(*cols) 
    return df


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


def handle_missing_values(df, treatment='drop', cols=[]):
    '''
    Handling the missing values in the dataset
    '''
    
    print(f'Total Number of rows : {df.count()}')
    
    # get #rows with missing values in sny of its columns 
    #TODO: fix this
    # print(f'Number of rows with missing values: {df_miss.count()}')
    
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



def detect_outliers(df):
    '''
    Detect outliers in all numerical columns
    Detect rows with outliers in any of its columns and if it has more than 1 outlier in any of its columns
    then it will be considered as an outlier row and can be removed 
    '''
    # Select only the numerical columns
    df_num = df.select(["Rating", "Rating Count", "Minimum Installs", "Maximum Installs", "Price"])

    # get the total number of rows in the DataFrame
    total_rows = df_num.count()

    for col in df_num.columns:
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
    new_df_with_no_outliers = new_df.filter(new_df["total_outliers"] <= 1)

    print(f'Number of rows before removing outliers: {total_rows}')
    print(f'Number of rows after removing outliers: {new_df_with_no_outliers.count()}')

    # Drop the extra columns created above
    new_df_with_no_outliers = new_df_with_no_outliers.drop(*selected_columns).drop("total_outliers")

    return new_df_with_no_outliers



def remove_outliers(df_with_no_outliers):
    '''
    Remove the outliers from rows
    '''
    return df_with_no_outliers


def handle_size_col(df):
    '''
    Since size is in G/M/K, we can convert it to be totally numerical for ease of analysis
    '''

    df_size= df.filter(df.Size == 'Varies with device').count()
    print(f'Percentage of apps with size "Varies with device": {(df_size/df.count())*100} %')

    # remove the 'Varies with device' value
    df = df.filter(df.Size != 'Varies with device')

    # remove the 'G', 'M' and 'k' from the values
    df = df.withColumn('Size', regexp_replace('Size', 'G', '000000000'))
    df = df.withColumn('Size', regexp_replace('Size', 'M', '000000'))
    df = df.withColumn('Size', regexp_replace('Size', 'k', '000'))   

    print("Converted all sizes to Bytes.")

def currency_col(df):
    '''
    Explore the currency column
    '''
    currency_col = df.select('Currency')
    currency_counts = currency_col.groupBy('Currency').count().sort('count', ascending=False)
    currency_counts.show()


#--------------------------------------------------------------------------
          
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

        # if col=='Minimum Installs':
        #     df[col]= df[col].astype('Int64')

    return df


def delimiter_to_comma(file_name='Google-Playstore'):
    '''
    Handle the delimiter in the dataset to be used in RDD
    '''
    df= pd.read_csv('../Dataset/'+file_name+'.csv',index_col=False,)
    df_new= remove_commas(df)
    df_new.to_csv('../Dataset/'+file_name+'-RDD'+'.csv', index=False)