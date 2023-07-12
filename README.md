# XL-Data
## Analysis of Google's Play Store applications using Pyspark and Pyspark ML

It is our Big-Data course project, using MR techniques.
Aiming to collect conclusions and drive a managerial decision based on analyzing numerous software applications deployed on Google's Play Store.
Our preference framework was Spark, hence its Python implementation, Pyspark.
Our data entered a very long pipeline:- cleaning, preprocessing, transformations, EDA, a lot of Map-Reduces, heavy clustering, AI modeling and finally Decision Making.
The dataset didn’t meet up to our expectations in various ways, but we worked our way around those stumps.

![pysparkImage](https://the-examples-book.com/data-engineering/intro-to-data-engineering/_images/pyspark.png)

## What are we doing? 👻
### We are helping a company develop a new Profitable app. 
#### We want to choose its price that maximizes the company's profits, choose the best suitable developers, its stance regarding ads, and what exact category should we make the app for.

<h3>
<strong>Problem Definition 🤔 </strong>
</h3>
If a company wants to develop a new app, What’s the best way to develop it to keep it highly profitable and highly rated?
In addition to predicting the best price for this app -if it’s paid- and predicting the number of installations for this app based on its given features.
Lastly, if this company wants to hire new mobile app developers, we can help it to know those whose apps have the highest ratings and number of installations.

### Dataset Source 👓
This dataset was scraped via a python script running on a cloud. (we didn’t scrape it, rather, we downloaded it from [here](https://www.kaggle.com/datasets/gauthamp10/google-playstore-apps).

### Pipeline 📈
1. Data Preprocessing and cleansing
2. Data Exploration (Involves visualization to extract knowledge from the data):
3. Descriptive analysis: using Map Reduce.
4. Diagnostic analysis: Using Pearson and Spearman’s correlation.
5. Clustering to gain insights about data: Using K-means, K-Medoids or ISODATA.
6. Model training and validation For Prediction and Classification: Using SVM, LR or Decision Trees, plus K-Fold.


### Let's Skip to the final results :happy:

# Results 👀

## As a manager, You should:
### Choose a Category from this list:
* Art & Design
* Games
* Role-Playing
* Photography
* Comics

### It is better to launch the app as Free, then make it paid after a year.

### Hire a Development group from this list:
* PT. Teknologi Usaha Sukses Bersama
* Petar Marković
* Rmapps
* GameWriterStudio
* 인디사이드게임즈
* Ads are optional, but we prefer not to support ads…
* If the app is paid, it is better to keep the price under 4$

## This predicts:
* Avg number of installs = 27k
* Avg Rating = 3.4		(assuming having more than 2000 critic)
* App’s price = 3.2$  	(if it was free at launch)

*With a confidence level of 99.999999% you will be a millionaire in just 3 hours 🐸*



### now with the boring details  :trollface:
### which you can also find in our [document](https://docs.google.com/document/d/1Ae7I7DVF83mGm-TuCs52gsKoteDXYJLdfNRjGs9GAz8/edit?usp=sharing) and our [presentation](https://docs.google.com/presentation/d/1uZg0dMd-H88134-BuGWc8wJUGf3oopY9ew_qKmsZ3ck/edit?usp=sharing):

### In this project, we did:
* Collect the Dataset
* Install Pyspark and all its dependencies
* Preprocessing and Cleaning our dataset
* Perform EDA using Pyspark's low level Map-Reduce functions
* Use RDDs whenever possible
* Perform Diagnostic analysis given the previous EDA
* Answer some predictive questions
* Clustering
* ML Modelling
* Business intelligence
