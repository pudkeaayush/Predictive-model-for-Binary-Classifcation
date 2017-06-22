# Predictive-model-for-Binary-Classifcation
Build a predictive model to classify the income of people above and below 50k per year.

We just have a single code.py file to run. Need to install xgboost library to run the code( and other libraries
like sklearn, nltk and so on most commonly used in python today). Also currently the code contains the local path of data train and test.
The code is sequential in the sense that first we are analyzing the data used in this experiment, then are cleaning the data, followed by 
data manipulation to change the values of features which are less into 'Others' followed by prediction using the different classifiers.

Our project revolved on binary classification of US Census data to classify the income of people
below and above 50k per year. The main objective behind taking this project was that it contains 
imbalanced data( 94% of data is majority class( <50k income per year ) whereas only 6% is minority
class class( >50k)). We tried to increase the precision of minority class from ( 21%) achieved via 
Naive Bayes to around 75%( achieved via xgboost ). In our code we have followed techniques of first 
anaylyzing the data by drawing various plots and then cleaning the data. In our case we removed 4 features
as they were containing around 50 percent missing data. The fields removed were:
a) migration_msa 
b ) migration_reg 
c ) migration_within_reg
d ) migration_sunbelt

Xgboost gave the best precision for our minority class.

