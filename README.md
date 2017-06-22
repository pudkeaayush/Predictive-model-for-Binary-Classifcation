# Predictive-model-for-Binary-Classifcation
Build a predictive model to classify the income of people above and below 50k per year. The original reference of this python code was the following blog post:
https://www.analyticsvidhya.com/blog/2016/09/this-machine-learning-project-on-imbalanced-data-can-add-value-to-your-resume/

We just have a single code.py file to run( python3 code.py will suffice). 
Need to install xgboost library to run the code( and other libraries like sklearn, nltk and so on most commonly used in python today). 
Also currently the code contains the local path of train and test data.
Please help to change the below code from code.py to the location of the file you have saved the extracted test and train data to.

<br />
#Open test data <br />
test = open(r'/home/devesh/Downloads/test.csv')<br />
testData = csv.reader(test)<br />
<br />
#Open training data<br />
training = open(r'/home/devesh/Downloads/train.csv')<br />

The code is sequential in the sense that first we are analyzing the data used in this experiment, then are cleaning the data, followed by 
data manipulation to change the values of features which are less into 'Others' followed by prediction using the different classifiers.

Our project revolved on binary classification of US Census data to classify the income of people
below and above 50k per year. The main objective behind taking this project was that it contains 
imbalanced data( 94% of data is majority class( <50k income per year ) whereas only 6% is minority
class class( >50k)). We tried to increase the precision of minority class from ( 21%) achieved via 
Naive Bayes to around 75%( achieved via xgboost ). In our code we have followed techniques of first 
analyzing the data by drawing various plots and then cleaning the data. In our case we removed 4 features
as they were containing around 50 percent missing data. The fields removed were:
a) migration_msa 
b ) migration_reg 
c ) migration_within_reg
d ) migration_sunbelt

Xgboost gave the best precision for our minority class.

