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

The results are shown below after cleaning. Xgboost gave the best precision for our minority class.
   

Naive Bayes
               precision    recall  f1-score   support

Less than 50k       0.99      0.79      0.88     93576
More than 50k       0.21      0.82      0.33      6186

  avg / total       0.94      0.79      0.84     99762


SVM

               precision    recall  f1-score   support

Less than 50k       0.95      0.96      0.95     93576
More than 50k       0.28      0.25      0.26      6186

  avg / total       0.91      0.91      0.91     99762


SVM using class_weight = Balanced

               precision    recall  f1-score   support

Less than 50k       0.98      0.77      0.86     93576
More than 50k       0.18      0.73      0.28      6186

  avg / total       0.93      0.77      0.83     99762



Smote on SVM ratio 0.4
               precision    recall  f1-score   support

Less than 50k       0.98      0.82      0.89     93576
More than 50k       0.21      0.71      0.33      6186

  avg / total       0.93      0.82      0.86     99762



Random Forest
               precision    recall  f1-score   support

Less than 50k       0.96      0.99      0.97     93576
More than 50k       0.65      0.38      0.48      6186

  avg / total       0.94      0.95      0.94     99762



Decision Tree
               precision    recall  f1-score   support

Less than 50k       0.96      0.96      0.96     93576
More than 50k       0.43      0.46      0.44      6186

  avg / total       0.93      0.93      0.93     99762



XGB Classifier
               precision    recall  f1-score   support

Less than 50k       0.96      0.99      0.98     93576
More than 50k       0.75      0.35      0.47      6186

  avg / total       0.95      0.95      0.94     99762



XGB Classifier Smote 0.4
               precision    recall  f1-score   support

Less than 50k       0.97      0.97      0.97     93576
More than 50k       0.56      0.54      0.55      6186

  avg / total       0.94      0.95      0.94     99762



XGB Classifier Smote 00.1
               precision    recall  f1-score   support

Less than 50k       0.96      0.99      0.98     93576
More than 50k       0.72      0.38      0.50      6186

  avg / total       0.95      0.95      0.95     99762

XGB Classifier Smote 00.07
               precision    recall  f1-score   support

Less than 50k       0.96      0.99      0.98     93576
More than 50k       0.74      0.35      0.48      6186

  avg / total       0.95      0.95      0.94     99762


