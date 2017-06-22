import csv
from collections import Counter
import operator
from scipy.stats import gaussian_kde
import numpy as np
from collections import defaultdict
from textwrap import wrap
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn import svm
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pylab

def Change_Data(train,test,category):
    List=[]

    for i in range (1,len(train)):
        List.append(train[i][category])
    List=list(set(List))
    
    
    for i in range(1,len(train)):
        for j in range(0,len(List)):
            if(train[i][category]==List[j]):
                train[i][category]=j
    
    for i in range(1,len(test)):
        for j in range(0,len(List)):
            if(test[i][category]==List[j]):
                test[i][category]=j


        
    for i in range(0,len(List)):
        x=0
        for j in range(1,len(train)):
            if(train[j][category]==i):
                x=x+1
        #print("List ",List[i],"Count " , x )
        if((((x*100)/len(train)))<5):
            List[i]=-1
   

    for i in range(1,len(train)):
        if(List[train[i][category]]==-1):
            train[i][category]=-1
    for i in range(1,len(test)):
        if(List[test[i][category]]==-1):
            test[i][category]=-1


def func_label(data):
    ret_list = []
    for i in data:
        if( i == 0):
            ret_list.append('red')
        else:
            ret_list.append('green')
    print(ret_list)
    return ret_list
    

#Open test data
test = open(r'/home/devesh/Downloads/test.csv')
testData = csv.reader(test)

#Open training data
training = open(r'/home/devesh/Downloads/train.csv')
trainingData = csv.reader(training)

#Make a list of Training and test data
listofTrainingData = list(trainingData)
listofTestData = list(testData)

#Extract Training data row and column length
row_length_train = len(listofTrainingData)
col_length_train = len(listofTrainingData[0])

row_length_test = len(listofTestData)
col_length_test = len(listofTestData[0])

#Extract column titles
column_titles = listofTrainingData[0]
#print(column_titles)

#get income_level column index
income_level_index = column_titles.index("income_level")
#print(income_level_index)

#get unique values of income_level


for i in range(len(listofTestData)):
    for j in range(len(listofTestData[0])):
        listofTestData[i][j]=listofTestData[i][j].strip()
        if(listofTestData[i][j]=='?'):
            listofTestData[i][j]='NA'

#Change value of -50000 to 0 and that of others to 1 for training data. Skip for first row
count = 0
for i in range(len(listofTrainingData)):
    if( count == 0):
        count = count + 1
        continue
    if(listofTrainingData[i][income_level_index].strip() == str(-50000)):
        listofTrainingData[i][income_level_index] = 0
    else:
        listofTrainingData[i][income_level_index] = 1
    count = count + 1
        
#Change value of -50000 to 0 and that of others to 1 for test data. Skip for 1st row
count = 0
for i in range(len(listofTestData)):
    if( count == 0):
        count = count + 1
        continue
    if(listofTestData[i][income_level_index].strip() == str(-50000)):
        listofTestData[i][income_level_index] = 0
    else:
        listofTestData[i][income_level_index] = 1
    count = count + 1


zerooneCounter = Counter(x[income_level_index] for x in listofTrainingData)
zero_count = round((zerooneCounter[0]*100) / (row_length_train - 1))
one_count = round((zerooneCounter[1]*100) / (row_length_train - 1))


print(" ")
print("Data Exploration")
print(" ")


income_level_data_train = map(operator.itemgetter(40),listofTrainingData)
income_level_data_train = [int(x) for x in income_level_data_train if x != str('income_level')]
income_level_data_test = map(operator.itemgetter(40), listofTestData)
income_level_data_test = [int(x) for x in income_level_data_test if x != str('income_level')] 




 #Plot age density curve
age_index = column_titles.index("age")
data = [int(x[age_index]) for x in listofTrainingData if x[age_index] != str('age')]
#print(data)
density = gaussian_kde(data)
xs = np.linspace(0,90,25)
density.covariance_factor = lambda : .020
density._compute_covariance()
plt.plot(xs,density(xs))
plt.xlabel('Age')
plt.ylabel('Density')
plt.title("Density curve for Age")
plt.show()
 
 #Plot captial_losses density curve
capital_losses_index = column_titles.index("capital_losses")
data = [int(x[capital_losses_index]) for x in listofTrainingData if x[capital_losses_index] != str('capital_losses')]
#print(data)
density = gaussian_kde(data)
xs = np.linspace(0,4000,1000)
density.covariance_factor = lambda : .020
density._compute_covariance()
plt.plot(xs,density(xs))
plt.xlabel('Capital losses')
plt.ylabel('Density')
plt.title("Density curve for Capital Losses")
plt.show()



 #Plot categorical variable class_of_worker




#Plot categorical variable education
education_index = column_titles.index("education")
data_education = [x[education_index] for x in listofTrainingData if  x[education_index] != str('education')]
data_education_0 = [data_education[i] for i in range(len(data_education)) if income_level_data_train[i] == 0]
data_education_1 = [data_education[i] for i in range(len(data_education)) if income_level_data_train[i] == 1]
set_of_data_education = list(set(data_education))
education0_Counter = Counter(x for x in data_education_0)
education1_Counter = Counter(x for x in data_education_1)

 
education_data_dict = defaultdict(list)
 
for d in (education0_Counter, education1_Counter): # you can list as many input dicts as you want here
    for key, value in d.items():
        education_data_dict[key].append(value)

keys_dict = education_data_dict.keys()
values = list(education_data_dict.values())

vals_0 = [x[0] for x in values]
vals_1 = [x[1] if len(x) > 1 else 0 for x in values ]
fig, ax = plt.subplots()
width = 0.35
ind = np.arange(len(set_of_data_education))
plt.bar(range(len(education0_Counter)), vals_0, width , color='red')
plt.bar(ind + width,vals_1, width , color='green')
plt.xticks(range(len(education0_Counter)), keys_dict, rotation='vertical')
plt.xlabel('Education')
plt.ylabel('Count')
plt.title("Education Data Count")
plt.show()
print(" ")


for i in range (1,len(listofTrainingData)):
   listofTrainingData[i][0]= int(listofTrainingData[i][0])
   listofTrainingData[i][5]= int(listofTrainingData[i][5])
   listofTrainingData[i][16]= int(listofTrainingData[i][16])
   listofTrainingData[i][17]= int(listofTrainingData[i][17])
   listofTrainingData[i][18]= int(listofTrainingData[i][18])
   listofTrainingData[i][29]= int(listofTrainingData[i][29])
   listofTrainingData[i][38]= int(listofTrainingData[i][38])
for i in range (1,len(listofTestData)):   
   listofTestData[i][0]= int(listofTestData[i][0])
   listofTestData[i][5]= int(listofTestData[i][5])
   listofTestData[i][16]= int(listofTestData[i][16])
   listofTestData[i][17]= int(listofTestData[i][17])
   listofTestData[i][18]= int(listofTestData[i][18])
   listofTestData[i][29]= int(listofTestData[i][29])
   listofTestData[i][38]= int(listofTestData[i][38])







Nominal_List=[1,2,3,4,6,7,8,9,10,11,12,13,14,15,19,20,21,22,23,27,30,31,32,33,34,35,36,37,39,40]
print(" ")
print("Data Manipulation")
print(" ")
for i in range(0,len(Nominal_List)):
    p=0
    p=Nominal_List[i]
    Change_Data(listofTrainingData,listofTestData,p)





#Plot categorical variable education
education_index = column_titles.index("education")
data_education = [x[education_index] for x in listofTrainingData if  x[education_index] != str('education')]
data_education_0 = [data_education[i] for i in range(len(data_education)) if income_level_data_train[i] == 0]
data_education_1 = [data_education[i] for i in range(len(data_education)) if income_level_data_train[i] == 1]
set_of_data_education = list(set(data_education))
education0_Counter = Counter(x for x in data_education_0)
education1_Counter = Counter(x for x in data_education_1)

 
education_data_dict = defaultdict(list)
 
for d in (education0_Counter, education1_Counter): # you can list as many input dicts as you want here
    for key, value in d.items():
        education_data_dict[key].append(value)

keys_dict = education_data_dict.keys()
values = list(education_data_dict.values())

vals_0 = [x[0] for x in values]
vals_1 = [x[1] if len(x) > 1 else 0 for x in values ]
fig, ax = plt.subplots()
width = 0.35
ind = np.arange(len(set_of_data_education))
plt.bar(range(len(education0_Counter)), vals_0, width , color='red')
plt.bar(ind + width,vals_1, width , color='green')
plt.xticks(range(len(education0_Counter)), keys_dict, rotation='vertical')
plt.xlabel('Education')
plt.ylabel('Count')
plt.title("Education Data Count After Merging")
plt.show()
print(" ")





print("Length of Training Data before Cleaning",len(listofTrainingData[0]))

print (" ")
print ("====Data Cleaning===")
#Data Cleaning
#find missing values per column for nominal data and remove ones that have more than 5% missing data
flag =[ False for x in range(len(column_titles))]
#print (flag)

for i in range(len(column_titles)):
    index = column_titles.index(column_titles[i])
    #print("Index : ",index)
    data = [x[index] for x in listofTrainingData if  x[index] != str(column_titles[i])]
    if ( ((data.count('NA') *100) / len(listofTrainingData)) > 5):
        print(column_titles[i] , " : " , (data.count('NA') *100) / len(listofTrainingData))
        flag[i] = True
      
print(flag.count(True))
list_of_lists = [list(elem) for elem in listofTrainingData]
for x in list_of_lists:
    for i in range(len(column_titles)-1,-1,-1):
        if( flag[i] == True):
            del x[i]

listofTrainingData = list_of_lists
column_titles = [column_titles[i] for i in range(len(column_titles)) if flag[i] == False  ]
del list_of_lists


print("")

list_of_lists = [list(elem) for elem in listofTestData]

for x in list_of_lists:
    for i in range(len(column_titles)-1,-1,-1):
        if( flag[i] == True):
            del x[i]

listofTestData = list_of_lists
column_titles = [column_titles[i] for i in range(len(column_titles)) if flag[i] == False  ]
del list_of_lists

print("Length of Training and Test Data after cleaning ", len(listofTrainingData[1]),len(listofTestData[1]))

#Need to delete the test data also with above cols


# Variable importance chart
target_names = ['Less than 50k', 'More than 50k']

Question_Train=[]
Answer_Train=[]
for i in range (1,len(listofTrainingData)):
    Question_Train.append(listofTrainingData[i][:-1])
    


#    Answer_Train.append(income_level_data_train[i])
#    for j in range (0,len(continuous_train_data[0])):
#        Question_Train[i][j]=(int)(Question_Train[i][j])


#print("continuous_test_data : ",len(continuous_test_data),continuous_test_data[0:10])
X=np.array(Question_Train)
Y=np.array(income_level_data_train)
model1 = GaussianNB()
print("")
#text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', SGDClassifier()),])
#_ = text_clf.fit(Question_Train,Answer_Train)
#    predicted = text_clf.predict(docs_test)
Question_Test=[]
Answer_Test_Correct=[]
for i in range (1,len(listofTestData)):
    Question_Test.append(listofTestData[i][:-1])
     #Question_Test[i-1]=list(map(int, Question_Train[i-1]))
    
  
model1.fit(X, Y)
#
print("Naive Bayes")
predicted = model1.predict(Question_Test)
print(metrics.classification_report(income_level_data_test, predicted,target_names=target_names))
del predicted


 

print("")
print("SVM")
print("")
model_SVM=SGDClassifier()
model_SVM.fit(X,Y)
predicted = model_SVM.predict(Question_Test)
print(metrics.classification_report(income_level_data_test, predicted,target_names=target_names))
print("")


print("")
print("SVM using class_weight = Balanced")
print("")
model_SVM=SGDClassifier(class_weight='balanced')
model_SVM.fit(X,Y)
predicted = model_SVM.predict(Question_Test)
print(metrics.classification_report(income_level_data_test, predicted,target_names=target_names))
print("")




print("Smote on SVM ratio 0.4")
sm = SMOTE(ratio = 0.40, k_neighbors = 3 , kind='regular')
new_X , new_Y = sm.fit_sample(X,Y)
model_SVM= SGDClassifier()
model_SVM.fit(new_X,new_Y)
predicted = model_SVM.predict(Question_Test)
print(metrics.classification_report(income_level_data_test, predicted,target_names=target_names))
print("")



print("")
print("Random Forest")
model_RFC=RandomForestClassifier()
model_RFC.fit(X,Y)
predicted = model_RFC.predict(Question_Test)
print(metrics.classification_report(income_level_data_test, predicted,target_names=target_names))
print("")



print("")
print("Decision Tree")
model_DT=DecisionTreeClassifier()
model_DT.fit(X,Y)
predicted = model_DT.predict(Question_Test)
print(metrics.classification_report(income_level_data_test, predicted,target_names=target_names))
print("")






print("")
print("XGB Classifier")
model_XGB=XGBClassifier()
model_XGB.fit(X,Y)
predicted = model_XGB.predict(Question_Test)
print(metrics.classification_report(income_level_data_test, predicted,target_names=target_names))
print("")
del new_X
del new_Y




print("")
print("XGB Classifier Smote 0.4")
sm = SMOTE(ratio = 0.40, k_neighbors = 3 , kind='regular')
new_X , new_Y = sm.fit_sample(X,Y)
model_Smote_XGB= XGBClassifier()
model_Smote_XGB.fit(new_X,new_Y)
predicted = model_Smote_XGB.predict(Question_Test)
print(metrics.classification_report(income_level_data_test, predicted,target_names=target_names))
print("")
del new_X
del new_Y

print("")
print("XGB Classifier Smote 00.1")
sm = SMOTE(ratio = 0.10, k_neighbors = 3 , kind='regular')
new_X , new_Y = sm.fit_sample(X,Y)
model_Smote_XGB= XGBClassifier()
model_Smote_XGB.fit(new_X,new_Y)
predicted = model_Smote_XGB.predict(Question_Test)
print(metrics.classification_report(income_level_data_test, predicted,target_names=target_names))
print("")
del new_X
del new_Y






print("")
print("XGB Classifier Smote 00.07")
sm = SMOTE(ratio = 0.07, k_neighbors = 3 , kind='regular')
new_X , new_Y = sm.fit_sample(X,Y)
model_Smote_XGB= XGBClassifier()
model_Smote_XGB.fit(new_X,new_Y)
predicted = model_Smote_XGB.predict(Question_Test)
print(metrics.classification_report(income_level_data_test, predicted,target_names=target_names))
print("")
del new_X
del new_Y






#print(nominal_train_data[1:3])
#print(income_level_data_train[1:3])
#income_level_data_train = [[x] for x in income_level_data_train]
#print(len(income_level_data_train))
#print(fit(nominal_train_data[1:5],income_level_data_train[1:5]))
#print(nominal_train_data[1])
#print(len(nominal_train_data))
#clf = RandomForestClassifier().fit(nominal_train_data[1:5],income_level_data_train[1:5])
#importances = clf.feature_importances_
#print(importances)

