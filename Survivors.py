import numpy as np
import pandas as pd
pd.set_option('display.width',1000)
pd.set_option('display.max_column',15)
pd.set_option('precision',2)

#visualization libraries
import matplotlib.pyplot as plt
import seaborn as sbn

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

#import train and test CSV files
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(train.describe()[:])

print(train.describe(include="all"))

print("\n\n", train.columns)

print(train.head())

print(train.sample(5))

print("Data types for each feature : -")
print(train.dtypes)

print(train.describe(include="all"))

print(pd.isnull(train).sum())

sbn.barplot(x="Sex", y="Survived", data=train)
plt.show()

print(train)

print("------------------\n\n")
print(train["Survived"])

print("------------------\n\n")
print(train["Sex"] == 'female')

print("**********\n\n")
print(train["Survived"][train["Sex"] == 'female'])

print("*****************\n\n")
print(train["Survived"][train["Sex"] == 'female'].value_counts() )

print("====================================\n\n")
print(train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True))

print(train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True)[1])

print("Percentage of females who survived:", train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True)[1]*100  )
print("Percentage of males who survived:", train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True)[1]*100  )

sbn.barplot(x="Pclass", y="Survived", data=train)
plt.show()

print("Percentage of Pclass = 1 who survived:", train["Survived"][train["Pclass"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of Pclass = 2 who survived:", train["Survived"][train["Pclass"] == 2].value_counts(normalize = True)[1]*100)

print("Percentage of Pclass = 3 who survived:", train["Survived"][train["Pclass"] == 3].value_counts(normalize = True)[1]*100)

print("Percentage of Pclass = 1 who survived:\n\n", train["Survived"][train["Pclass"] == 1].value_counts())

print()
print("Percentage of Pclass = 1 who survived:\n\n", train["Survived"][train["Pclass"] == 1].value_counts(normalize = True)    )

print()
print("Percentage of Pclass = 1 who survived:\n\n", train["Survived"][train["Pclass"] == 1].value_counts(normalize = True)[1]     )

sbn.barplot(x="SibSp", y="Survived", data=train)

print("Percentage of SibSp = 0 who survived:",
      train["Survived"][train["SibSp"] == 0].value_counts(normalize = True)[1]*100)

print("Percentage of SibSp = 1 who survived:",
      train["Survived"][train["SibSp"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of SibSp = 2 who survived:",
      train["Survived"][train["SibSp"] == 2].value_counts(normalize = True)[1]*100)

plt.show()

sbn.barplot(x="Parch", y="Survived", data=train)
plt.show()

train["Age"] = train["Age"].fillna(-0.5)
test["Age"] = test["Age"].fillna(-0.5)

bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)
test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)
print(train)

sbn.barplot(x="AgeGroup", y="Survived", data=train)
plt.show()

train["CabinBool"] = (train["Cabin"].notnull().astype('int'))
test["CabinBool"] = (test["Cabin"].notnull().astype('int'))

print("###################################\n\n")
print(train)

print("Percentage of CabinBool = 1 who survived:",
      train["Survived"][train["CabinBool"] == 1].value_counts(
                                     normalize = True)[1]*100)

print("Percentage of CabinBool = 0 who survived:",
      train["Survived"][train["CabinBool"] == 0].value_counts(
                                     normalize = True)[1]*100)

sbn.barplot(x="CabinBool", y="Survived", data=train)
plt.show()

print(test.describe(include="all"))

train = train.drop(['Cabin'], axis = 1)
test = test.drop(['Cabin'], axis = 1)

train = train.drop(['Ticket'], axis = 1)
test = test.drop(['Ticket'], axis = 1)

print("Number of people embarking in Southampton (S):" ,  )


print("\n\nSHAPE = " , train[train["Embarked"] == "S"].shape)
print("SHAPE[0] = " , train[train["Embarked"] == "S"].shape[0])

southampton = train[train["Embarked"] == "S"].shape[0]
print(southampton)

print("Number of people embarking in Cherbourg (C):",)
cherbourg = train[train["Embarked"] == "C"].shape[0]
print( cherbourg  )

print("Number of people embarking in Queenstown (Q):" ,)
queenstown = train[train["Embarked"] == "Q"].shape[0]
print(queenstown)

train = train.fillna({"Embarked": "S"})

combine = [train, test]
print(combine[0])

for dataset in combine:
    dataset['Title'] = dataset['Name'].str.extract(', ([A-Za-z]+)\.', expand=False)

print("\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")
print(train)
print()

print(pd.crosstab(train['Title'], train['Sex']))

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(
       ['Lady', 'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'],
        'Rare')

    dataset['Title'] = dataset['Title'].replace(['Countess', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

print("\n\nAfter grouping rare title : \n" , train)


print(train[['Title', 'Survived']].groupby(['Title'],
                                    as_index=False).count())

print("\nMap each of the title groups to a numerical value.")
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}

for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

print("\n\nAfter replacing title with neumeric values.\n")
print(train)

mr_age = train[train["Title"] == 1]["AgeGroup"].mode() # Mr.= Young Adult
print("mode() of mr_age : ", mr_age)

miss_age = train[train["Title"] == 2]["AgeGroup"].mode()  #Miss.= Student
print("mode() of miss_age : ", miss_age)

mrs_age = train[train["Title"] == 3]["AgeGroup"].mode() #Mrs.= Adult
print("mode() of mrs_age : ", mrs_age)

master_age = train[train["Title"] == 4]["AgeGroup"].mode() # Baby
print("mode() of master_age : ", master_age)

royal_age = train[train["Title"] == 5]["AgeGroup"].mode() # Adult
print("mode() of royal_age : ", royal_age)

rare_age = train[train["Title"] == 6]["AgeGroup"].mode()  # Adult
print("mode() of rare_age : ", rare_age)

print("\n\n**************************************************\n\n")
print(train.describe(include="all"))
print(train)

print("\n\n********   train[AgeGroup][0] :  \n\n")

for x in range(10) :
    print(train["AgeGroup"][x])

age_title_mapping = {1: "Young Adult", 2: "Student",
                3: "Adult", 4: "Baby", 5: "Adult", 6: "Adult"}

for x in range(len(train["AgeGroup"])):
    if train["AgeGroup"][x] == "Unknown":
        train["AgeGroup"][x] = age_title_mapping[train["Title"][x]]

for x in range(len(test["AgeGroup"])):
    if test["AgeGroup"][x] == "Unknown":
        test["AgeGroup"][x] = age_title_mapping[test["Title"][x]]

print("\n\nAfter replacing Unknown values from AgeGroup column : \n")
print(train)

age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3,
               'Student': 4, 'Young Adult': 5,
               'Adult': 6, 'Senior': 7}

train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
test['AgeGroup'] = test['AgeGroup'].map(age_mapping)

print(train)

train = train.drop(['Age'], axis=1)
test = test.drop(['Age'], axis=1)

print("\n\nAge column droped.")
print(train)

train = train.drop(['Name'], axis = 1)
test = test.drop(['Name'], axis = 1)

sex_mapping = {"male": 0, "female": 1}
train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)

print(train)

embarked_mapping = {"S": 1, "C": 2, "Q": 3}
train['Embarked'] = train['Embarked'].map(embarked_mapping)
test['Embarked'] = test['Embarked'].map(embarked_mapping)

print(train.head())

for x in range(len(test["Fare"])):
    if pd.isnull(test["Fare"][x]):
        pclass = test["Pclass"][x]
        test["Fare"][x] = round(train[train["Pclass"] == pclass ]["Fare"].mean(), 2)

train['FareBand'] = pd.qcut(train['Fare'], 4,
                            labels = [1, 2, 3, 4])

test['FareBand'] = pd.qcut(test['Fare'], 4,
                           labels = [1, 2, 3, 4])

train = train.drop(['Fare'], axis = 1)
test = test.drop(['Fare'], axis = 1)

print("\n\nFare column droped\n")
print(train)

print()
print(test.head())

from sklearn.model_selection import train_test_split

input_predictors = train.drop(['Survived', 'PassengerId'], axis=1)
ouptut_target = train["Survived"]

x_train, x_val, y_train, y_val=train_test_split(
    input_predictors, ouptut_target, test_size = 0.20, random_state = 7)

from sklearn.metrics import accuracy_score

#MODEL-1) LogisticRegression

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print("MODEL-1: Accuracy of LogisticRegression : ", acc_logreg)

#MODEL-2) Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
print("MODEL-2: Accuracy of GaussianNB : ", acc_gaussian)

#MODEL-3) Support Vector Machines

from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_val)
acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print("MODEL-3: Accuracy of Support Vector Machines : ", acc_svc)

#MODEL-4) Linear SVC

from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_val)
acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print("MODEL-4: Accuracy of LinearSVC : ",acc_linear_svc)

#MODEL-5) Perceptron

from sklearn.linear_model import Perceptron

perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_val)
acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)
print("MODEL-5: Accuracy of Perceptron : ",acc_perceptron)

#MODEL-6) Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print("MODEL-6: Accuracy of DecisionTreeClassifier : ", acc_decisiontree)

#MODEL-7) Random Forest

from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print("MODEL-7: Accuracy of RandomForestClassifier : ",acc_randomforest)

#MODEL-8) KNN or k-Nearest Neighbors

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print("MODEL-8: Accuracy of k-Nearest Neighbors : ",acc_knn)

#MODEL-9) Stochastic Gradient Descent

from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_val)
acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)
print("MODEL-9: Accuracy of Stochastic Gradient Descent : ",acc_sgd)

#MODEL-10) Gradient Boosting Classifier

from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
print("MODEL-10: Accuracy of GradientBoostingClassifier : ",acc_gbk)

models = pd.DataFrame({
    'Model': ['Logistic Regression','Gaussian Naive Bayes','Support Vector Machines',
              'Linear SVC', 'Perceptron',  'Decision Tree',
              'Random Forest', 'KNN','Stochastic Gradient Descent',
              'Gradient Boosting Classifier'],
    'Score': [acc_logreg, acc_gaussian, acc_svc,
              acc_linear_svc, acc_perceptron,  acc_decisiontree,
              acc_randomforest,  acc_knn,  acc_sgd, acc_gbk]
                    })

print(models.sort_values(by='Score', ascending=False))

ids = test['PassengerId']
predictions = randomforest.predict(test.drop('PassengerId', axis=1))

output = pd.DataFrame({'PassengerId': ids, 'Survived': predictions})
output.to_csv('submission.csv', index=False)

print("All survival predictions done.")
print("All predictions exported to submission.csv file.")

print(output)

