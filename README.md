# Customer-churn-prediction
(Predict whether a customer will change telco provider.) </br>

The project is based on the Kaggle competition "Customer Churn Prediction 2020", as you can find [here](https://www.kaggle.com/c/customer-churn-prediction-2020/overview/description). </br> The evaluation is based on the test Accuracy criterion: (Accuracy = Number of correct predictions/Number of total test samples). </br> </br>

The following approach managed to achieve a score of 0.99111 on Public Leaderboard and 0.97333 on Private.

## Table of Contents
* [1. Introduction](#1-introduction)
* [2. Data](#2-data explanation)




## 1. Introduction
Customer churn or customer attrition is one of the biggest expenditures of any organization. In this project we tried to foresee the customer churn prediction by training various machine learning models on a dataset provided by Kaggle(train.csv). We used various algorithms (Decision Tree,Gradient Booster, SVM, Random Forrest) Random Forrest was the most accurate algorithm . We also used correlation matrix and feature importance in  order to drop columns that are not so essential. Finally we used a confusion matrix to calculate precision, recall and the f1-score.

## 2. Data
The dataset is consisted in total of 5000 instances, of two separate files, a training set with 4250 cases and a testing set with 750 cases. The training set is composed of 19 features and one Boolean variable “churn”, which consists our target variable. The given columns from the training set are summarized in the following table: </br>
ATTRIBUTES OF THE TRAINING SET  </br>
| **Col. No** | **Attribute Name** | **Type** | **Description of the Attribute** |
| :--- | :--- | :--- | :--- |
| 1 | state | string | 2-letter code of the US state of customer residence. |
| 2 | account_length | numerical | Number of months the customer has been with the current telco provider. |
| 3 | area_code | string | 3 digit area code. |
| 4 | international_plan | boolean | The customer has international plan. |
| 5 | voice_mail_plan | boolean | The customer has voice mail plan. |
| 6 | number_vmail_messages | numerical | Number of voice-mail messages. |
| 7 | total_day_minutes | numerical | Total minutes of day calls. |
| 8 | total_day_calls | numerical | Total number of day calls. |
| 9 | total_day_charge | numerical | Total charge of day calls. |
| 10 | total_eve_minutes | numerical | Total minutes of evening calls. |
| 11 | total_eve_calls | numerical | Total number of evening calls. |
| 12 | total_eve_charge | numerical | Total charge of evening calls. |
| 13 | total_night_minutes | numerical | Total minutes of night calls. |
| 14 | total_night_calls | numerical | Total number of night calls. |
| 15 | total_night_charge | numerical | Total charge of night calls. |
| 16 | total_intl_minutes | numerical | Total minutes of international calls. |
| 17 | total_intl_calls | numerical | Total number of international calls. |
| 18 | total_intl_charge | numerical | Total charge of international calls. |
| 19 | number_customer_service_calls | numerical | Number of calls to customer service. |
| 20 | churn | boolean | Customer churn - target variable. |

## 3. Exploratory Data Analysis
The most significant part of our research is the EDA sector. Here you can see the graphs we generated. 

### Checking for imbalance
Estimate the churn percentage: 

![imbalance](https://user-images.githubusercontent.com/81081046/113518796-2b0cfc80-9591-11eb-8e55-4db140420963.png)</br>

### Churn By Customers with Voice mail plan & International plan
Estimate churn customers with Voice mail plan & International plan 

![Group by churn](https://user-images.githubusercontent.com/81081046/113519341-39104c80-9594-11eb-9c5f-5d4257fe9bf5.png)

### plot the distribution of the numerical features 
Churn distribution plots 

![total calls](https://user-images.githubusercontent.com/81081046/113519463-0ca90000-9595-11eb-98f3-952bff38cfaf.png)

Plot total call distribution </br> </br> 
![total charge](https://user-images.githubusercontent.com/81081046/113519474-28aca180-9595-11eb-9118-beeb7ed6981c.png)

Plot total charge distribution </br> </br> 
![total mins](https://user-images.githubusercontent.com/81081046/113519477-2fd3af80-9595-11eb-8830-eed07f051b3e.png)

Plot total mins distribution </br> </br> 

### Correlation Matrix

![correlation matrix](https://user-images.githubusercontent.com/81081046/113519955-69f28080-9598-11eb-935f-325cda968ede.png)

## 4. Feature Engineering
The most important phase of our project aiming to achieve the best accuracy score, is feature engineering. We used label encoding ni order to convert labels into numerical values.We also merged Features and used correlation matrix to drop unwanted columns (see above our plot). Finally significant procedure that contributes to obtaining superior results is that of the dummy’s method to create dummy variables. 

### A. Transform labels to numerical values 

```ruby
label_encoder = preprocessing.LabelEncoder()

train_df['international_plan'] = label_encoder.fit_transform(train_df['international_plan'])
train_df['voice_mail_plan'] = label_encoder.fit_transform(train_df['voice_mail_plan'])
train_df['churn'] = label_encoder.fit_transform(train_df['churn'])
```

### B. Merge Features

```ruby

#merge all matching features 

train_df['total_minutes']=train_df.total_day_minutes + train_df.total_eve_minutes + train_df.total_night_minutes + train_df.total_intl_minutes
train_df['total_calls']=train_df.total_day_calls + train_df.total_eve_calls + train_df.total_night_calls + train_df.total_intl_calls
train_df['total_charge']=train_df.total_day_charge + train_df.total_eve_charge + train_df.total_night_charge + train_df.total_intl_charge
train_df['total_hours'] = train_df.total_minutes/60
```

### C. Dummies for categorical features
First we created to groups to based on the proportion of the churners and convert state and are code into dummies in order to enhabce our classifiers result.

```ruby

# Create two bins for State based on the proportion of churners
dn = train_df.groupby(by='state').agg(lambda x: x.sum()/ x.count()).reset_index()
np.mean(dn.churn.values)
group_A = (dn.loc[dn.churn>0.14]).state.values
group_B = (dn.loc[dn.churn<=0.14]).state.values


# Convert State and Area_code into dummies 
def preprocess(df):
    df_dummies = pd.get_dummies(df.area_code) 
    df=pd.concat([df_dummies,df],axis=1)
    
    for i in group_A:
        for j in group_B:
            df.state.replace((i,j), ('group_A','group_B'), inplace=True)
    
    df_dummies2 = pd.get_dummies(df.state)
    df=pd.concat([df_dummies2,df],axis=1)
    
    df.drop(['state','area_code'],inplace=True,axis=1)
    return df
	

train_df=preprocess(train_df)
```
## 5. Machine Learning Classifiers 
A Cross Validation using stratified kfold with 10 splits was implemented in order to train the following classifiers: </br>

- Support Vector Machines
- Random Forest
- Gradient Boosting Machine
- Decision Tree

## . Train our Classifiers 
We trained our classifiers using also Hyperparameter tuning aiming to achive the best accuracy score .

Here we used Hyperparameter tuning trying to find the most efficient combination to achive the best score.
```ruby

#Set parameters to choose Random Forest
n_estimators = [100, 300, 500, 800, 1200]
max_depth = [5, 8, 15, 25, 30]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10] 

hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  
              min_samples_split = min_samples_split, 
             min_samples_leaf = min_samples_leaf)
# Create a based model
forest = RandomForestClassifier()
gridF  = GridSearchCV(estimator = forest, param_grid = hyperF, cv = 3, n_jobs = -1, verbose = 2)
bestF = gridF.fit(X_train, y_train)
print(bestF.best_params_)
```

Now we trained the optimized model 
```ruby
#Optimized Random Forest Classifier
forestOpt = RandomForestClassifier(random_state = 1, max_depth = 25,n_estimators = 800, min_samples_split = 2, min_samples_leaf = 2)    
forestOpt.fit(X_train,y_train)
print('Accuracy of the RFC on test set: {:.3f}'.format(forestOpt.score(X_test, y_test)))
fpred=forestOpt.predict(X_test)
print(classification_report(y_test, fpred))
```

Our Results 
```ruby

Accuracy of the RFC on test set: 0.979
              precision    recall  f1-score   support

           0       0.98      1.00      0.99      1102
           1       1.00      0.84      0.92       173

    accuracy                           0.98      1275
   macro avg       0.99      0.92      0.95      1275
weighted avg       0.98      0.98      0.98      1275
```


## 7. Evaluation of classifiers
We can see now summarized all our results and scores for each classifier . We used confusion matrix to measure all classifiers .Furthermore, learning curves (ROC) and feature importance was implemented in order to find the most efficient classifier 

### A. Confusion Matrix for all classifiers 
```ruby
#Making confusion matrix 
forest_cm = metrics.confusion_matrix(fpred,y_test)
print("Confusion Matrix of random forest:\n",forest_cm)
tree_cm = metrics.confusion_matrix(tree_pred,y_test)
print("Confusion Matrix of Desicion Tree:\n",tree_cm)
gradient_boooster_cm = metrics.confusion_matrix(gb_pred,y_test)
print("Confusion Matrix of Gradient booster :\n",gradient_boooster_cm)
SVM_cm = metrics.confusion_matrix(svm_pred,y_test)
print("Confusion Matrix of SVM:\n",SVM_cm)
```

Results 
```ruby
Confusion Matrix of random forest:
 [[1102   27]
 [   0  146]]
Confusion Matrix of Desicion Tree:
 [[1102   64]
 [   0  109]]
Confusion Matrix of Gradient booster :
 [[1102   34]
 [   0  139]]
Confusion Matrix of SVM:
 [[1102  169]
 [   0    4]]
 ```
 
 ### B. ROC curves 
 
 ![ROC](https://user-images.githubusercontent.com/81081046/113521166-5d722600-95a0-11eb-964b-520575dff57e.png)

Learning curves </br> </br> 

 ### B. Feature Importance
Feature importance was implemented in Random Forest clasiffier aiming to see the most important feature. 

![feature importance](https://user-images.githubusercontent.com/81081046/113521308-4ed83e80-95a1-11eb-9457-7d659b8a3e03.png)

SUMMARAZATION RESULTS </br>
| **Classifiers** | **Accuracy(%)** |
| :--- | :--- |
| **SVM** | 86.7 |
| **Random Forest** | 97.9 |
| **Gradient Boosting** | 97.3 |
| **Decision Tree** | 95.0 |
