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
imbalance </br>




