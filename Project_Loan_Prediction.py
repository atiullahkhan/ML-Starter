# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 20:10:53 2019

@author: Lenovo
"""

import pandas as pd
import numpy as np                     # For mathematical calculations
import seaborn as sns                  # For data visualization
import matplotlib.pyplot as plt        # For plotting graphs
%matplotlib inline
import warnings                        # To ignore any warnings
warnings.filterwarnings("ignore")

train=pd.read_csv("G:/Practise/Loan Prediction Problem/train_u6lujuX_CVtuZ9i.csv")
test=pd.read_csv("G:/Practise/Loan Prediction Problem/test_Y3wMUE5_7gLdaTN.csv")

# Copy of original data set
train_original=train.copy()
test_original=test.copy()

train.head()
train.columns
test.columns
train.describe()
train['Property_Area'].value_counts()
train.dtypes
train.shape
test.shape

train['Loan_Status'].value_counts()
# Normalize can be set to True to print proportions instead of number 
train['Loan_Status'].value_counts(normalize=True)

train['Loan_Status'].value_counts().plot.bar()

# Now lets visualize each variable separately. Different types of variables are Categorical, ordinal and numerical.
# Categorical features: These features have categories (Gender, Married, Self_Employed, Credit_History, Loan_Status)
# Ordinal features: Variables in categorical features having some order involved (Dependents, Education, Property_Area)
# Numerical features: These features have numerical values (ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term)

# Independent Variable (Categorical)
plt.figure(1)
plt.subplot(221)
train['Gender'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'Gender')

plt.subplot(222)
train['Married'].value_counts(normalize=True).plot.bar(title= 'Married')

plt.subplot(223)
train['Self_Employed'].value_counts(normalize=True).plot.bar(title= 'Self_Employed')

plt.subplot(224)
train['Credit_History'].value_counts(normalize=True).plot.bar(title= 'Credit_History')

plt.show()

#It can be inferred from the above bar plots that:
# 80% applicants in the dataset are male.
# Around 65% of the applicants in the dataset are married.
# Around 15% applicants in the dataset are self employed.
# Around 85% applicants have repaid their debts.

# Independent Variable (Ordinal)
plt.figure(1)
plt.subplot(131)
train['Dependents'].value_counts(normalize=True).plot.bar(figsize=(24,6), title= 'Dependents')

plt.subplot(132)
train['Education'].value_counts(normalize=True).plot.bar(title= 'Education')

plt.subplot(133)
train['Property_Area'].value_counts(normalize=True).plot.bar(title= 'Property_Area')

plt.show()

# Following inferences can be made from the above bar plots:
# Most of the applicants don’t have any dependents.
# Around 80% of the applicants are Graduate.
# Most of the applicants are from Semiurban area.

# Independent Variable (Numerical)
# Lets look at the distribution of Applicant income first.
plt.figure(1)
plt.subplot(121)
sns.distplot(train['ApplicantIncome']);

plt.subplot(122)
train['ApplicantIncome'].plot.box(figsize=(16,5))

plt.show()

# It can be inferred that most of the data in the distribution of applicant income is towards
# left which means it is not normally distributed. We will try to make it normal in later 
# sections as algorithms works better if the data is normally distributed.

# The boxplot confirms the presence of a lot of outliers/extreme values. This can be attributed
# to the income disparity in the society. Part of this can be driven by the fact that we are
# looking at people with different education levels. Let us segregate them by Education:
train.boxplot(column='ApplicantIncome', by = 'Education')
plt.suptitle("")

# Let’s look at the Coapplicant income distribution.
plt.figure(1)
plt.subplot(121)
sns.distplot(train['CoapplicantIncome']);

plt.subplot(122)
train['CoapplicantIncome'].plot.box(figsize=(16,5))

plt.show()

# We see a similar distribution as that of the applicant income. Majority of coapplicant’s income ranges from 0 to 5000.
# We also see a lot of outliers in the coapplicant income and it is not normally distributed.

# Let’s look at the distribution of LoanAmount variable.
plt.figure(1)
plt.subplot(121)
df=train.dropna()
sns.distplot(df['LoanAmount']);

plt.subplot(122)
train['LoanAmount'].plot.box(figsize=(16,5))

plt.show()

# We see a lot of outliers in this variable and the distribution is fairly normal.
# We will treat the outliers in later sections.

# Lets recall some of the hypotheses that we generated earlier:

# 1. Applicants with high income should have more chances of loan approval.
# 2. Applicants who have repaid their previous debts should have higher chances of loan approval.
# 3. Loan approval should also depend on the loan amount. If the loan amount is less, chances of loan approval should be high.
# 4. Lesser the amount to be paid monthly to repay the loan, higher the chances of loan approval.

# Lets try to test the above mentioned hypotheses using bivariate analysis
# After looking at every variable individually in univariate analysis, we will now explore them again with respect to the target variable.

# Categorical Independent Variable vs Target Variable
Gender=pd.crosstab(train['Gender'],train['Loan_Status'])
Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))

# It can be inferred that the proportion of male and female applicants is more or less same for both approved and unapproved loans.

# Now let us visualize the remaining categorical variables vs target variable.

Married=pd.crosstab(train['Married'],train['Loan_Status'])
Dependents=pd.crosstab(train['Dependents'],train['Loan_Status'])
Education=pd.crosstab(train['Education'],train['Loan_Status'])
Self_Employed=pd.crosstab(train['Self_Employed'],train['Loan_Status'])

Married.div(Married.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.show()

Dependents.div(Dependents.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.show()

Education.div(Education.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.show()

Self_Employed.div(Self_Employed.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.show()

# Inference
# 1. Proportion of married applicants is higher for the approved loans.
# 2. Distribution of applicants with 1 or 3+ dependents is similar across both the categories of Loan_Status.
# 3. There is nothing significant we can infer from Self_Employed vs Loan_Status plot.

# Now we will look at the relationship between remaining categorical independent variables and Loan_Status.
Credit_History=pd.crosstab(train['Credit_History'],train['Loan_Status'])
Property_Area=pd.crosstab(train['Property_Area'],train['Loan_Status'])

Credit_History.div(Credit_History.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))
plt.show()

Property_Area.div(Property_Area.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.show()

# Inference
# It seems people with credit history as 1 are more likely to get their loans approved.
# Proportion of loans getting approved in semiurban area is higher as compared to that in rural or urban areas.

# Numerical Independent Variable vs Target Variable

# We will try to find the mean income of people for which the loan has been approved vs the mean income of people for which the loan has not been approved.

train.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar()

# Here the y-axis represents the mean applicant income. We don’t see any change in the mean income.
# So, let’s make bins for the applicant income variable based on the values in it and analyze the corresponding loan status for each bin.

bins=[0,2500,4000,6000,81000]
group=['Low','Average','High', 'Very high']
df['ApplicantIncome']   # a look of the column
train['Income_bin']=pd.cut(df['ApplicantIncome'],bins,labels=group)

train[['ApplicantIncome','Income_bin']]

Income_bin=pd.crosstab(train['Income_bin'],train['Loan_Status'])
Income_bin.div(Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('ApplicantIncome')
P = plt.ylabel('Percentage')

# It can be inferred that Applicant income does not affect the chances of loan approval
# which contradicts our hypothesis in which we assumed that
# if the applicant income is high the chances of loan approval will also be high.

# We will analyze the coapplicant income and loan amount variable in similar manner.

bins=[0,1000,3000,42000]
group=['Low','Average','High']
train['Coapplicant_Income_bin']=pd.cut(df['CoapplicantIncome'],bins,labels=group)

Coapplicant_Income_bin=pd.crosstab(train['Coapplicant_Income_bin'],train['Loan_Status'])
Coapplicant_Income_bin.div(Coapplicant_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('CoapplicantIncome')
P = plt.ylabel('Percentage')

# It shows that if coapplicant’s income is less the chances of loan approval are high.
# But this does not look right. The possible reason behind this may be that most of
# the applicants don’t have any coapplicant so the coapplicant income for such applicants is 0
# and hence the loan approval is not dependent on it. So we can make a new variable in which
# we will combine the applicant’s and coapplicant’s income to visualize the combined effect of income on loan approval.

# Let us combine the Applicant Income and Coapplicant Income and see the combined effect of Total Income on the Loan_Status.

train['Total_Income']=train['ApplicantIncome']+train['CoapplicantIncome']

bins=[0,2500,4000,6000,81000]
group=['Low','Average','High', 'Very high']
train['Total_Income_bin']=pd.cut(train['Total_Income'],bins,labels=group)

Total_Income_bin=pd.crosstab(train['Total_Income_bin'],train['Loan_Status'])
Total_Income_bin.div(Total_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('Total_Income')
P = plt.ylabel('Percentage')

# We can see that Proportion of loans getting approved for applicants having low
# Total_Income is very less as compared to that of applicants with Average, High and Very High Income.

# Let’s visualize the Loan amount variable.

bins=[0,100,200,700]
group=['Low','Average','High']
train['LoanAmount_bin']=pd.cut(df['LoanAmount'],bins,labels=group)

LoanAmount_bin=pd.crosstab(train['LoanAmount_bin'],train['Loan_Status'])
LoanAmount_bin.div(LoanAmount_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('LoanAmount')
P = plt.ylabel('Percentage')

# It can be seen that the proportion of approved loans is higher for Low and Average Loan Amount
# as compared to that of High Loan Amount which supports our hypothesis in which we considered
# that the chances of loan approval will be high when the loan amount is less.

# Let’s drop the bins which we created for the exploration part.
# We will change the 3+ in dependents variable to 3 to make it a numerical variable.
# We will also convert the target variable’s categories into 0 and 1 so that we can find
# its correlation with numerical variables. One more reason to do so is few models like
# logistic regression takes only numeric values as input. We will replace N with 0 and Y with 1.

train=train.drop(['Income_bin', 'Coapplicant_Income_bin', 'LoanAmount_bin', 'Total_Income_bin', 'Total_Income'], axis=1)

train['Dependents'].replace('3+', 3,inplace=True)
test['Dependents'].replace('3+', 3,inplace=True)
train['Loan_Status'].replace('N', 0,inplace=True)
train['Loan_Status'].replace('Y', 1,inplace=True)

# Now lets look at the correlation between all the numerical variables.
# We will use the heat map to visualize the correlation.
# Heatmaps visualize data through variations in coloring.
# The variables with darker color means their correlation is more.

matrix = train.corr()
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu")

# We see that the most correlated variables are (ApplicantIncome - LoanAmount) and
# (Credit_History - Loan_Status). LoanAmount is also correlated with CoapplicantIncome.

# After exploring all the variables in our data, we can now impute the missing values and treat
# the outliers because missing data and outliers can have adverse effect on the model performance.

# ------------------------------------
# Missing Value and Outlier Treatment
# ------------------------------------

# After exploring all the variables in our data, we can now impute the missing values and
# treat the outliers because missing data and outliers can have adverse effect on the model performance.

# Missing value imputation

# Let’s list out feature-wise count of missing values.

train.isnull().sum()

# There are missing values in Gender, Married, Dependents, Self_Employed, LoanAmount,
# Loan_Amount_Term and Credit_History features.
# We will treat the missing values in all the features one by one.

# We can consider these methods to fill the missing values:

# 1. For numerical variables: imputation using mean or median
# 2. For categorical variables: imputation using mode

# There are very less missing values in Gender, Married, Dependents, Credit_History and Self_Employed features
# so we can fill them using the mode of the features.

train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)

# Now let’s try to find a way to fill the missing values in Loan_Amount_Term.
# We will look at the value count of the Loan amount term variable.

train['Loan_Amount_Term'].value_counts()

# It can be seen that in loan amount term variable, the value of 360 is repeating the most.
# So we will replace the missing values in this variable using the mode of this variable.

train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)

# Now we will see the LoanAmount variable. As it is a numerical variable,
# we can use mean or median to impute the missing values.
# We will use median to fill the null values as earlier we saw that
# loan amount have outliers so the mean will not be the proper approach as
# it is highly affected by the presence of outliers.

train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

# Now lets check whether all the missing values are filled in the dataset.

train.isnull().sum()

# As we can see that all the missing values have been filled in the test dataset.
# Let’s fill all the missing values in the test dataset too with the same approach.

test['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
test['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
test['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
test['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
test['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

# Due to these outliers bulk of the data in the loan amount is at the left and the right tail
# is longer. This is called right skewness. One way to remove the skewness is by doing the log
# transformation. As we take the log transformation, it does not affect the smaller values much,
# but reduces the larger values. So, we get a distribution similar to normal distribution.

# Let’s visualize the effect of log transformation. We will do the similar changes to the test file simultaneously.

train['LoanAmount_log'] = np.log(train['LoanAmount'])
train['LoanAmount_log'].hist(bins=20)
test['LoanAmount_log'] = np.log(test['LoanAmount'])

# Now the distribution looks much closer to normal and effect of extreme values has been significantly subsided.
# Let’s build a logistic regression model and make predictions for the test dataset.

# -------------------------
# Model Building : Part I
# -------------------------

# Lets drop the Loan_ID variable as it do not have any effect on the loan status.
# We will do the same changes to the test dataset which we did for the training dataset.

train=train.drop('Loan_ID',axis=1)
test=test.drop('Loan_ID',axis=1)

# Sklearn requires the target variable in a separate dataset.
# So, we will drop our target variable from the train dataset and save it in another dataset.

X = train.drop('Loan_Status',1)
y = train.Loan_Status

# Now we will make dummy variables for the categorical variables.

X=pd.get_dummies(X)
train=pd.get_dummies(train)
test=pd.get_dummies(test)

# We will use the train_test_split function from sklearn to divide our train dataset.
# So, first let us import train_test_split.

from sklearn.model_selection import train_test_split

x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size =0.3)

# The dataset has been divided into training and validation part.
# Let us import LogisticRegression and accuracy_score from sklearn and
# fit the logistic regression model.

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression()
model.fit(x_train, y_train)

# Let’s predict the Loan_Status for validation set and calculate its accuracy.

pred_cv = model.predict(x_cv)

# Let us calculate how accurate our predictions are by calculating the accuracy.

accuracy_score(y_cv,pred_cv)

# So our predictions are almost 80% accurate, i.e. we have identified 80% of the loan status correctly.

# Let’s make predictions for the test dataset.

pred_test = model.predict(test)

# Lets import the submission file which we have to submit on the solution checker.

submission=pd.read_csv("G:/Practise/Loan Prediction Problem/Sample_Submission.csv")

# We only need the Loan_ID and the corresponding Loan_Status for the final submission.
# we will fill these columns with the Loan_ID of test dataset and the predictions that we made,
# i.e., pred_test respectively.

submission['Loan_Status']=pred_test
submission['Loan_ID']=test_original['Loan_ID']

# Remember we need predictions in Y and N. So let’s convert 1 and 0 to Y and N.

submission['Loan_Status'].replace(0, 'N',inplace=True)
submission['Loan_Status'].replace(1, 'Y',inplace=True)

# Finally we will convert the submission to .csv format and make submission
# to check the accuracy on the leaderboard.

pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('logistic.csv')

# From this submission we got an accuracy of 0.7847 on the leaderboard.