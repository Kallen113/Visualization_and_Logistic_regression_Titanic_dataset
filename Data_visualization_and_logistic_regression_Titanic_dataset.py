#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 21:43:18 2018

@author: kevinallen
"""

import numpy as np
import pandas as pd
from pandas import Series, DataFrame

#Imports the Titanic training dataset from Kaggle
df_titanic = pd.read_csv('Titanic_train.csv',sep=',',nrows=891)

#Displays 1st 10 rows of data.
df_titanic.head(10)

#display the number of observations and type of variable for each column from the DataFrame 
#Shows the # of observations for each variable, and also classifies each variable's data type: i.e., int, float, or object (string).
df_titanic.info()

#import data visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

'''create a column comprising an indicator variable that will equal 1 
if the given passenger has any siblings, spouses, parents, or children on board with them
:'''
df_titanic['with_family'] = (df_titanic['SibSp']>=1) | (df_titanic['Parch']>=1)

'''Convert the with_family column into a quantitative indicator variable: i.e., 0's and 1's:'''
with_family = pd.get_dummies(df_titanic['with_family'])

#create a column to differentiate passengers who were children
def male_female_child(passenger):
    age,sex = passenger
    
    if age <16:
        return 'child'
    else:
        return sex

'''Create a new column in the DataFrame to call upon the male_female_child function
so that there will be a column identifying passengers who are male, female, OR children:
'''     
df_titanic['person'] = df_titanic[['Age','Sex']].apply(male_female_child,axis=1)


'''Examine three questions:
    1.) did the deck have an effect on the passengers' survival rate? Does this match with your intuition?
    2.) Did having a family member increase the odds of surviving the Titanic?
    3.) Estimate a logistic regression to estimate how various factors actually affected the likelihood that a passenger survived the Titanic

'''
#Let's tart by implementing a scatterplot examining the relationship between having a family member, age, and survival
sns.lmplot('Age','Survived',df_titanic,hue='with_family')

'''Notice that those who had 1 or more family members were moderately more likely--around 20% higher--to have survived the Titanic.
Age also matters somewhat more significantly for those who had family, relative to those who did not
Perhaps then, elderly adults who had family were more likely to sacrifice for their younger family members (e.g., didn't go on a rescue boat, with younger family in their stead).
'''

#3.) Implement a logistic regression model:
#import statsmodels library for regression models such as logistic
import statsmodels.api as sm

#Replace the empty rows of data from the Age variable with null values
#Replaces any empty cells/rows of data from the Age column with Nan (i.e., null) values
df_titanic['Age'].replace(' ',np.nan, inplace=True)

#Drop all null values from the Age column.
df_titanic['Age'].dropna()

'''Since the person column is merely a qualitative/object data variable, 
this needs to be converted to binary/quantiative variables.
Also, delete the original column as well, since this cannot be used for 
the regression analysis:
'''
#Delete the Sex column from the DataFrame, and creates TWO binary variable columns: one called 'Male', and the other called 'Female'. 
#These column titles are automatically created by pandas based on the name of the strings from the variable 
df_titanic = pd.concat([df_titanic.drop('person', axis=1), pd.get_dummies(df_titanic['person'])], axis=1)


#Create a vector of covariates, add an intercept to it, and create a vector for the dependent variable
X= df_titanic[['Pclass', 'SibSp', 'Parch', 'female','child']]

#add constant to the RHS of the regression model
X = sm.add_constant(X)

#specify the column for the dependent variable
y= df_titanic['Survived']

#Specify the logit model 
logit = sm.Logit(y,X)


'''Finally, estimate the logit model, given the 
specified parameters, and print:
a.) the logit model coefficients and intercept, &
b.) the odds ratio for each independent variable
:'''
#Estimate the logit model
reg_results = logit.fit()

#Print the logistic regression results for the coefficients and intercept.
print(reg_results.summary())

print (np.exp(reg_results.params))

'''What do the logit regression model results show? 
It appears that each of the variables are moderately to highly statistically significant; 4 of the 5 variables are in fact
signficant at the 0.01 level. 

However, 3 of the 5 variables show a signifcant reduction in the probability that an individual would have survived the 
Titanic. Namely, a.) being in a higher-number passenger class (the higher classes referring to cheaper fares/tickets located lower
within the ship), b.) having a sibling or spouse, c.) being a parent with children on board the Titanic all were associated 
with lower odds of survival. Somewhat surprisingly, passenger class shows the largest magnitude. 

On the other hand, 2 of the variables were aossciated with a significant increase in the odds of surviving the Titanic: 
namely,passengers who were either adult females or children regardless of gender. Those who were children were the 
most likely to survive the Titanic, with adult female passengers also having a nearly-as-high increase in the odds of
their survival, relative to adult male passengers, holding constant passenger class. In fact, the magnitude of the female
and child variables is even higher than the passenger class, sibling/spouse on board, and parents with children variables combined,
so even many children and women from the less-expesnive passenger classes were more likely to survive than adult male passengers
who were on board with the most expensive passenger classes.

In short, the results of the data suggests the prevailing claim that ships in accidents/wrecks would try to prioritize and
save women and children first (adult men being the lowest on the priority list) is by and large quite true for the Titanic.
'''

