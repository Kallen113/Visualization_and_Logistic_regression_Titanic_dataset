#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 21:43:18 2018

@author: kevinallen
"""

import numpy as np
import pandas as pd
from pandas import Series, DataFrame

df_titanic = pd.read_csv('Titanic_train.csv',sep=',',nrows=891)
#Imports the Titanic training dataset from Kaggle

df_titanic.head(10)
#Displays 1st 10 rows of data.

'''display the number of observations and type of variable for each column from the DataFrame:''' 
df_titanic.info()
#Shows the # of observations for each variable, and also classifies each variable's data type: i.e., int, float, or object (string).

import matplotlib.pyplot as plt
import seaborn as sns

df_titanic = pd.read_csv('Titanic_train.csv',sep=',',nrows=891)

'''Recreate the column comprising an indicator variable that will 
equal 1 if the given passenger has any siblings, spouses, parents, or children on board with them
:'''
df_titanic['with_family'] = (df_titanic['SibSp']>=1) | (df_titanic['Parch']>=1)

'''Convert the with_family column into a quantitative indicator variable: i.e., 0's and 1's:'''
with_family = pd.get_dummies(df_titanic['with_family'])

'''Recreate the person column to differentiate passengers who were children
:'''
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
'''2.) Did having a family member increase one's odds of survival?
'''
'''Implement a scatterplot examining the relationship between having a family member, age, and survival
:'''
sns.lmplot('Age','Survived',df_titanic,hue='with_family')
#Notice that those who had 1 or more family members were moderately more likely--around 20% higher--to have survived the Titanic
#Age also matters somewhat more significantly for those who had family, relative to those who did not
#Perhaps then, elderly adults who had family were more likely to sacrifice for their younger family members (e.g., didn't go on a rescue boat, with younger family in their stead).
#

'''3.) Implement a logistic regression model:
'''
import statsmodels.api as sm

'''Replace the empty rows of data from the Age variable with null values:'''
df_titanic['Age'].replace(' ',np.nan, inplace=True)
#Replaces any empty cells/rows of data from the Age column with Nan (i.e., null) values

df_titanic['Age'].dropna()
#Drops all null values from the Age column.

'''Since the person column is merely a qualitative/object data variable, 
this needs to be converted to binary variables.
Also, delete the original column as well, since this cannot be used for 
the regression analysis:
'''
df_titanic = pd.concat([df_titanic.drop('person', axis=1), pd.get_dummies(df_titanic['person'])], axis=1)
#This deletes the Sex column from the DataFrame, and creates TWO binary variable columns: one called 'Male', and the other called 'Female'. 
#These column titles are automatically created by pandas based on the name of the strings from the variable 

'''Create the vector of covariates, add an intercept to it, and create a vector for the dependent variable
'''
X= df_titanic[['Pclass', 'SibSp', 'Parch', 'female','child']]
'''IMPORTANT NOTE:
    the code is running with errors. It's probably associated with the variables I'm using 
    as covariates:
        perhaps there are some null values, or some of hte data is actually qualitative and not quantiative
UPDATE!: I got the code to work once I concatenated the DataFrame with the qualitative person variable to a binary using the .get_dummies() variable
in tandem with the pd.concat() command'''

X = sm.add_constant(X)

y= df_titanic['Survived']

logit = sm.Logit(y,X)
#Specifies the logit model 
'''Finally, estimate the logit model, given the 
specified parameters, and print:
a.) the logit model coefficients and intercept, &
b.) the odds ratio for each independent variable
:'''
reg_results = logit.fit()
#Estimates the logit model

print(reg_results.summary())
#Prints the logistic regression results for the coefficients and intercept.

print (np.exp(reg_results.params))
#Prints the log of the odds ratio for each of the coefficients and the intercept. 