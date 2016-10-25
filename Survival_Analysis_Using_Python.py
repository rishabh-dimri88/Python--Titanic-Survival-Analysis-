# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 01:48:06 2016

@author: Rishabh
"""

import pandas as pd
from pandas import Series,DataFrame

titanic_df=pd.read_csv('C:\\Rishabh\\DIM(F)_FALL16\\Udemy\\train.csv')
titanic_df.head()
titanic_df.info()# in cabin we are missing some values there as only 204 non nulls
# everything else is much closer to 891

#some basic questions
# who were the passengers on the titanic

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# factor plot
sns.factorplot('Sex',kind='count', data =titanic_df) # column name and where data is coming from

sns.factorplot('Pclass', data=titanic_df)

# need to see children present 

def male_female_child(passenger):
    age,sex= passenger
    # taking passenger as an object
    # take age and sex of the passenger dataset
    if age <16:
        return 'Child'
    else:
        return sex

titanic_df['Person']= titanic_df[['Age','Sex']].apply(male_female_child,axis=1)     
# creating a new column person, takinf age and sex from titanic df and applying the data frame to the function on column axis
titanic_df.head()

sns.factorplot('Pclass',data= titanic_df, hue='Person')

titanic_df['Age'].hist(bins=70)
# Mean age 
titanic_df['Age'].mean()
titanic_df['Person'].value_counts()

# kernel density plots to know more about the passengers

fig=sns.FacetGrid(titanic_df,hue='Sex',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)
oldest_limit=titanic_df['Age'].max()
fig.set(xlim=(0,oldest_limit))
fig.add_legend()


fig=sns.FacetGrid(titanic_df,hue='Person',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)
oldest_limit=titanic_df['Age'].max()
fig.set(xlim=(0,oldest_limit))
fig.add_legend()


fig=sns.FacetGrid(titanic_df,hue='Pclass',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)
oldest_limit=titanic_df['Age'].max()
fig.set(xlim=(0,oldest_limit))
fig.add_legend()

deck= titanic_df['Cabin'].dropna()
deck.head()

#############
list=[]

for level in deck:
    list.append(level[0])
    
cabin_df=DataFrame(list)
cabin_df.columns=['cabin']
sns.factorplot(x='cabin',kind='count', data=cabin_df,palette='winter_d')

cabin_df=cabin_df[cabin_df['cabin'] !='T']
sns.factorplot(x='cabin',kind='count', data=cabin_df,palette='summer')

#where did the people come from
titanic_df.head()
sns.factorplot(x='Embarked',kind='count', data=titanic_df,palette='summer',hue='Pclass')

# Alone or with family
titanic_df.head()
titanic_df['Alone']=titanic_df.Parch+titanic_df.SibSp
titanic_df.Alone

titanic_df['Alone'].loc[titanic_df['Alone']>0]='With Family' # locaiton where Alone column of titanic dataframe is greater thnan 
titanic_df['Alone'].loc[titanic_df['Alone']==0]='Alone'

titanic_df.head()
sns.factorplot(x='Alone',kind='count',data=titanic_df,palette='Blues')

titanic_df['Survivor']=titanic_df.Survived.map({0:'No',1:'Yes'})
titanic_df.head()
sns.factorplot(x='Survivor',kind='count',data=titanic_df,palette='Blues')

# factors affecting the survival
sns.factorplot('Pclass','Survived',data=titanic_df,hue='Person',palette='winter_d')
#shows that children and females from any class had better survival chance
#than males. Being a male or being a 3rd class was definitely not favorable
#for survival

sns.lmplot('Age','Survived',data=titanic_df)
# General trend if we look a the linear line, the older the passenger was, the less likely they survived

sns.lmplot('Age','Survived',hue='Pclass',data=titanic_df)
# similar story in all the class. 

# making some age bins to make the plot more clear
generations=[10,20,40,60,80]
sns.lmplot('Age','Survived',hue='Pclass',data=titanic_df,palette='winter',x_bins=generations)
# higher standard deviation on older people in the first class. Older people
# in first class were tried to be saved


sns.lmplot('Age','Survived',hue='Sex',data=titanic_df,palette='winter',x_bins=generations)
# Wow interisting! 
# looks like older females had better chance of survival than older males

# effect of familySize on the Survival 
titanic_df['Family_Size']=titanic_df.Parch+titanic_df.SibSp
sns.factorplot(x='Family_Size',hue='Sex',kind='count',data=titanic_df,palette='Blues')

sns.lmplot('Family_Size','Survived',hue='Sex',data=titanic_df,palette='winter')

#########################
#revisitng the cabin once again

new_list=[]

for level in titanic_df['Cabin']:
    new_list.append(level[0])
    
cabin_df=DataFrame(list)
cabin_df.columns=['cabin']
sns.factorplot(x='cabin',kind='count', data=cabin_df,palette='winter_d')

cabin_df=cabin_df[cabin_df['cabin'] !='T']
sns.factorplot(x='cabin',kind='count', data=cabin_df,palette='summer')