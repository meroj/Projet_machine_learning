import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#--------------------------------------------------

df = pd.read_csv('food_coded.csv',index_col=0)
print("---dataframe shape : ", df.shape)
print("---dataframe head---")
print(df.head(3))

#--------------------------------------------------

#Colonne à droper
to_drop =['calories_chicken', 'sports', 'drink', 'employment', 'ideal_diet_coded', 'life_rewarding', 'veggies_day', 'vitamins', 'calories_scone', 'type_sports', 'comfort_food', 'cuisine', 'soup', 'comfort_food_reasons', 'comfort_food_reasons_coded', 'comfort_food_reasons_coded.1', 'diet_current', 'eating_changes', 'eating_changes_coded', 'eating_changes_coded1', 'ethnic_food', 'father_education', 'father_profession', 'fav_cuisine', 'fav_cuisine_coded', 'fav_food', 'food_childhood', 'fav_food', 'fruit_day', 'grade_level', 'greek_food', 'healthy_meal', 'ideal_diet', 'meals_dinner_friend', 'mother_education', 'mother_profession', 'nutritional_check', 'pay_meal_out', 'mother_education', 'mother_education', 'calories_day','tortilla_calories', 'turkey_calories','waffle_calories','thai_food', 'persian_food', 'italian_food', 'indian_food']
df.drop(to_drop, axis = 1, inplace = True)

print("---dataframe shape : ", df.shape)

#--------------------------------------------------

#Remplir les valeurs manquantes
df["cook"].fillna(3,inplace=True) #RAREMENT
df["on_off_campus"].fillna(3,inplace=True) #CHEZ LES PARENTS
df["marital_status"].fillna(1,inplace=True) #SEUL
df["income"].fillna(1,inplace=True) #LESS THAN 15K$
df["self_perception_weight"].fillna(3,inplace=True) #just right
df["exercise"].fillna(4,inplace=True) #RAREMENT
df["weight"].fillna(170,inplace=True) #170LBS
print("---check if all empty value are ok---")
print(df.isnull().sum())

#--------------------------------------------------

#Change les phrases en valeur pour être modifié en int par la suite
for index in range (1, 125, 1):
    if df["weight"][index] == "I'm not answering this. " :
      df["weight"][index] = "200"
    if df["weight"][index] == "Not sure, 240" :
      df["weight"][index] = "240"
    if df["weight"][index] == "144 lbs" :
      df["weight"][index] = "144"
df = df.astype(int)

print("---dataframe head---")
print(df.head(3))

#--------------------------------------------------

y = df.weight
X = df.drop('weight', axis=1, inplace=False)
print("---X shape : ", X.shape)
print("---y shape : ", y.shape)
y.hist(bins=125)

#--------------------------------------------------
print("REG---------------------")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=48)
print("train set shape: ", X_train.shape, y_train.shape)
print("test set shape: ", X_test.shape, y_test.shape)

reg = LinearRegression()
reg.fit(X_train, y_train)
train_score = reg.score(X_train, y_train)
test_score=reg.score(X_test, y_test)
print  ('train score =', train_score)
print  ('test score = {}'.format(test_score))

index_to_predict = 100
print("Value to predict is ",y[index_to_predict])

print(reg.predict(np.array(X.iloc[index_to_predict]).reshape(1,-1))[0])

#--------------------------------------------------
print("CLF---------------------")

print(X.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=48)
print("train set shape: ", X_train.shape, y_train.shape)
print("test set shape: ", X_test.shape, y_test.shape)

clf = LogisticRegression(max_iter=3000) 
clf.fit(X_train, y_train)
train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)
print ('train accuracy =', train_score)
print ('test accuracy =', test_score)

index_to_predict = 100
print("Value to predict is ",y[index_to_predict])

print(clf.predict(np.array(X.iloc[index_to_predict]).reshape(1,-1))[0])


