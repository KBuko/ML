'''
Building a model to predict passenger satisfaction/dissatisfaction
with a flight based on flight data and passenger survey. Logistic regression
'''

import pandas as pd
from matplotlib import pyplot as plt
from sklearn import  linear_model, model_selection
import joblib

df_original = pd.read_csv('ml/data-logistic_regression_binary/airline_passenger_satisfaction.csv')

df = df_original.copy(deep=True)

df['Satisfaction'].replace(['Satisfied','Neutral or Dissatisfied'],[1,0], inplace=True)
df['Type_of_Travel'].replace(['Business','Personal'],['Business_travel','Personal_travel'], inplace=True)

#analysis of data on mean num values regarding satisfaction/dissatisfaction with the flight. Identification of key parameters
df_nums = df.drop(['ID','Gender','Customer_Type','Type_of_Travel','Class'], axis=1)
print(df_nums.groupby('Satisfaction').mean())

#Identification of the most important non-numerical parameters on the level of overall flight satisfaction
pd.crosstab(df_original.Gender, df_original.Satisfaction).plot.bar(rot=0)
plt.show()
pd.crosstab(df_original.Customer_Type, df_original.Satisfaction).plot.bar(rot=0)
plt.show()
pd.crosstab(df_original.Type_of_Travel, df_original.Satisfaction).plot.bar(rot=0)
plt.show()
pd.crosstab(df_original.Class, df_original.Satisfaction).plot.bar(rot=0)
plt.show()

#Preparation of the data set: introduction of dummy variables for significant non-numeric values,
#exclusion of insignificant parameters
c_type_dummy = pd.get_dummies(df.Customer_Type, dtype=int)
type_dummy = pd.get_dummies(df.Type_of_Travel, dtype=int)
class_dummy = pd.get_dummies(df.Class, dtype=int)

df = pd.concat([df, c_type_dummy, type_dummy, class_dummy], axis=1)
df.drop(['ID','Gender', 'Age', 'Customer_Type', 'Type_of_Travel', 'Class',
         'Departure and Arrival Time Convenience', 'Ease of Online Booking',
         'Check-in Service', 'Gate Location', 'On-board Service', 'Leg Room Service',
         'Cleanliness','Food and Drink', 'Food and Drink', 'In-flight Wifi Service',
         'Baggage Handling', 'Returning', 'Personal_travel', 'Economy Plus'], axis = 1, inplace=True)
df = df.fillna(0)

#Dataset splitting, model training
x = df.drop('Satisfaction', axis=1)
y = df.Satisfaction

x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y, test_size=0.2)
training = linear_model.LogisticRegression(max_iter=2000)

training.fit(x_train,y_train)

joblib.dump(training, 'ml/airline_passenger_satisfaction_model')
model = joblib.load('ml/airline_passenger_satisfaction_model')

print(model.predict(x_test))
print(model.score(x_test,y_test)) # ≈0.85%

