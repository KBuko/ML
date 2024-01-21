'''
The purpose of the program is to predict the size of an employee's annual bonus
based on his position and ready salary.
'''

import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model, model_selection
import joblib

df = pd.read_excel('ml/data-dummy_v/Employee Sample Data.xlsx')
df['Bonus'] *= 100

#using graphs, the relationship between the source data was discovered
'''
plt.scatter(df.Bonus, df.Annual_Salary, marker='o', color='red')
plt.xlabel('Bonus (%)')
plt.ylabel('Annual salary ($)')
plt.show()

plt.scatter(df.Bonus, df.Job_Title, marker='o', color='red')
plt.xlabel('Bonus (%)')
plt.ylabel('Job title')
plt.show()
'''

dummies = pd.get_dummies(df.Job_Title, dtype=int)

joined_df = pd.concat([df,dummies], axis=1)
final_df = joined_df.drop(['Job_Title','Hire_Date', 'Test Engineer'], axis=1)

x = final_df.drop('Bonus', axis = 1)
y = final_df['Bonus']

#splitting the initial dataset into training and test datasets
x_train, x_test, y_train, y_test = (
    model_selection.train_test_split(x,y, test_size=0.1, random_state=0))

training = linear_model.LinearRegression()
training.fit(x_train,y_train)

joblib.dump(training, 'ml/bonus_model')
model = joblib.load('ml/bonus_model')

#demonstrating the quality of the model’s performance by comparing training datasets
plt.scatter(x_train.Annual_Salary, y_train,color='green', alpha=0.5)
plt.scatter(x_test.Annual_Salary, y_test, color='red')
plt.xlabel('Annual_Salary ($)')
plt.ylabel('Annual Bonus(%)')
plt.show()

print(model.score(x_test,y_test))

