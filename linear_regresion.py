'''
Predicting the effect of population size in a country on the unemployment rate.
The solution is presented using linear regression algorithms
with preliminary preparation of dataframes taken from several files.
The data are obtained from the source: https://www.worldbank.org/
'''

import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model

def prepare_dataset(file, country, start_year, end_year):
    df = pd.read_csv(file, skiprows=4, sep=',')
    region = df[df['Country Name'] == country]
    start_index = region.columns.get_loc(start_year)
    end_index = region.columns.get_loc(end_year)+1
    region = region.iloc[:, start_index:end_index]
    region = region.T
    return region

population = prepare_dataset('ml/population_total.csv', 'Japan', '1993','2022')
unemployment = prepare_dataset('ml/unemployment_total.csv', 'Japan', '1993','2022')
result = pd.concat([population, unemployment], axis=1)
result.columns = ['population', 'unemployment']

plt.scatter(result.population, result.unemployment, marker='o', color='green')
plt.xlabel('Population, total')
plt.ylabel('Unemployment, total (% of total labor force)')

training = linear_model.LinearRegression()
training.fit(result[['population']],result.unemployment)
plt.plot(result.population, training.predict(result[['population']]), color='blue')

plt.show()

print(training.predict([[123000000]]))

