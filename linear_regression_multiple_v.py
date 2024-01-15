'''
Predicting the combined effect of a country's population growth percentage
and unemployment rate on a country's inflation rate.
The solution is presented using linear regression
with preliminary preparation of data sets.
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
    median = round(float(region.median()), 6)
    region = region.fillna(median)
    return region

inflation = prepare_dataset('ml/data-linear_regression_multiple_v/inflation.csv', 'Japan', '1990','2022')
unemployment = prepare_dataset('ml/data-linear_regression_multiple_v/unemployment_total.csv', 'Japan', '1990','2022')
p_growth = prepare_dataset('ml/data-linear_regression_multiple_v/p_growth.csv', 'Japan', '1990','2022')

result = pd.concat([inflation, unemployment, p_growth], axis=1)
result.columns = ['inflation', 'unemployment', 'p_growth']

#The obtained graph proves the direct relationship between the inflation factor and the unemployment rate (Phillips curve)
plt.scatter(result.unemployment,result.inflation, marker='x', color='red')
plt.xlabel('Unemployment, total (% of total labor force)')
plt.ylabel('Inflation, consumer prices (annual %)')
plt.show()

#Training the model on the generated dataset
training = linear_model.LinearRegression()
training.fit(result[['unemployment', 'p_growth']],result.inflation)

#model testing
print(training.predict([[0, 1]]))

