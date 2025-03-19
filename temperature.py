import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#%% Load data:
temperature_data_df = pd.read_csv('./data/seattle-weather.csv')
# temperature_data_df = pd.read_csv('./parcial_I/data/seattle-weather.csv')

date = pd.to_datetime(temperature_data_df['date'])
max_temperature = temperature_data_df['temp_max']

# Make an index column data with same length of date:
days= date.dt.day
idx_date = [i for i in range(len(days))]
print(f'len(idx_date): {len(idx_date)}')

# Scatter plot for date vs Max temperature
plt.scatter(idx_date, max_temperature, label='original data')
plt.title('No. of Days vs Maximum temperature (2012/01/01 - 2015/12/31)')
plt.xlabel('Number of Days')
plt.ylabel('Max Temp (°C)')
plt.legend()
plt.show()

#%% Machine Learning stage:
for n in range(6, 14, 2):
    model = LinearRegression()

    # Polynomial grade is given by n in the for loop:
    poly = PolynomialFeatures(degree=n)

    # Data transformation:
    x = np.array(idx_date).reshape(-1, 1)
    y = np.array(max_temperature).reshape(-1, 1)

    # Polynomial transformation:
    x_poly = poly.fit_transform(x)

    # MODEL TRAINING:
    model.fit(x_poly, y)

    # model coefficients:
    a = model.coef_[0]
    print(f'Model coefficients: {a}')

    # Intercept:
    intercept = model.intercept_[0]
    print(f'Intercept: {intercept}')

    #%% Model predictions:
    y_pred = model.predict(x_poly)

    #%% Calculating Scores:
    score = model.score(x_poly, y)
    print(f'Score: {round(score*100, 2)}%')

    #%% Figures:
    # Scatter plot for date vs Max temperature
    plt.scatter(idx_date, max_temperature, label='original data')
    plt.title('No. of Days vs Maximum temperature (2012/01/01 - 2015/12/31)')
    plt.xlabel('Number of Days')
    plt.ylabel('Max Temp (°C)')
    plt.legend()

    plt.plot(x, y_pred, '-r', label='model (n=' + str(n) + ')')
    plt.legend()
    plt.show()

    #%% Model Testing:
    x_test = [[50]]
    x_test_poly = poly.fit_transform(x_test)
    y_pred = model.predict(x_test_poly)
    print(f'Testing Result = {y_pred[0][0]}')
