from pathlib import Path
import datetime as dt

import pandas as pd  # import .xlsx file
import numpy as np  # perform any transformation in the dataset
import matplotlib.pyplot as plt  # plot the graphs and figures
from sklearn import linear_model  # apply linear model
from sklearn.preprocessing import PolynomialFeatures  # apply quadratic model
import statsmodels.api as sm  # apply Baxter-King and Hodrick-Prescott filters
import quantecon as qe  # apply Hamilton Filter


def calc_trend(df, var_name, method):
    data = df.dropna(how='any', subset=var_name)
    X = np.array(data['DATE']).reshape(-1, 1)
    y = data[var_name]

    reg = linear_model.LinearRegression()
    predicted_values = 0

    if method == 'linear':
        reg.fit(X, y)
        predicted_values = reg.predict(X)

    if method == 'quadratic':
        poly = PolynomialFeatures(2, include_bias=False)
        features = poly.fit_transform(X, y)
        reg.fit(features, y)
        predicted_values = reg.predict(features)

    trend = pd.Series(data=predicted_values, index=data.index,)

    return trend


def calc_cycle(df, var_name, method):
    data = df[var_name].dropna()

    if method in ['linear', 'quadratic']:
        trend = calc_trend(df, var_name, method)
        cycle = (df[var_name] - trend) / trend * 100

    if method == 'BK':
        cycle = sm.tsa.filters.bkfilter(data) * 100

    if method == 'HP':
        cycle, trend = sm.tsa.filters.hpfilter(data.values, lamb=1600)
        cycle *= 100

    if method == 'Hamilton':
        cycle_h, trend_h = qe.hamilton_filter(data, 8, 4)
        cycle = pd.Series(data=cycle_h, index=data.index).dropna() * 100

    return cycle


def plot_graphs(df, variables, method):
    for var in variables:
        data = df.dropna(how='any', subset=['GDP [Q]', var])

        graph_df = pd.DataFrame()
        graph_df['GDP [Q]'] = calc_cycle(data, 'GDP [Q]', method)
        graph_df[var] = calc_cycle(data, var, method)

        figure, axis = plt.subplots(1, 1)

        axis.plot(graph_df, label=graph_df.columns.values)
        axis.legend(loc='lower left')
        #axis.set_yticks(np.arange(0, 3.8, 0.1))
        axis.grid(color='k', linestyle='--', linewidth=0.2)
        axis.set_xlabel('Date')
        axis.set_ylabel('Deviation from Trend (%)')

        figure = plt.gcf()
        figure.set_size_inches(32, 18)
        plt.savefig(method + '_' + var + '.png', bbox_inches='tight')
        plt.close(figure)


def calc_summary(df, variables, method):
    table_dict = {}

    for var in variables:
        data = df.dropna(how='any', subset=['GDP [Q]', var])

        cycle_df = pd.DataFrame()
        cycle_df['GDP [Q]'] = calc_cycle(data, 'GDP [Q]', method)
        cycle_df[var] = calc_cycle(data, var, method)
        cycle_df['lag'] = cycle_df[var].shift(1)
        corrs = cycle_df.corr()

        temp_dict = {
            'Standard Deviation': cycle_df[var].std(),
            'Relative Standard Deviation': cycle_df[var].std() / cycle_df['GDP [Q]'].std(),
            'First Order Auto-correlation': corrs[var]['lag'],
            'Contemporaneous Correlation with Output': corrs[var]['GDP [Q]']
        }
        table_dict[var] = temp_dict

    table = pd.DataFrame(table_dict)

    return table.transpose()


def analyse_facts_by_method(df, var, methods):
    table_dict = {}

    for method in methods:
        data = df.dropna(how='any', subset=['GDP [Q]', var])

        cycle_df = pd.DataFrame()
        cycle_df['GDP [Q]'] = calc_cycle(data, 'GDP [Q]', method)
        cycle_df[var] = calc_cycle(data, var, method)
        cycle_df['lag'] = cycle_df[var].shift(1)
        corrs = cycle_df.corr()

        temp_dict = {
            'Standard Deviation': cycle_df[var].std(),
            'Relative Standard Deviation': cycle_df[var].std() / cycle_df['GDP [Q]'].std(),
            'First Order Auto-correlation': corrs[var]['lag'],
            'Contemporaneous Correlation with Output': corrs[var]['GDP [Q]']
        }
        table_dict[method] = temp_dict

    table = pd.DataFrame(table_dict)

    return table.transpose()


data_file = Path(__file__).parent / 'data' / 'Brazilian Data (Adv Macro, Insper MPE).xlsx'
xl = pd.ExcelFile(data_file)
data_df = xl.parse()
data_df.set_index('DATE', inplace=True, drop=False)
data_df['DATE'] = data_df['DATE'].map(dt.datetime.toordinal)

methods = [
    'linear',
    'quadratic',
    'BK',
    'HP',
    'Hamilton'
]

variables = [
    'Consumption of Non-Durables',
    'Consumption of Durables',
    'Investment',
    'Government Expenditures',
    'Total Hours Worked',
    'Capital',
    'Capital Utilization',
    'Labor Productivity',
    'Hours per Worker',
    'Employment',
    'Minimum Real Wages'
]
for method in methods:
    plot_graphs(data_df, variables, method)
    table = calc_summary(data_df, variables, method)
    print(method)


for variable in variables:
    table_by_method = analyse_facts_by_method(data_df, variable, methods)
    print(variable)
