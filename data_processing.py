import os

import numpy as np
import pandas as pd
from scipy import stats

DATASET_FOLDER = 'datasets'

NUM_FEATURES = ['AGE', 'CHILD_TOTAL', 'DEPENDANTS', 'PERSONAL_INCOME', 'LOAN_NUM_TOTAL', 'LOAN_NUM_CLOSED']


def get_joined_df(df1, df2, how):
    right_on = 'ID_CLIENT'
    df_joined = df1.merge(df2, left_on='ID', right_on=right_on, how=how)
    return df_joined.drop(right_on, axis=1)


def load_datasets():
    clients = pd.read_csv(os.path.join(DATASET_FOLDER, 'd_clients.csv'))
    close_loan = pd.read_csv(os.path.join(DATASET_FOLDER, 'd_close_loan.csv'))
    loan = pd.read_csv(os.path.join(DATASET_FOLDER, 'd_loan.csv'))
    salary = pd.read_csv(os.path.join(DATASET_FOLDER, 'd_salary.csv'))
    target = pd.read_csv(os.path.join(DATASET_FOLDER, 'd_target.csv'))
    return clients, close_loan, loan, salary, target


def aggregate_loan(loan, close_loan):
    """Добавляет в loan данные по закрытию займов"""
    loan = loan.merge(close_loan, on='ID_LOAN')
    return loan.groupby('ID_CLIENT').agg(LOAN_NUM_TOTAL=('ID_CLIENT', 'count'),
                                         LOAN_NUM_CLOSED=('CLOSED_FL', 'sum')).reset_index()


def compile_final_df(clients, target, salary, loan_agg):
    final_df = get_joined_df(clients, target, 'inner')
    for df in [salary, loan_agg]:
        final_df = get_joined_df(final_df, df, 'left')

    final_df.drop(['ID', 'REG_ADDRESS_PROVINCE', 'FACT_ADDRESS_PROVINCE', 'POSTAL_ADDRESS_PROVINCE',
                   'FAMILY_INCOME', 'FL_PRESENCE_FL', 'OWN_AUTO', 'EDUCATION', 'MARITAL_STATUS'], axis=1, inplace=True)
    final_df = final_df.set_index('AGREEMENT_RK')
    return final_df


def remove_outliers(df):
    no_outliers_boolean_series = pd.Series([True] * len(df), index=df.index)
    for feature_name in NUM_FEATURES:
        zscore_series = np.abs(stats.zscore(df[feature_name]))
        no_outliers_boolean_series &= zscore_series < 5
    return df[no_outliers_boolean_series]


def process_data():
    clients, close_loan, loan, salary, target = load_datasets()
    loan_agg = aggregate_loan(loan, close_loan)

    salary.drop_duplicates(inplace=True)

    final_df = compile_final_df(clients, target, salary, loan_agg)

    final_df = remove_outliers(final_df)

    final_df.to_pickle('df.pkl')


if __name__ == '__main__':
    process_data()
