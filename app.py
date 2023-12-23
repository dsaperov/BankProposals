from data_processing import process_data, NUM_FEATURES

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

LABELS_FOR_CAT_FEATURES = {
    'GENDER': {1: 'Мужчины', 0: 'Женщины'},
    'SOCSTATUS_WORK_FL': {1: 'Работает', 0: 'Не работает'},
    'SOCSTATUS_PENS_FL': {1: 'Пенсионеры', 0: 'Не пенсионеры'},
    'TARGET': {1: 'Отклик был', 0: 'Отклика не было'}
}

HEADER_4_PREFIX = '#### '
HR = '___'


def display_histogram(df, feature_name):
    plt.figure()
    sns.histplot(df[feature_name], bins=30, kde=True, edgecolor='black')
    st.pyplot(plt)


def label_cat_features(df):
    """Заменяет числовые значения категориальных признаков на человекочитаемые."""
    df_labeled = df.copy()
    for feature_name, labels in LABELS_FOR_CAT_FEATURES.items():
        df_labeled[feature_name] = df_labeled[feature_name].replace(labels)
    return df_labeled


def plot_cat_features_distr(df):
    df_melted = pd.melt(df, value_vars=LABELS_FOR_CAT_FEATURES.keys())

    fig, ax = plt.subplots(figsize=(10,6))

    sns.countplot(data=df_melted, x='variable', hue='value', ax=ax)

    ax.set(xlabel='Features', ylabel='Count')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    st.pyplot(fig)


def plot_num_feature_distr(df, label, feature, conclusion, is_last_feature):
    st.markdown(HEADER_4_PREFIX + label)
    fig, ax = plt.subplots()
    sns.histplot(df[feature], kde=True, ax=ax)
    st.pyplot(fig)
    if conclusion:
        st.write(conclusion)
    if not is_last_feature:
        st.markdown(HR)


def build_dependency_graph(feature1_type, df, feature1, is_last_feature, feature2='TARGET'):
    st.markdown(HEADER_4_PREFIX + f'`{feature2}` & `{feature1}`')

    if feature1_type == 'num':
        fig, axs = plt.subplots(2)

        # Построить диаграмму размаха
        sns.boxplot(x=feature2, y=feature1, data=df, hue=feature2, palette='Set2', legend=False, ax=axs[0])

        # Построить диаграмму пересечения распределений
        feature2_unique_values = df[feature2].unique()
        unique_val_1, unique_val_2 = feature2_unique_values[0], feature2_unique_values[1]
        df1, df2 = df[df[feature2] == unique_val_1], df[df[feature2] == unique_val_2]

        pars = dict(x=feature1, kde=True, alpha=0.3, edgecolor='grey', stat='density')
        sns.histplot(data=df1, color='blue', label=str(unique_val_1), **pars)
        sns.histplot(data=df2, color='orange', label=str(unique_val_2), **pars)

        axs[1].legend(loc='upper right')
        axs[1].yaxis.set_visible(False)
        plt.subplots_adjust(hspace=0.4)

    # Построить сгруппированную столбчатую диаграмму
    else:
        fig, ax = plt.subplots()
        pd.crosstab(df[feature1], df[feature2]).plot(kind='bar', xlabel=feature1, ylabel=feature2, ax=ax)
        plt.xticks(rotation=0)

    st.pyplot(fig)

    if not is_last_feature:
        st.markdown(HR)


st.set_page_config(page_title='Отклик на предложение банка')

if not os.path.exists('df.pkl'):
    process_data()

df = st.cache_data(pd.read_pickle)('df.pkl')

st.title('Вероятность отклика на предложение банка')
st.write('Исследуем склонность клиента выдать положительный или отрицательный отклик на предложение банка.')

st.header('Датасет')
st.dataframe(df.head())

st.header('Распределение признаков')
st.subheader('Категориальные признаки', divider='rainbow')
df_labeled = label_cat_features(df)
plot_cat_features_distr(df_labeled)
st.write('Мужчин почти в 2 раза больше, чем женщин. Большинство клиентов работают. '
         'Большинство клиентов не находятся на пенсии. Отсутствие отклика превалирует.')

st.subheader('Числовые признаки', divider='rainbow')
labels = ['Возраст', 'Количество детей', 'Количество иждивенцев', 'Личный доход', 'Количество ссуд', 'Количество погашенных ссуд']
conclusions = ['Самый распространённый возраст - от 25 до 35 лет.', 'Большинство клиентов имеет от 0 до 2 детей.',
               'Большинство клиентов не имеет иждивенцев.', 'Доход большинства не превышает 15 000 рублей.', '', '']
num_features_count = len(NUM_FEATURES)
for i, label, feature, conclusion in zip(range(1, num_features_count + 1), labels, NUM_FEATURES, conclusions) :
    is_last_feature = i == num_features_count
    plot_num_feature_distr(df, label, feature, conclusion, is_last_feature)

st.header('Тепловая карта')
plt.figure(figsize=(12,10))
sns.heatmap(df.corr().round(2), cmap="Blues", annot=True)
st.pyplot(plt)
st.markdown('''Наибольшая корреляция (>0.5) наблюдается между признаками:
- `AGE` и `SOCSTATUS_PENS_FL` (0.56)
- `CHILD_TOTAL` и `DEPENDANTS` (0.51)
- `LOAN_NUM_TOTAL`И `LOAN_NUM_CLOSED` (0.84)
            
**Целевая переменная** не имеет значимой корреляции ни с одним признаком.''')
st.header('Числовые характеристики')
st.write(df.describe())

st.header('Диаграммы рассеяния')
with st.spinner('Пожалуйста, подождите. Диаграммы формируются.'):
    df_numeric = df[NUM_FEATURES]
    g = sns.pairplot(df_numeric, kind='reg')
    st.pyplot(g)

st.header('Графики зависимости целевой переменной и признаков')
st.subheader('Числовые признаки', divider='rainbow')
num_features_count = len(NUM_FEATURES)
for i, feature in enumerate(NUM_FEATURES, start=1):
    is_last_feature = i == num_features_count
    build_dependency_graph('num', df_labeled, feature, is_last_feature)

st.subheader('Категориальные признаки', divider='rainbow')
cat_features_without_target = [feature for feature in LABELS_FOR_CAT_FEATURES.keys() if feature != 'TARGET']
cat_features_without_target_count = len(cat_features_without_target)
for i, feature in enumerate(cat_features_without_target, start=1):
    is_last_feature = i == cat_features_without_target_count
    build_dependency_graph('cat', df_labeled, feature, is_last_feature)
