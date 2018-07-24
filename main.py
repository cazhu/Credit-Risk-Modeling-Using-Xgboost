import itertools
from datetime import datetime
import folium
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from pandas import HDFStore
from pivottablejs import pivot_ui
from sklearn import datasets, linear_model, preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (auc, classification_report, confusion_matrix,
                             roc_curve)
from sklearn.utils import resample
from xgboost import XGBClassifier, plot_importance
plt.style.use('ggplot')


# parse data and create HDF5 dataset
f_data_index = {2017: [2]}
f_data = {}
LCDB = HDFStore('lcdb.h5')
df = {}
date_list = ['issue_d', 'earliest_cr_line', 'last_pymnt_d',
             'next_pymnt_d', 'last_credit_pull_d']
for year in f_data_index:
    for q in f_data_index[year]:
        key = 'Y' + str(year) + 'Q' + str(q)
        f_data[key] = 'input/LoanStats_securev1_{}.csv'.format(key[1:])
        if key in LCDB:
            df[key] = LCDB[key]
        else:
            df[key] = pd.read_csv(f_data[key], header=1,
                                  parse_dates=date_list, low_memory=False)[:-2]
            LCDB[key] = df[key]
df = pd.concat([x for x in df.values()])
for i in date_list:
    df[i] = df[i].dt.to_period('M')

# check columns and rows
print('Data shape: {}'.format(df.shape))

# check missing values
df.dropna(axis=0, how='all', inplace=True)
check_null = df.isnull().sum(axis=0).sort_values(ascending=False) / float(len(df))
df.drop(check_null[check_null > 0.6].index, axis=1, inplace=True)

# data cleaning
df = df[df['loan_status'] != 'Current']
df['loan_status'] = df['loan_status'].apply(
    lambda x: 1 if x == 'Fully Paid' else 0)

df['term'] = df['term'].str.split(' ').str[1].astype(float)
df['int_rate'] = df['int_rate'].str.split('%').str[0].astype(float)
df['int_rate'] = df.int_rate.astype(float) / 100
df['revol_util'] = df['revol_util'].str.split('%').str[0].astype(float)
df['emp_length'] = df['emp_length'].str.extract('(\d+)').astype(float)
df['grade'] = df['grade'].replace(
    {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1})
df['sub_grade'] = df['sub_grade'].str[1].astype(float)

# employment title
df['is_manager'] = df['emp_title'].fillna('None').apply(
    lambda x: 1 * ('manager' in x.lower()))
df['is_tech'] = df['emp_title'].fillna('None').apply(
    lambda x: 1 * ('tech' in x.lower()))
df['is_director'] = df['emp_title'].fillna('None').apply(
    lambda x: 1 * ('director' in x.lower()))

# log transformation
for i in ['fico_range_low', 'funded_amnt',
          'annual_inc', 'dti']:
    df[i + '(log)'] = np.log1p(df[i])

# numeric variables
num_list = [
    'is_manager',
    'is_tech',
    'is_director',
    'revol_util',
    'dti(log)',
    'emp_length',
    'fico_range_low(log)',
    'term',
    'int_rate',
    'funded_amnt(log)',
    'grade',
    'sub_grade',
    'annual_inc(log)']

# categorical variables
cate_list = ['addr_state',
             'purpose',
             'home_ownership']

# make pivot table
pivot_ui(df[['loan_status'] + [x if '(log)' not in x else x[:-5] for x in num_list + cate_list]],
         outfile_path="pivot_table/2017Q2LendingClub.html")

# make hist
plt.figure()
df[num_list + ['loan_status']
   ].hist(bins=50, figsize=(15, 15), edgecolor='white')
plt.savefig('figures/dist.png')
plt.gcf().clear()

for var_name in ['funded_amnt', 'annual_inc']:
    get_map(var_name, log=1, method='mean')

for var_name in ['loan_status', 'grade', 'emp_length']:
    get_map(var_name, log=0, method='mean')

# handling imbalanced data
df_majority = df[df['loan_status'] == 1]
df_minority = df[df['loan_status'] == 0]
n_majority = len(df_majority)
n_minority = len(df_minority)
df_majority_downsampled = resample(df_majority,
                                   replace=False,
                                   n_samples=n_minority,
                                   random_state=111)
df = pd.concat([df_majority_downsampled, df_minority], ignore_index=True)

# get features(X) and target (y)
features = df[num_list]
features = features.fillna(features.median())
for i in cate_list:
    df_dummy = pd.get_dummies(df[i])
    df_dummy = df_dummy.rename(
        columns={x: i + '_' + x for x in df_dummy.columns})
    features = features.join(df_dummy)

X = features
y = df['loan_status']

# check data shape
print('Feature data shape: {}'.format(features.shape))
print('Target variable shape: {}'.format(y.shape))

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=111)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    # Visualize confusion martix
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.grid(False)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 fontsize=10,
                 horizontalalignment="center",
                 color="red" if cm[i, j] > thresh else "black")

    plt.tight_layout(pad=5)
    plt.ylabel('True')
    plt.xlabel('Predicted')


def run_model(model_name='LogRegCV'):

    if model_name == 'LogRegCV':
        clf = linear_model.LogisticRegressionCV(cv=5, scoring='roc_auc')
    else:
        clf = XGBClassifier(objective='binary:logistic',
                            eval_metric='auc',
                            reg_alpha=2,
                            max_depth=3,
                            learning_rate=0.1,
                            n_estimators=100,
                            n_jobs=2,  # parallel threads
                            random_state=999,
                            subsample=0.7,
                            reg_lambda=2,
                            )

    # train model
    model = clf.fit(X_train, y_train)

    # check score
    print('Accuary: {}'.format(model.score(X_train, y_train)))

    # predict values
    y_pred = model.predict(X_test)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure(figsize=(5, 5))
    plot_confusion_matrix(cnf_matrix, classes=[0, 1],
                          title='Confusion matrix')
    plt.savefig('figures/confusion_{}.png'.format(model_name))
    # Plot normalized confusion matrix
    plt.figure(figsize=(5, 5))
    plot_confusion_matrix(cnf_matrix, classes=[0, 1], normalize=True,
                          title='Normalized confusion matrix')
    plt.savefig('figures/confusion_norm_{}.png'.format(model_name))

    # print coeff
    if model_name == 'LogRegCV':
        coeff = []
        for i, j in zip([x for x in features.columns], model.coef_.T):
            coeff.append([i, j[0]])
        coeff = pd.DataFrame(coeff, columns=['feature', 'coeff']).sort_values(
            'coeff', ascending=False)

        coeff.to_csv('logit_reg_coeff.csv')
    else:
        plt.figure(figsize=(100, 100))
        plot_importance(model, xlabel=None)
        plt.tight_layout(pad=2)
        plt.savefig('figures/xgboost_importance.png')

    # ROC curve
    y_pred_prob = clf.predict_proba(X_test)
    y_pred_prob = [x[1] for x in y_pred_prob]
    fpr, tpr, threshold = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5, 5))
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('figures/roc_curve_{}'.format(model_name))

    report = classification_report(y_test, y_pred)
    return report


for model_name in ['LogRegCV', 'xgboost']:
    print(model_name)
    print(run_model(model_name))
