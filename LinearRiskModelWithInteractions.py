import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import load_data

from public_tests import *

X = pd.read_csv('data/X_data.csv',index_col=0)
y_df = pd.read_csv('data/y_data.csv',index_col=0)
y = y_df['y']

X.head()
y.head()

from sklearn.model_selection import train_test_split
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=0)
for col in X.columns:
    X_train_raw.loc[:, col].hist()
    plt.title(col)
    plt.show()
    
from scipy.stats import norm
data = np.random.normal(50,12, 5000)
fitting_params = norm.fit(data)
norm_dist_fitted = norm(*fitting_params)
t = np.linspace(0,100, 100)
plt.hist(data, bins=60, density=True)
plt.plot(t, norm_dist_fitted.pdf(t))
plt.title('Example of Normally Distributed Data')
plt.show()
for col in X_train_raw.columns:
    np.log(X_train_raw.loc[:, col]).hist()
    plt.title(col)
    plt.show()


def make_standard_normal(df_train, df_test):
    """
    In order to make the data closer to a normal distribution, take log
    transforms to reduce the skew.
    Then standardize the distribution with a mean of zero and standard deviation of 1. 
  
    Args:
      df_train (dataframe): unnormalized training data.
      df_test (dataframe): unnormalized test data.
  
    Returns:
      df_train_normalized (dateframe): normalized training data.
      df_test_normalized (dataframe): normalized test data.
    """
    df_train_unskewed = np.log(df_train)
    df_test_unskewed = np.log(df_test)
    mean = df_train_unskewed.mean(axis=0)
    stdev = df_train_unskewed.std(axis=0)
    df_train_standardized = (df_train_unskewed-mean)/stdev
    return df_train_standardized, df_test_standardized
    
X_train, X_test = make_standard_normal(X_train_raw, X_test_raw)
for col in X_train.columns:
    X_train[col].hist()
    plt.title(col)
    plt.show()
    

def lr_model(X_train, y_train):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

model_X = lr_model(X_train, y_train)

def cindex(y_true, scores):
    '''
    Input:
    y_true (np.array): a 1-D array of true binary outcomes (values of zero or one)
        0: patient does not get the disease
        1: patient does get the disease
    scores (np.array): a 1-D array of corresponding risk scores output by the model

    Output:
    c_index (float): (concordant pairs + 0.5*ties) / number of permissible pairs
    '''
    n = len(y_true)
    assert len(scores) == n
    concordant = 0
    permissible = 0
    ties = 0
    for i in range(n):
        for j in range(i+1, n):
            if y_true[i]!=y_true[j]:
                permissible+=1
                if scores[i]==scores[j]:
                    ties+=1
                    continue
                if y_true[i] == 0 and y_true[j] == 1:
                    if scores[i]<scores[j]:
                        concordant+=1
                if y_true[i]==1 and y_true[j] == 0:
                    if scores[i]>scores[j]:
                        concordant+=1
    c_index = (concordant+0.5*ties)/permissible
    return c_inde

scores = model_X.predict_proba(X_test)[:, 1]
c_index_X_test = cindex(y_test.values, scores)
print(f"c-index on test set is {c_index_X_test:.4f}")

coeffs = pd.DataFrame(data = model_X.coef_, columns = X_train.columns)
coeffs.T.plot.bar(legend=None);

def add_interactions(X):
    """
    Add interaction terms between columns to dataframe.
    Args:
    X (dataframe): Original data
    Returns:
    X_int (dataframe): Original data with interaction terms appended. 
    """
    features = X.columns
    m = len(features)
    X_int = X.copy(deep=True)
    for i in range(m):
        feature_i_name = features[i]
        feature_i_data = X.loc[:, feature_i_name]
        for j in range(i+1, m):
            feature_j_name = features[j]
            feature_j_data = X.loc[:, feature_j_name]
            feature_i_j_name = feature_i_name+"_x_"+feature_j_name
            X_int[feature_i_j_name] = feature_i_data*feature_j_data
    return X_int

X_train_int = add_interactions(X_train)
X_test_int = add_interactions(X_test)


model_X_int = lr_model(X_train_int, y_train)
scores_X = model_X.predict_proba(X_test)[:, 1]
c_index_X_int_test = cindex(y_test.values, scores_X)
scores_X_int = model_X_int.predict_proba(X_test_int)[:, 1]
c_index_X_int_test = cindex(y_test.values, scores_X_int)
print(f"c-index on test set without interactions is {c_index_X_test:.4f}")
print(f"c-index on test set with interactions is {c_index_X_int_test:.4f}")
int_coeffs = pd.DataFrame(data = model_X_int.coef_, columns = X_train_int.columns)
int_coeffs.T.plot.bar();
index = index = 3432
case = X_train_int.iloc[index, :]
print(case)
new_case = case.copy(deep=True)
new_case.loc["Age_x_Cholesterol"] = 0
new_case
print(f"Output with interaction: \t{model_X_int.predict_proba([case.values])[:, 1][0]:.4f}")
print(f"Output without interaction: \t{model_X_int.predict_proba([new_case.values])[:, 1][0]:.4f}")

