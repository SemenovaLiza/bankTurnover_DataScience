import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv('Churn.csv')
print(df.head(8))
df.drop(['RowNumber', 'Surname', 'CustomerId'], axis=1, inplace=True)
print(df.info())
print(df.describe())
df.columns = df.columns.str.lower()
print(df.columns)
print(df.duplicated().sum())
print(df.isna().sum())
# proportion of people left the bank
print(sum(df['exited']/len(df)))
print(df)
# proporting of bank clients
print(1 - sum(df['exited']/len(df)))
# converting categorical features into num
gender_one_hot = pd.get_dummies(df['gender'], drop_first=True)
country_one_hot = pd.get_dummies(df['geography'], drop_first=True)
df.drop(['gender', 'geography'], axis=1, inplace=True)
df_one_hot = pd.concat([df, gender_one_hot, country_one_hot], axis=1)
print(df_one_hot.head(8))
# separate objects and target variable
features = df.drop("exited", axis=1)
target = df["exited"]
features.fillna(-1, inplace=True)
features['tenure'] = features['tenure'].astype('object')
tenure_one_hot = pd.get_dummies(df["tenure"], drop_first=True)
df = df.drop('tenure',axis = 1)
df = df.join(tenure_one_hot)
print(features)
# one dataset - split into 3 parts. 60% training, 20% validation, 20% testing
# separate test dataset
x, features_test, y, target_test = train_test_split(features, target, test_size=0.2, train_size=0.8)
# separate validation and training datasets
features_train, features_valid, target_train, target_valid = train_test_split(x, y, test_size=0.25, train_size=0.75)
# scalling
scaler = StandardScaler()
scaler.fit(features_train) # learning from dataset
features_train_scaled = scaler.transform(features_train)
features_valid_scaled = scaler.transform(features_valid)
features_test_scaled = scaler.transform(features_test)
def nptodf(data, ind, column):
    frame = pd.DataFrame(data, index=ind.index, columns=column.columns)
    return frame
print(features_train_scaled)
features_train_scaled = nptodf(features_train_scaled, target_train, features)
features_valid_scaled = nptodf(features_valid_scaled, target_valid, features)
features_test_scaled = nptodf(features_test_scaled, target_test, features)
print(features_train_scaled)
def check(data):
    res = len(data)/len(df)
    print(f'Proportion procent {res}, Number of rows and columns: {data.shape}')
data_list = [features_train_scaled, features_valid_scaled, target_train, target_valid, features_test_scaled, target_test]
for i in data_list:
    check(i)

#  Decision Tree
best_model_dt = None
best_model_dt_depth = 0
best_model_dt_leaf = 0
best_result_dt = 0 #  highest accuracy so far

for depth in range(1, 13):
    #  tuning hyperparameters - fine tuning

    for i in range(1, 50):
        model_dt = DecisionTreeClassifier(random_state=12345, max_depth=depth, min_samples_leaf=i)
        model_dt.fit(features_train_scaled, target_train)
        predictions_dt = model_dt.predict(features_valid_scaled)
        result_dt = accuracy_score(target_valid, predictions_dt)
        if result_dt > best_result_dt:
            best_model_dt = model_dt
            best_model_dt_depth = depth
            best_model_dt_leaf = i
            best_result_dt = result_dt
result_f1_dt = f1_score(target_valid, predictions_dt)
print(f'F1 for Decision Tree: {result_f1_dt}')
print(f'Best result: {best_result_dt}')
print(f'Hyperparameters: max_depth: {best_model_dt_depth}, min_samples_leaf: {best_model_dt_leaf}')
