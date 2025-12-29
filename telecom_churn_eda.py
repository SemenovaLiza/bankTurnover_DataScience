import os
import glob
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import math
from sklearn.preprocessing import FunctionTransformer
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Loading data
os.chdir('final_provider')
glob.glob('*.csv')
#phone_df = pd.read_csv('final_provider/phone.csv')
dfs = dict(map(lambda f:(os.path.basename(f).replace('.csv', ''), pd.read_csv(f)), glob.glob('*.csv')))


def run_function(dfs, function, *fun_args):
    for name, features in dfs.items():
        print(50*"-"+name.upper()+50*"-")
        print(function(dfs[name], *fun_args))


def info_(df, sample_n):
    df_sample = df.sample(sample_n)
    df_info = df.info()
    print(f"{sample_n} SIZED SAMPLE")
    print()
    print(df_sample)
    print()
    print(df_info)


def study_null(df, null_only=False):
    null_sum = df.isna().sum()
    df_null = pd.concat([df.dtypes, null_sum, null_sum / len(df) * 100], axis=1)
    df_null.columns = ["dtypes", "null_count", "missing_%"]
    if null_only:
        df_null = df_null[df_null['null_count'] > 0]
    df_null.sort_values(by='missing_%', ascending=False)
    return df_null


def describe_full(df):
    df_describe = df.describe().T
    df_numeric = df._get_numeric_data()
    df_describe['dtypes'] = df_numeric.dtypes
    df_describe['missing_%'] = df_numeric.isna().sum() / len(df_numeric) * 100
    cardinality = df_numeric.apply(pd.Series.nunique)
    df_describe['Cardinality'] = cardinality
    df_describe['Skew'] = df_numeric.skew(axis=0, skipna=False)
    return df_describe


run_function(dfs, info_, 5)
run_function(dfs, study_null)
run_function(dfs, describe_full)

dfs['contract']['MonthlyCharges'].hist()
dfs['internet']['StreamingMovies'].isna().sum()

telecom_df = (dfs['personal']
             .merge(dfs['contract'], how='left', on='customerID')
             .merge(dfs['internet'], how='left', on='customerID')
             .merge(dfs['phone'], how='left', on='customerID')
)

print(telecom_df)

print(study_null(telecom_df))

def na_to_no(df, *features):
    for feature in [*features]:
        df[feature] = df[feature].fillna("No")

na_to_no(telecom_df, 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'MultipleLines')
telecom_df['InternetService'] = telecom_df['InternetService'].fillna('No service')

print(study_null(telecom_df))

telecom_df.iloc[753] # just to compare with preprocessed result
def convert_type(df):
    for col in df.select_dtypes('object').columns:
        if sorted(df[col].dropna().unique())==['No', 'Yes']:
            # df[col] = (df[col] == 'Yes').astype('int')
            df[col] = df[col].map({'No': 0, 'Yes': 1})
        elif sorted(df[col].dropna().unique())==['False', 'True']:
            df[col] = df[col].map({'False': 0, 'True': 1})
    return df

telecom_df = convert_type(telecom_df)
print(telecom_df)

print(telecom_df.iloc[753])
telecom_df['gender'] = telecom_df['gender'].map({'Male': 1, 'Female': 0})
print(telecom_df.iloc[753])
print(telecom_df)
# row 753 has empty TotalCharges - this feature need to be checked
telecom_df[telecom_df['TotalCharges'] == " "]
# converts strings to floats and handles conversion erro by converting 'non-numeric' strings to Nan. Then fills nulls with 0.0
telecom_df['TotalCharges'] = pd.to_numeric(telecom_df['TotalCharges'], errors='coerce').fillna(0.0)
print(telecom_df.info())
telecom_df['churn'] = (telecom_df['EndDate'] != 'No').astype('int')
print(telecom_df)
final_df = telecom_df.drop(['BeginDate', 'EndDate'], axis=1)
sns.countplot(x='churn', data=final_df)
plt.show()
