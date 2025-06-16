## %% [markdown]
# # requirements
import math
import itertools
import time
from datetime import timedelta, datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# root path
import sys
import os

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Now you can import from src
from src.app.utils import open_json

# %%
df = pd.read_json('../data/job_offers.json')
df
# %%
df.info()
# %%
df.describe().T
# %%
df.columns
# %%
df_types = pd.DataFrame(df.dtypes)
object_features = df_types[df_types[0] == 'object'].index.to_list()
object_features = [col for col in object_features if col != 'skills']
object_features
# %%
float_features = df_types[df_types[0] == 'float64'].index.to_list()
float_features
# %%
bool_features = df_types[df_types[0] == 'bool'].index.to_list()
bool_features
# %%
df[object_features].describe(include='all').T
# %%
for col in object_features:
    print(f'------>{col} : {df[col].unique()}')
    print(f'------>{col} : {df[col].value_counts(dropna=False)}')
# %%
df.shape
# %% [markdown]
# # null values visualization
# %%
sns.heatmap(df.isnull(), cbar=False)
plt.title("missing values")
plt.xlabel('Variable')
plt.ylabel('Fila')
# %%
# # categorical variables
#%%
plt.figure(figsize=(5,7))
a = pd.DataFrame(df['industries'].value_counts()[:11])
y = np.array(list(a.index))
x = np.array(list(a['count']))
sns.barplot(x=x, y=y, palette="rocket", hue=y, legend=False)
plt.grid(True)
plt.title("'industries'")
# %%
plt.figure(figsize=(5,7))
a = pd.DataFrame(df['company'].value_counts()[:11])
y = np.array(list(a.index))
x = np.array(list(a['count']))
sns.barplot(x=x, y=y, palette="rocket", hue=y, legend=False)
plt.grid(True)
plt.title("'company'")
#%%
plt.figure(figsize=(5,7))
a = pd.DataFrame(df['location'].value_counts()[:11])
y = np.array(list(a.index))
x = np.array(list(a['count']))
sns.barplot(x=x, y=y, palette="rocket", hue=y, legend=False)
plt.grid(True)
plt.title("'location'")
# %%
plt.figure(figsize=(5,7))
a = pd.DataFrame(df['work_modality_english'].value_counts()[:11])
y = np.array(list(a.index))
x = np.array(list(a['count']))
sns.barplot(x=x, y=y, palette="rocket", hue=y, legend=False)
plt.grid(True)
plt.title("'work_modality_english'")
# %%
plt.figure(figsize=(5,7))
a = pd.DataFrame(df['seniority'].value_counts()[:11])
y = np.array(list(a.index))
x = np.array(list(a['count']))
sns.barplot(x=x, y=y, palette="rocket", hue=y, legend=False)
plt.grid(True)
plt.title("'seniority'")
# %%
plt.figure(figsize=(5,7))
a = pd.DataFrame(df['publication_date'].value_counts()[:11])
y = np.array(list(a.index))
x = np.array(list(a['count']))
sns.barplot(x=x, y=y, palette="rocket", hue=y, legend=False)
plt.grid(True)
plt.title("'publication_date'")
# %%
print(f'job offers in the last 3 days: {df[df["publication_date"] > (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")].shape[0]}')
df[df['publication_date'] > (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')][['skills']].head(60)
# %%
print(f'job offers of 4 days ago: {df[df["publication_date"] == (datetime.now() - timedelta(days=4)).strftime("%Y-%m-%d")].shape[0]}')
df[df['publication_date'] == (datetime.now() - timedelta(days=4)).strftime('%Y-%m-%d')][['skills']].head(60)

# %%
