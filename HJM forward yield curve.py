# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 21:19:40 2022

@author: sigma
"""

# Import libraries
import numpy as np
import pandas as pd

# Plot settings
#pip install cufflinks
import cufflinks as cf
cf.set_config_file(offline=True)

# scikit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

data = pd.read_csv(r'D:\Python for Quant Model\hjm-pca.txt', index_col=0, sep ='\t')
data.head()
data.shape

# Plot curve
data.iloc[0].iplot(title = 'Representation of a Yield Curve')


#Volatility by taking first difference
diff_ = data.diff(-1)
diff_.dropna(inplace=True)
diff_.tail()
#Derive volatility
#The drift of forward rate is fully determined by volatility of forward rate dynamics

vol = np.std(diff_, axis=0) * 10000
vol

vol[:21].iplot(title='Volatility of daily UK government yields', xTitle='Tenor', yTitle='Volatility (bps)',
         color='cornflowerblue')


cov_= pd.DataFrame(np.cov(diff_, rowvar=False)*252/10000, columns=diff_.columns, index=diff_.columns)
cov_.style.format("{:.4%}")


#Eigen 
# Perform eigen decomposition
eigenvalues, eigenvectors = np.linalg.eig(cov_)

# Sort values (good practice)
idx = eigenvalues.argsort()[::-1]   
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:,idx]

# Format into a DataFrame 
df_eigval = pd.DataFrame({"Eigenvalues": eigenvalues})

eigenvalues

#Explained variance
# Work out explained proportion 
df_eigval["Explained proportion"] = df_eigval["Eigenvalues"] / np.sum(df_eigval["Eigenvalues"])
df_eigval = df_eigval[:10]
df_eigval

#Format as percentage
df_eigval.style.format({"Explained proportion": "{:.2%}"})


# Subsume first 3 components into a dataframe
pcadf = pd.DataFrame(eigenvectors[:,0:3], columns=['PC1','PC2','PC3'])
pcadf[:10]


pcadf.iplot(title='First Three Principal Components', secondary_y='PC1', secondary_y_title='PC1', 
            yTitle='change in yield (bps)')

