import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 创建pandas 

dates = pd.date_range('20130101', periods=6)
print(dates)

df = pd.DataFrame(np.random.randn(6,4),index=dates,columns=list('ABCD'))
print(df)

# csv
df.to_csv('foo.csv')
pd.read_csv('foo.csv')


# excel
df.to_excel('foo.xlsx', sheet_name='Sheet1')
pd.read_excel('foo.xlsx','Sheet1', index_col=None, na_values=['NA'])

# HDF5