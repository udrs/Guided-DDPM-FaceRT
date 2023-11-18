import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 创建pandas 
s = pd.Series([1,3,5,np.nan,6,8])
print(s)

dates = pd.date_range('20130101', periods=6)
print(dates)

df = pd.DataFrame(np.random.randn(6,4),index=dates,columns=list('ABCD'))
print(df)

df2 = pd.DataFrame({'A': 1.,
                    'B': pd.Timestamp('20130102'),
                    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                    'D': np.array([3] * 4, dtype='int32'),
                    'E': pd.Categorical(["test", "train", "test", "train"]),
                    'F': 'foo'
                    })
print(df2)

print(df2.dtypes)

# 查看数据
print(df.head()) # 返回前几行，默认值是5

print(df.tail(3)) # 返回后几行

print(df.index) # 横行左侧的

print(df.columns) # 数列上边的

print(df.values)

print(df.describe()) # 数据的基本统计汇总

print(df.T) # 转置

print(df.sort_index(axis=1, ascending=False)) # 按轴排序

# print(df.sort(column='B')) # 按值进行排序