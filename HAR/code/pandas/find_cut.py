import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 选取pandas 
s = pd.Series([1,3,5,np.nan,6,8])
print(s)

dates = pd.date_range('20130101', periods=6)
print(dates)

df = pd.DataFrame(np.random.randn(6,4),index=dates,columns=list('ABCD'))
print(df)

# 用[] 选取
print(df['A']) # 取单独的数列

print(df[0:3]) # 上下切割，选择针对axis=0

print(df['20130102':'20130104'])

# loc 获取 ; 与[]类似
# loc 在, 之前row index; 在, 之后 column index
print(df.loc[dates[0]])

print(df.loc[:,['A','B']])

print(df.loc['20130102':'20130104',['A','B']])


# iloc 是通过 index 选择； loc用标签选择； 其余类似
print(df.iloc[3]) # 默认选择行：第四行

print(df.iloc[3:5,0:2])

print(df.iloc[[1,2,4],[0,2]]) # 指定行数

print(df.iloc[1:3,:]) # 切横行

print(df.iloc[:,1:3]) # 切竖列

print(df.iloc[1,1]) # 直接指定元素位置

# 布尔索引
print(df[df.A >0]) # 选取A列中 大于零元素 所在的行

print(df[df<0]) # 所有元素比较

df2 = df.copy()
df2['E'] = ['one','one','two','three','four','three']

print(df2[df2['E'].isin(['two','four'])])

