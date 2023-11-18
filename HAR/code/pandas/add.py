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

# 设定'F' 列 的值
s1 = pd.Series([1,2,3,4,5,6],index=pd.date_range('20130102',periods=6))
df['F']=s1
print(df)

df.iat[0,1] = 0 # 设定具体值
print(df)

df.loc[:,'D'] = np.array([5] * len(df))
print(df)


