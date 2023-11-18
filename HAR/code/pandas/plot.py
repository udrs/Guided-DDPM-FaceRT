import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ts = pd.Series(np.random.randn(1000), index = pd.date_range('1/1/2000', periods=1000))

print(ts)

ts = ts.cumsum()

print(ts)

ts.plot()
plt.show() # this expression is necessary

df2 = pd.DataFrame(np.random.randn(1000, 4),
                index=ts.index, 
                columns=['A', 'B', 'C', 'D'])

print(df2)

df2 = df2.cumsum()

print(df2)

plt.figure(); df2.plot(); plt.legend(loc='best') # loc 就是 location 是最佳
plt.show()



