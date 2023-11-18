import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

tuples = list(zip(*[['bar', 'bar', 'baz', 'baz',
                    'foo', 'foo', 'qux', 'qux'],
                    ['one', 'two', 'one', 'two',
                    'one', 'two', 'one', 'two']]))
index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])

df = pd.DataFrame(np.random.randn(8,2), index= index, columns=['A', 'B'])

df2 = df[:4]

print('df:')
print(df)

print('df2: ')
print(df2)


stacked = df2.stack() # stack 把竖列标记 转换成 横轴seris标记
print('stacked: ')
print(stacked)

print('unstack: ')
print(stacked.unstack()) # unstack: 把横轴seris标记 转换成 竖列标记

print(stacked.unstack(1))

print(stacked.unstack(0))