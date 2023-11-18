import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.DataFrame(np.random.randn(10,4))

print(df)

pieces = [df[:3], df[3:7]] # []针对上下切割
df2 = pd.concat(pieces)
print(df2)

