import numpy as np
# `=`赋值方式会带有关联性
a = np.arange(4)
print(a) # [0 1 2 3]
 
b = a
c = a
d = b
a[0] = 11
print(a) # [11  1  2  3]
print(b) # [11  1  2  3]
print(c) # [11  1  2  3]
print(d) # [11  1  2  3]
print(b is a) # True
print(c is a) # True
print(d is a) # True
 
d[1:3] = [22,33]
print(a) # [11 22 33  3]
print(b) # [11 22 33  3]
print(c) # [11 22 33  3]

 
# 此时a与b已经没有关联
a = np.arange(4)
print(a) # [0 1 2 3]
b =a.copy() # deep copy
print(b) # [0 1 2 3]
a[3] = 44
print(a) # [ 0  1  2 44]
print(b) # [0 1 2 3]
 
# 此时a与b已经没有关联