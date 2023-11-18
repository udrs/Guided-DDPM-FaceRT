import numpy as np
# 一维矩阵运算
a = np.array([10,20,30,40])
b = np.arange(4)
print(a,b)
# [10 20 30 40] [0 1 2 3]
c = a - b
print(c)
# [10 19 28 37]
print(a*b) # 若用a.dot(b),则为各维乘积后再之和
# [  0  20  60 120]
print(a.dot(b))
# 200
# 在Numpy中，想要求出矩阵中各个元素的乘方需要依赖双星符号 **，以二次方举例，即：
c = b**2
print(c)
# [0 1 4 9]
# Numpy中具有很多的数学函数工具
c = np.sin(a)
print(c)
# [-0.54402111  0.91294525 -0.98803162  0.74511316]
print(b<2)
# [ True  True False False]
a = np.array([1,1,4,3])
b = np.arange(4)
print(a==b)
# [False  True False  True]