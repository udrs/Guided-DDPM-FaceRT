import numpy as np

array = np.array([
    [1,3,5],
    [4,6,9]
])
 
print(array)

print('number of dim:', array.ndim) # 维度

print('shape:', array.shape) # 行数与列数

print("size:", array.size) # 元素个数，即乘积结果

# 一维数组
a = np.array([2,23,4], dtype=np.int32) 
print(a)
print(a.dtype)
# 多维数组
b = np.array([[2, 3, 4],
            [3,4,5]])
print(b)
# 全零数组
c = np.zeros((3,4))
print(c)
# 全一数组
d = np.ones((3,4), dtype=np.int)
print(d)
# 全空数组
e = np.empty((3,4))
print(e)
# 连续数组
f = np.arange(10,21,2) # 从10到20，步长为2
print(f)
# reshpae 操作
g = f.reshape((2,3))
print(g)
# 连续型数据
h = np.linspace(1,10,20) # 从1到10，等分成20个数据
print(h)
i = h.reshape((5,4))
print(i)







