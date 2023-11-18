import numpy as np
 
A = np.arange(2,14).reshape((3,4))
print(A)
 
# 最小元素索引
print(np.argmin(A)) # 0
# 最大元素索引
print(np.argmax(A)) # 11
# 求整个矩阵的均值
print(np.mean(A)) # 7.5
print(np.average(A)) # 7.5
print(A.mean()) # 7.5
# 中位数
print(np.median(A)) # 7.5
# 累加
print(np.cumsum(A))
# [ 2  5  9 14 20 27 35 44 54 65 77 90]
# 累差运算
B = np.array([[3,5,9],
              [4,8,10]])
print(np.diff(B))
'''
[[2 4]
 [4 2]]
'''
C = np.array([[0,5,9],
              [4,0,10]])
print(np.nonzero(B))
print(np.nonzero(C))
# 返回值是 非零元素对应的坐标，只是把坐标按照x,y,z等分开表示
'''
# 将所有非零元素的行与列坐标分割开，重构成两个分别关于行和列的矩阵
(array([0, 0, 0, 1, 1, 1], dtype=int64), array([0, 1, 2, 0, 1, 2], dtype=int64))
(array([0, 0, 1, 1], dtype=int64), array([1, 2, 0, 2], dtype=int64))
'''
# 仿照列表排序
A = np.arange(14,2,-1).reshape((3,4)) # -1表示反向递减一个步长
print(A)
'''
[[14 13 12 11]
 [10  9  8  7]
 [ 6  5  4  3]]
'''
print(np.sort(A))
'''
# 只是对每行进行递增排序
[[11 12 13 14]
 [ 7  8  9 10]
 [ 3  4  5  6]]
'''
# 矩阵转置
print(np.transpose(A))
'''
[[14 10  6]
 [13  9  5]
 [12  8  4]
 [11  7  3]]
'''
print(A.T)
'''
[[14 10  6]
 [13  9  5]
 [12  8  4]
 [11  7  3]]
'''
print(A)
print(np.clip(A,5,9))
'''
clip(Array,Array_min,Array_max)
将Array_min<X<Array_max  X表示矩阵A中的数，如果满足上述关系，则原数不变。
否则，如果X<Array_min，则将矩阵中X变为Array_min;
如果X>Array_max，则将矩阵中X变为Array_max.
[[9 9 9 9]
 [9 9 8 7]
 [6 5 5 5]]
'''