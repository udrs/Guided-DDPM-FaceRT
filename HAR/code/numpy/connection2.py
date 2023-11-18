import numpy as np
A = np.array([1,1,1])
B = np.array([2,2,2])

# concatenate的第一个例子
print("------------")
print(A[:,np.newaxis].shape) # (3,1)
A = A[:,np.newaxis] # 数组转为矩阵
B = B[:,np.newaxis] # 数组转为矩阵
# axis=0纵向合并
C = np.concatenate((A,B,B,A),axis=0)  # 即第0维 增加： 3*1 + 3*1 + 3*1 + 3*1 = 12*1
print(C)
'''
[[1]
 [1]
 [1]
 [2]
 [2]
 [2]
 [2]
 [2]
 [2]
 [1]
 [1]
 [1]]
'''
# axis=1横向合并
C = np.concatenate((A,B),axis=1) # 即第1维增加：3*1 + 3*1 = 3*2
print(C)
'''
[[1 2]
 [1 2]
 [1 2]]
'''

# concatenate的第二个例子
print("-------------")
a = np.arange(8).reshape(2,4)
b = np.arange(8).reshape(2,4)
print(a)
print(b)
print("-------------")
# axis=0多个矩阵纵向合并
c = np.concatenate((a,b),axis=0)
print(c)
# axis=1多个矩阵横向合并
c = np.concatenate((a,b),axis=1)
print(c)
'''
[[0 1 2 3]
 [4 5 6 7]
 [0 1 2 3]
 [4 5 6 7]]
[[0 1 2 3 0 1 2 3]
 [4 5 6 7 4 5 6 7]]
'''