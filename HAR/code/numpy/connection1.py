import numpy as np
A = np.array([1,1,1])
B = np.array([2,2,2])
print(np.vstack((A,B)))
# vertical stack 上下合并,对括号的两个整体操作。
'''
[[1 1 1]
 [2 2 2]]
'''
C = np.vstack((A,B))
print(C)
print(A.shape,B.shape,C.shape)
# (3,) (3,) (2, 3)
# 从shape中看出A,B均为拥有3项的数组(数列)
# horizontal stack左右合并
D = np.hstack((A,B))
print(D)
# [1 1 1 2 2 2]
print(A)
print(B)
print(C)
print(D)
print(A.shape,B.shape,D.shape)
# (3,) (3,) (6,)
# 对于A,B这种，为数组或数列，无法进行转置，需要借助其他函数进行转置

# 数组 转置成 矩阵
print(A[np.newaxis,:]) # [1 1 1]变为[[1 1 1]]
print(A[np.newaxis,:].shape) # (3,)变为(1, 3)
print(A[:,np.newaxis])
'''
[[1]
 [1]
 [1]]
'''