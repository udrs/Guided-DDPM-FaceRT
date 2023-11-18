import numpy as np
A = np.arange(3,15)
print(A)
# [ 3  4  5  6  7  8  9 10 11 12 13 14]
print(A[3])
# 6
B = A.reshape(3,4)
print(B)
'''
[[ 3  4  5  6]
 [ 7  8  9 10]
 [11 12 13 14]]
'''
print(B[2])
# [11 12 13 14]
print(B[0][2])
# 5
print(B[0,2])
# 5
# list切片操作
print(B[1,1:3]) # [8 9] 1:3表示1-2不包含3
 
for row in B:
    print(row)
 
'''
[3 4 5 6]
[ 7  8  9 10]
[11 12 13 14]
'''
# 如果要打印列，则进行转置即可

for column in B.T:
    print(column)
'''
[ 3  7 11]
[ 4  8 12]
[ 5  9 13]
[ 6 10 14]
'''
# 多维转一维
A = np.arange(3,15).reshape((3,4))
# print(A)
print(A.flatten())
# flat是一个迭代器，本身是一个object属性
for item in A.flat:
    print(item)