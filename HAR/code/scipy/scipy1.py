# https://www.yiibai.com/scipy/scipy_basic_functionality.html

import numpy as np
list = [1,2,3,4]
arr = np.array(list)
print (arr)

# 矩阵 2-D
mat = np.matrix('1 2; 3 4')
print(mat)

print(mat.H) # 共轭矩阵

print(mat.T) # 转置矩阵

