import scipy as sp
from scipy.integrate import quad,dblquad,tplquad
from scipy import linalg as la
from scipy import sparse as s
import numpy as np
from scipy.optimize import optimize
 
 
def f(x):
     return x

# quad()函数进行积分
x_lower = 0  # the lower limit of x
x_upper = 1  # the upper limit of x
val, abserr = quad(f, x_lower, x_upper)
print(val, abserr)
 
A = sp.rand(2, 2)
B = sp.rand(2, 2)
print(A)
 
X = la.solve(A, B)
# C = la.dot(A, B)
print(X)
 
# 特征值和特征向量
evals = la.eigvals(A)
print(evals)
 
# 特征向量如下所示
evals, evect = la.eig(A)
print(evals, evect)
# 求逆、转置
D = la.inv(A)
E = la.det(A)
print(D)
print(E)
 
# 稀疏矩阵
A = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
print(A)
 
C = s.csr_matrix(A)
print(C)
print(C.toarray())
print(C.todense())
 
B = np.array([[1, 1, 0], [0, 2, 0], [0, 0, 3]])
print(B)
 
C = s.csr_matrix(B)
print(C)
 
 
# 优化
def f(x):
         return x**2-4
 
 
print(optimize.fmin_bfgs(f, 0))
help(optimize.fmin_bfgs)