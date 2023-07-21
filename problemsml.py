import numpy as np
import pandas as pd
import numpy as np
from matplotlib import pyplot as pt
def computeCost(X,y,theta):
    m=len(y)
    predictions= X*theta-y
    sqrerror=np.power(predictions,2)
    return 1/(2*m)*np.sum(sqrerror)

def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    jhistory = np.zeros((num_iters,1))
    for i in range(num_iters):
        h = X * theta
        s = h - y
        theta = theta - (alpha / m) * (s.T*X).T
        jhistory_iter = computeCost(X, y, theta)
    return theta,jhistory_iter


data = open(r'C:\Users\Coding\Desktop\machine-learning-ex1\ex1\ex1data1.txt')
data1=np.array(pd.read_csv(r'C:\Users\Coding\Desktop\machine-learning-ex1\ex1\ex1data1.txt',header=None))
y =np.array(data1[:,1])
m=len(y)
y=np.asmatrix(y.reshape(m,1))
X = np.array([data1[:,0]]).reshape(m,1)
X = np.asmatrix(np.insert(X,0,1,axis=1))
theta=np.zeros((2,1))
iterations = 1500
alpha = 0.01;

print('Testing the cost function ...')
J = computeCost(X, y, theta)
print('With theta = [0 , 0]\nCost computed = ', J)
print('Expected cost value (approx) 32.07')
theta=np.asmatrix([[-1,0],[1,2]])
J = computeCost(X, y, theta)
print('With theta = [-1 , 2]\nCost computed =', J)
print('Expected cost value (approx) 54.24')

theta,JJ = gradientDescent(X, y, theta, alpha, iterations)
print('Theta found by gradient descent:')
print(theta)
print('Expected theta values (approx)')
print(' -3.6303\n  1.1664\n')
predict1 = [1, 3.5] *theta
print(predict1*10000)
