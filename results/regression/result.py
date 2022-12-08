import numpy as np
from numpy import genfromtxt
import math

def PearsonCoefficient(X, Y):
    r=np.sum((X-np.average(X))*(Y-np.average(Y)))/math.sqrt(np.sum((X-np.average(X))**2)*np.sum((Y-np.average(Y))**2))
    return r
def RMSE(X,Y):
    rmse=math.sqrt(np.sum((Y-X)**2)/len(Y))
    return rmse
def MAE(X,Y):
    mae=np.average(abs(X-Y))
    return mae

test_results = genfromtxt('test_results.csv', delimiter=',')
test_target = test_results[:,1]
test_predict = test_results[:,2]

test_mae = MAE(test_target, test_predict)
test_rmse = RMSE(test_target ,test_predict)
r_GB=PearsonCoefficient(test_target ,test_predict )

print('test_mae:',test_mae)
print('test_rmse:',test_rmse)
print('test_r:',r_GB)

