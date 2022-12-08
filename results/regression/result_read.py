import numpy as np
from numpy import genfromtxt
import math
from sklearn.metrics import mean_squared_error,r2_score


test_results = genfromtxt('test_results.csv', delimiter=',')
test_target = test_results[1:,1]
test_predict = test_results[1:,2]

def PearsonCoefficient(X, Y):
    r=np.sum((X-np.average(X))*(Y-np.average(Y)))/math.sqrt(np.sum((X-np.average(X))**2)*np.sum((Y-np.average(Y))**2))
    return r

test_mae = abs(test_target -test_predict).mean()
test_rmse = np.sqrt(mean_squared_error(test_target, test_predict))
test_r2 = r2_score(test_target, test_predict)
r_GB=PearsonCoefficient(test_target ,test_predict )

print('test_mae:',test_mae)
print('test_rmse:',test_rmse)
print('test_r2:',test_r2)
print('test_r:',r_GB)


test_results = genfromtxt('valid_results.csv', delimiter=',')
test_target = test_results[1:,1]
test_predict = test_results[1:,2]

def PearsonCoefficient(X, Y):
    r=np.sum((X-np.average(X))*(Y-np.average(Y)))/math.sqrt(np.sum((X-np.average(X))**2)*np.sum((Y-np.average(Y))**2))
    return r

test_mae = abs(test_target -test_predict).mean()
test_rmse = np.sqrt(mean_squared_error(test_target, test_predict))
test_r2 = r2_score(test_target, test_predict)
r_GB=PearsonCoefficient(test_target ,test_predict )

print('val_mae:',test_mae)
print('val_rmse:',test_rmse)
print('val_r2:',test_r2)
print('val_r:',r_GB)

test_results = genfromtxt('train_results.csv', delimiter=',')
test_target = test_results[1:,1]
test_predict = test_results[1:,2]

def PearsonCoefficient(X, Y):
    r=np.sum((X-np.average(X))*(Y-np.average(Y)))/math.sqrt(np.sum((X-np.average(X))**2)*np.sum((Y-np.average(Y))**2))
    return r

test_mae = abs(test_target -test_predict).mean()
test_rmse = np.sqrt(mean_squared_error(test_target, test_predict))
test_r2 = r2_score(test_target, test_predict)
r_GB=PearsonCoefficient(test_target ,test_predict )

print('train_mae:',test_mae)
print('train_rmse:',test_rmse)
print('train_r2:',test_r2)
print('train_r:',r_GB)
