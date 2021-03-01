import numpy as np
import math #allowed?

def W_matrix(X): #each row is a point, meaning that d = # of columns and n = # of rows
    n = X.shape[0]
    W = [[exponent(X[i],X[j]) for j in range(n)] for i in range(n)]
    return np.array(W)

def exponent(v1,v2):
    exp = (-1)*(np.linalg.norm(v1-v2))/2
    return math.exp(exp)

def D_matrix(W): #represented as a vector containing diagonal terms
    D = W.sum(axis = 1)
    D_half = [1/math.sqrt(D[i]) for i in range(len(D))]
    return D_half

def Lnorm_matrix(X):
    W = W_matrix(X)
    n = W.shape[0]
    D = np.diag(D_matrix(W))
    return np.eye(n) - D.dot(W.dot(D))


