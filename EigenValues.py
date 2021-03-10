from numpy import transpose, diagonal
from numpy.linalg import norm
import numpy as np
import sklearn.datasets
import Laplacian
import time
import math #used only for ceil, check if it's fine

def QR(A):

    n = A.shape[0]
    R = np.zeros((n,n))
    Q = np.zeros((n,n))
    Norm = np.zeros(n)
    U = A.copy()
    
    for i in range(n): 
        Norm[i] = norm(U[:,i])
        Q[:,i] = U[:,i]/Norm[i]
        R[i][i+1:n] = Q[:,i].dot(U[:,i+1:n])
        U[:,i+1:n] = U[:,i+1:n] - transpose(np.array([Q[:,i]])).dot(np.array([R[i,i+1:n]]))

    np.fill_diagonal(R,Norm)
    return (Q,R)

def QR_Iterations(A, epsilon = 0.0001): #doesn't converge all the way

    n = A.shape[0]
    Abar = A.copy()
    Qbar = np.eye(n,dtype=float)
    Sum = 0

    for i in range(max(n,10)):
        t1 = time.time()
        (Q,R) = QR(Abar)
        Sum += time.time() - t1
        Abar = R.dot(Q)
        
        diff_mat = Qbar- Qbar.dot(Q)
        if (abs(diff_mat) <= epsilon).all():
            return (Abar,Qbar)

        Qbar = Qbar.dot(Q)
        
    print("QR time tot is: ", Sum)
    return (Abar,Qbar)
    
def eigengap_test(v, Random, k): #check this function
    n = len(v)
    if(not Random):
        delta = [abs(v[i]-v[i+1]) for i in range(math.ceil(n/2))] #odd values of n?
        return delta.index(max(delta))
    return k

def U_matrix(A, Random, k):
    n = A.shape[0]
    (Abar,Qbar) = QR_Iterations(A)

    dtype = [('key',float),('value',int)]
    tup_arr = [(Abar[i][i],i) for i in range(n)]
    tup_arr = np.array(tup_arr,dtype = dtype)
    tup_arr = np.sort(tup_arr,order = 'key')

    k = eigengap_test([tup_arr[i][0] for i in range(n)], Random, k)
    U = [Qbar[:,tup_arr[j][1]] for j in range(k+1)] #should it be k+1 or k? I think k+1.
    U = np.array(U)
    U = transpose(U) #why necessary?
    return U

def T_matrix(A, Random, k):
    U = U_matrix(A, Random, k)
    n = U.shape[0]
    m = U.shape[1]
    T = [[U[i][j]/norm(U[i]) for j in range(m)] for i in range(n)]
    return np.array(T)

def Create_vectors_to_cluster(X, Random, k):
    Lnorm = Laplacian.Lnorm_matrix(X)
    T = T_matrix(Lnorm, Random, k)
    return T

###########################################################  CURRENTLY NOT IN USE ####################################################################

def QR_draft(A): #less efficient, not in use
    n = A.shape[0]
    R = np.zeros((n,n))
    Q = np.zeros((n,n))
    U = A.copy()
    
    for i in range(n):
        norm = np.linalg.norm(U[:,i])
        R[i][i] = norm
        Q[:,i] = U[:,i]/norm
        for j in range(i+1,n): 
            R[i][j] = Q[:,i].dot(U[:,j])
            U[:,j] = U[:,j] - (R[i][j]*Q[:,i])
    return (Q,R)

def multiply_triangular(R,Q): #currently not int use
    n = R.shape[0]
    k = R.shape[1]
    m = Q.shape[1]
    return np.array([[R[i,:(k-i)].dot(Q[:(k-i),j]) for j in range(m)] for i in range(n)])

######################################################################################################################################################

###########################################################  DEBUG FUNCTIONS #########################################################################

def debug_general():
    A = np.array([(5.0,3.0,1.0,4.0),(3.0,6.0,0.0,2.5),(1.0,0.0,3.0,1.7),(4.0,2.5,1.7,10.0)])
    #A = np.full((3,3),2)
    (Q,R) = QR(A)
    print("Q is:")
    print(Q)
    print("R is:")
    print(R)
    print("The QR multiplication is:")
    print(Q.dot(R))
    print("")
    (Abar,Qbar) = QR_Iterations(A)
    print(Abar)
    print("")
    #print("")
    #print(Qbar)
    #print("")
    #print(U_matrix(A))
    #print(T_matrix(U_matrix(A)))
    print(np.linalg.eigvalsh(A))
    #(Abar,Qbar) = QR_Iterations(A)
    #print(Abar)
    #A = np.array([(5.0,3.0),(3.0,6.0)])
    #print("")
    #[Q,R] = QR(A)
    #print(np.matmul(np.transpose(Q),Q))
    #print(R)
    #print(np.matmul(Q,R))
    return 0

#print(Create_vectors_to_cluster(np.random.rand(4,400)))

def debug_QR():
    lst = []
    for i in range(50):
        A = np.random.rand(i,i)
        (Q,R) = QR(A)
        lst.append((abs(A - np.matmul(Q,R)) < 0.00000001).all())
    print(lst)

def debug_QRIterations(): 
    lst = []
    for i in range(1):
        #A = np.random.rand(50,3)*0.1
        t1 = time.time()
        A = sklearn.datasets.make_blobs(n_samples = 200,n_features = 12,centers = 12)[0]
        Lnorm = Laplacian.Lnorm_matrix(A)
        T = T_matrix(Lnorm, False,0)
        t2 = time.time()
        print("The time is approximatly ", t2 - t1, " seconds")
        #(Abar,Qbar) = QR_Iterations(Lnorm)
        #eig_sorted_1 = [Abar[j][j] for j in range(Lnorm.shape[0])]
        #eig_sorted_1.sort()
        #eig_sorted_2 = np.linalg.eigvalsh(Lnorm)
        #eig_sorted_2.sort()
        #lst.append(max([abs(eig_sorted_1[j] - eig_sorted_2[j]) for j in range(len(eig_sorted_1))]))
        print(np.array(T).shape[1])
    #print("")
    #print(lst)
        
######################################################################################################################################################

debug_QRIterations()
#debug_QR()


