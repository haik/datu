import networkx as nx
import numpy as np
import time
import os
from sklearn.cluster import KMeans
import pickle
from all_ss_module import *
from test_eigen_ss import *
import gc

def EigVec(eigval, eigvec, cluster_num):
    dim = len(eigval)
    dictEigval = dict(zip(eigval, range(0,dim)))
    kEig = np.sort(eigval)[:cluster_num]
    ix = [dictEigval[k] for k in kEig]
    return eigval[ix], eigvec[:,ix]

# t1=time.time()
# G = nx.read_edgelist('combined_facebook.txt', create_using=nx.Graph(), nodetype=int)
# t2=time.time()
# L = nx.laplacian_matrix(G)
# t3=time.time()
# del G
# gc.collect()
# print("read graph", t2-t1)
# print("Laplacian matrix", t3-t2)
# LM = L.A
# del L
# gc.collect()

with open('laplacian_facebook', 'rb') as f1:
	LM = pickle.load(f1)
print(type(LM))
# LM = np.asarray(LM)
# print(type(LM))
row, column = LM.shape
print(row, column)
# print(np.max(LM), np.min(LM))
L1 = np.random.randint(-10, 10, size=(row, column))
# print(type(L1))
t4=time.time()
L2 = LM - L1
t5 = time.time()
print("splitting", t5-t4)
del LM
gc.collect()

t6 = time.time()
V1, V2, T1, T2, toff, ton = LanczosTri(L1, L2, 150)
t7 = time.time()
print("Lanczos", t7-t6, toff, ton)
# print(np.max(T1+T2), np.min(T1+T2))
del L1, L2
gc.collect()

t7 = time.time()
eigval1, eigval2, eigvec1, eigvec2, toff, ton = EigVecSs(T1, T2)
t8 = time.time()
print("decomposition", t8-t7, toff, ton)

t8 = time.time()
eigvec1, eigvec2, toff, ton = MatMulss(V1, V2, eigvec1, eigvec2)
eigval = eigval1 + eigval2
eigval = np.diagonal(eigval)
eigvec = eigvec1 + eigvec2
t9 = time.time()
print("finaVec", t9-t8, toff, ton)

cluster_num = 10
eigval, eigvec = EigVec(eigval, eigvec, cluster_num)
clf = KMeans(n_clusters=cluster_num)
s = clf.fit(eigvec)
C = s.labels_
print(C)

with open('label_facebook_computed', 'wb') as f1:
	pickle.dump(C, f1)
