import networkx as nx
import numpy as np
import time
from sklearn.cluster import KMeans
import pickle
import scipy
import sys

def EigVec(eigval, eigvec, cluster_num):
    dim = len(eigval)
    dictEigval = dict(zip(eigval, range(0,dim)))
    kEig = np.sort(eigval)[:cluster_num]
    ix = [dictEigval[k] for k in kEig]
    return eigval[ix], eigvec[:,ix]


t1=time.time()
G = nx.read_edgelist('combined_facebook.txt', create_using=nx.Graph(), nodetype=int)
t2=time.time()
# L = nx.laplacian_matrix(G)
A = nx.to_numpy_matrix(G, dtype=np.int8)
A = np.asarray(A)
t3=time.time()
diag = A.sum(axis=1)
A = np.diag(diag) - A


with open('laplacian_facebook', 'wb') as f1:
	pickle.dump(A, f1)


# # e = np.linalg.eigvals(L.A)
# # print("Largest:", max(e))
# # print("Smallest:", min(e))


# val, vec = scipy.linalg.eig(L.A)
# t4=time.time()
# print(t2-t1, t3-t2, t4-t3)

# cluster_num = 3
# eigval, eigvec = EigVec(val, vec, cluster_num)
# clf = KMeans(n_clusters=cluster_num)
# s = clf.fit(eigvec)
# C = s.labels_
# print(C)
# with open('standard_label', 'wb') as f1:
# 	pickle.dump(C, f1)

# print(t2-t1, t3-t2, t4-t3)
