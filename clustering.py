import numpy as np
import networkx as nx
import pickle
from scipy.sparse.linalg.eigen import eigsh
import gc
from sklearn.cluster import KMeans

def EigVec(eigval, eigvec, cluster_num):
    dim = len(eigval)
    dictEigval = dict(zip(eigval, range(0,dim)))
    kEig = np.sort(eigval)[:cluster_num]
    ix = [dictEigval[k] for k in kEig]
    return eigval[ix], eigvec[:,ix]

with open('laplacian_facebook', 'rb') as f1:
	L = pickle.load(f1)
# L = LM.A
# del LM
# gc.collect()

# dim = 2
# nnodes, _ = L.shape
# k = dim + 1
# ncv = max(2*k+1, int(np.sqrt(nnodes)))

# eigvalues, eigenvectors = eigsh(L, 3, which='SM', ncv=ncv)
# index = np.argsort(eigenvalues)[1:k]
# print(np.real(eigenvectors[:, index]))

eigval, eigvec = np.linalg.eig(L)

cluster_num = 10
eigval, eigvec = EigVec(eigval, np.real(eigvec), cluster_num)
clf = KMeans(n_clusters=cluster_num)
s = clf.fit(eigvec)
C = s.labels_
print(C)

with open('label_facebook', 'wb') as f1:
	pickle.dump(C, f1)
