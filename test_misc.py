import numpy as np
A = np.random.randint(0, 10, size=(10, 10), dtype=np.int8)
print(type(A), A.shape)
diag = A.sum(axis=1)
print(diag)
D = np.diag(diag)
print(D)