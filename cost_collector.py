import numpy as np
import time
import sys

# row = 107614
# column = 10
# L1 = np.random.random(size=(row, column))
# L2 = np.random.random(size=(row, column))
# # row, column = L0.shape
# # print(row, column)
# t4=time.time()
# L0 = L1 + L2
# t5 = time.time()
# print("splitting", t5-t4)
# print(sys.getsizeof(L1))
# print(sys.getsizeof(L2))
# print(sys.getsizeof(L2))

row = 500
column = 2
L0 = np.random.random(size=(row,column))
t4=time.time()
L1 = np.random.random(size=(row,column))
L2 = L0 - L1
t5 = time.time()
print("splitting", t5-t4)
print(L1[0])
print(sys.getsizeof(L1))
