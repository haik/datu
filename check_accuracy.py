import pickle

with open('label_facebook_computed', 'rb') as f1:
	computed = pickle.load(f1)
for i in range (100):
	data = computed[(40*i):(40*i+40)]
	print(data)


# with open('computed_label', 'rb') as f1:
# 	computed = pickle.load(f1)
# computed -= 1
# for i in range (400):
# 	data = computed[(10*i):(10*i+10)]
# 	for j in range(10):
# 		if data[j] == -1:
# 			data[j] += 3
# 	print(data)