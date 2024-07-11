import pyDOE as pdoe
from scipy.stats.distributions import norm
import numpy as np
import json
from sys import argv

base_file = argv[1]
output_file = argv[2]

with open(f'../time_series_data/{base_file}','r') as file:
    base_series = json.load(file)

noise_train = pdoe.lhs(50,samples = 40)
noise_test = pdoe.lhs(50,samples = 40)
noise_train = norm(loc = 0, scale = 0.12).ppf(noise_train)
noise_test = norm(loc = 0, scale = 0.12).ppf(noise_test)

noised_series_train = dict()
noised_series_test = dict()

for key,value in base_series.items():
    for episode in range(50):
        X = np.array(value)
        X = X + X*noise_train[:,episode]
        X = np.floor(X)
        X = list(map(lambda t : int(t) if t>0 else 0, X))
        if key in noised_series_train.keys():
            noised_series_train[key].append(X)
        else:
            noised_series_train[key] = [X,]

for key,value in base_series.items():
    for episode in range(50):
        X = np.array(value)
        X = X + X*noise_test[:,episode]
        X = np.floor(X)
        X = list(map(lambda t : int(t) if t>0 else 0, X))
        if key in noised_series_test.keys():
            noised_series_test[key].append(X)
        else:
            noised_series_test[key] = [X,]

for episode in range(50):
    x = np.zeros(40, dtype = int)
    y = np.zeros(40, dtype =int)
    for app_indx in range(1,4):
        x = np.add(x,noised_series_train[f'(5, {app_indx})'][episode])
        y = np.add(y,noised_series_train[f'(4, {app_indx})'][episode])
    #noised_series_train["rem_time"][episode] = list(map(lambda y:np.random.randint(y) if y>0 else 0, x))
    #print(x - noised_series_train["rem_time"][episode] )
    #z1 = np.zeros(40, dtype = int)
    z = np.zeros(40, dtype = int)
    w1 = list(map(lambda t:np.random.randint(t) if t>0 else 0, x))
    w2 = list(map(lambda t:np.random.randint(t) if t>0 else 0, y))
    #z1 = w1
    z = x + w2
    #noised_series_train["rem_time"][episode] = w1
    noised_series_train["rem_time_2"][episode] = z.tolist()
    


for episode in range(50):
    x = np.zeros(40, dtype = int)
    y = np.zeros(40, dtype =int)
    for app_indx in range(1,4):
        x = np.add(x,noised_series_test[f'(5, {app_indx})'][episode])
        y = np.add(y,noised_series_test[f'(4, {app_indx})'][episode])
    #noised_series_train["rem_time"][episode] = list(map(lambda y:np.random.randint(y) if y>0 else 0, x))
    #print(x - noised_series_train["rem_time"][episode] )
    #z1 = np.zeros(40, dtype = int)
    z = np.zeros(40, dtype = int)
    w1 = list(map(lambda t:np.random.randint(t) if t>0 else 0, x))
    w2 = list(map(lambda t:np.random.randint(t) if t>0 else 0, y))
    #z1 = w1
    z = x + w2
    #noised_series_test["rem_time"][episode] = w1
    noised_series_test["rem_time_2"][episode] = z.tolist()
    
#print(noised_series_test)
with open(f'../time_series_data/{output_file}_train.json','w') as file:
    json.dump(noised_series_train,file)

with open(f'../time_series_data/{output_file}_test.json','w') as file:
    json.dump(noised_series_test,file)
