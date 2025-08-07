

from keras import layers
from keras import backend as K
from keras.models import Model
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input, Reshape, Add
from keras.utils.generic_utils import get_custom_objects
from keras.regularizers import l1_l2, l2, l1
from keras.metrics import categorical_accuracy
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from collections import Counter
from collections import deque
from scipy.optimize import nnls

import pandas as pd
import fcntl, os
import errno
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import glob
import os
import sys
import subprocess
import math
import time
import random
import socket
import innvestigate
#import innvestigate.backend as ibackend

import threading

import shutil

graph = tf.get_default_graph()


#tf.compat.v1.disable_eager_execution()

q = deque()
cur_cycle = 0

target_dir = sys.argv[1] #Make this customizable for the future
out_fol = sys.argv[2]

retrain_dir = "./targets/" + target_dir + "/" + out_fol + "/update_queue"

IP = "127.0.0.1"
PORT = 4455
ADDR = (IP, PORT)
SIZE = 1024
FORMAT = "iso-8859-1"

os.path.isdir("./targets/" + target_dir + "/bitmaps/") or os.makedirs("./targets/" + target_dir + "/bitmaps/")
os.path.isdir("./targets/" + target_dir + "/in/gen/") or os.makedirs("./targets/" + target_dir + "/in/gen/")
os.path.isdir("./targets/" + target_dir + "/checkpoint") or os.makedirs("./targets/" + target_dir + "/checkpoint")
os.path.isdir("./targets/" + target_dir + "/out/queue_grads") or os.makedirs("./targets/" + target_dir + "/out/queue_grads")
os.path.isdir("./targets/" + target_dir + "/inc_learn") or os.makedirs("./targets/" + target_dir + "/inc_learn")
os.path.isdir("./targets/" + target_dir + "/" + out_fol + "/update_queue") or os.makedirs("./targets/" + target_dir + "/" + out_fol + "/update_queue")


init_seed_list = glob.glob('./targets/' + target_dir + '/out/queue/id*') #Note that you will need seperate input and output directories between the two times the fuzzer is run
runtime_seeds = glob.glob('./targets/' + target_dir + '/' + out_fol + '/queue/*')
cond_init_seeds = []#glob.glob('./targets/' + target_dir + '/' + out_fol + '/queue/*')

argvv = sys.argv[1:] #This is supposed to give you the name of the program you are running (.exe)
call = subprocess.check_output

startup = True




#Setting up the data
def preprocessing(out):
    print("called")
    global MAX_BITMAP_SIZE
    global MAX_FILE_SIZE
    global seed_list
    global runtime_seeds
    global cond_init_seeds
    global counter
    global label
    global u_idx
    
    seed_list = glob.glob('./targets/' + target_dir + '/' + out + '/queue/id*')
    #print(seed_list)
    max_file_name = call(['ls', '-S', './targets/' + target_dir + '/' + out + '/queue/']).decode('utf8').split('\n')[0].rstrip('\n')
    MAX_FILE_SIZE = os.path.getsize('./targets/' + target_dir + '/' + out + '/queue/' + max_file_name)
    
    cond_init_seeds.append(max_file_name)
    print(str(max_file_name) + ':' + str(MAX_FILE_SIZE)) #Run again to check this tomorrow
    
    seed_list.sort()
    SPLIT_RATIO = len(seed_list)
    rand_index = np.arange(SPLIT_RATIO)
    np.random.shuffle(seed_list)
    #cwd = os.getcwd()
    unique_values = {}

    #print(seed_list)
    
        # obtain raw bitmaps
    raw_bitmap = {}
    tmp_cnt = []
    out = ''
    for file in seed_list:
        tmp_list = []
        try:
            # append "-o tmp_file" to strip's arguments to avoid tampering tested binary.
            if argvv[0] == './strip':
                out = call(['./afl-showmap', '-q', '-e', '-o', '/dev/stdout', '-m', '512', '-t', '500'] + ['./targets/' + target_dir + '/' + target_dir] + [file] + ['-o', 'tmp_file'])
            else:
                out = call(['./afl-showmap', '-q', '-e', '-o', '/dev/stdout', '-m', '512', '-t', '500'] + ['./targets/' + target_dir + '/' + target_dir] + [file])

        except subprocess.CalledProcessError:
            print("Its a crash")
        for line in out.splitlines():
            edge = line.split(b':')[0]
            if edge not in unique_values:
                unique_values[edge] = 1
                cond_init_seeds.append(file)
            
            tmp_cnt.append(edge)
            tmp_list.append(edge)
        raw_bitmap[file] = tmp_list
    counter = Counter(tmp_cnt).most_common()

    label = [int(f[0]) for f in counter]
    bitmap = np.zeros((len(seed_list), len(label)))
    for idx, i in enumerate(seed_list):
        tmp = raw_bitmap[i]
        for j in tmp:
            if int(j) in label:
                bitmap[idx][label.index((int(j)))] = 1

    fit_bitmap, u_idx = np.unique(bitmap, axis=1, return_index=True)

    MAX_BITMAP_SIZE = fit_bitmap.shape[1]
    for idx, i in enumerate(seed_list):
        file_name = "./targets/" + target_dir + "/bitmaps/" + i.split('/')[-1]
        np.save(file_name, fit_bitmap[idx])
    
    init_seeds = np.unique(cond_init_seeds)    
    return init_seeds
        
def runtime_preprocessing(fn):
    global MAX_BITMAP_SIZE
    global MAX_FILE_SIZE
    global seed_list
    global runtime_seeds
    global counter
    global label
    global u_idx
    
    
    runtime_seeds.append('./targets/' + target_dir + '/' + fn)
    
    if os.path.getsize('./targets/' + target_dir + '/' + fn) > MAX_FILE_SIZE:
        max_file_name = fn
        MAX_FILE_SIZE = os.path.getsize('./targets/' + target_dir + '/' + f)
    
    runtime_seeds.sort()
    SPLIT_RATIO = len(runtime_seeds)
    rand_index = np.arange(SPLIT_RATIO)
    
    np.random.shuffle(runtime_seeds)
    
    #cwd = os.getcwd()
    

    #print(runtime_seeds)
    
        # obtain raw bitmaps
    raw_bitmap = {}
    tmp_cnt = []
    out = ''
    for file in runtime_seeds:
        #print(file)
        tmp_list = []
        try:
            # append "-o tmp_file" to strip's arguments to avoid tampering tested binary.
            if argvv[0] == './strip':
                out = call(['./afl-showmap', '-q', '-e', '-o', '/dev/stdout', '-m', '512', '-t', '500'] + ['./targets/' + target_dir + '/' + target_dir] + [file] + ['-o', 'tmp_file'])
            else:
                out = call(['./afl-showmap', '-q', '-e', '-o', '/dev/stdout', '-m', '512', '-t', '500'] + ['./targets/' + target_dir + '/' + target_dir] + [file])

        except subprocess.CalledProcessError:
            print("Its a crash")
        for line in out.splitlines():
            edge = line.split(b':')[0]
            tmp_cnt.append(edge)
            tmp_list.append(edge)
        raw_bitmap[file] = tmp_list
    counter = Counter(tmp_cnt).most_common()

    label = [int(f[0]) for f in counter]
    bitmap = np.zeros((len(runtime_seeds), len(label)))
    for idx, i in enumerate(runtime_seeds):
        tmp = raw_bitmap[i]
        for j in tmp:
            if int(j) in label:
                bitmap[idx][label.index((int(j)))] = 1

    fit_bitmap, u_idx = np.unique(bitmap, axis=1, return_index=True)
   # print("data dimension" + str(fit_bitmap.shape))
    #print("data dimension" + str(bitmap.shape))

    #print(fit_bitmap)    
    #print(bitmap)

    MAX_BITMAP_SIZE = fit_bitmap.shape[1]
    for idx, i in enumerate(runtime_seeds):
        file_name = "./targets/" + target_dir + "/bitmaps/" + i.split('/')[-1]
        np.save(file_name, fit_bitmap[idx])



# compute jaccard accuracy for multiple label
def jaccard(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)
    y_true = tf.round(y_true)
    y_pred = tf.round(y_pred)
    matching = tf.cast(tf.math.count_nonzero(tf.equal(y_true,y_pred)), tf.float64)
    denom = tf.cast(tf.math.add(tf.size(y_true), tf.size(y_pred)), tf.float64)
    denom2 = tf.cast(tf.math.subtract(denom, matching), tf.float64)
    return tf.cast(tf.divide(matching, denom2), tf.float64)


#prepare input into usable vectors (dehex-ifying the file and storing it as a numerical array)
def vectorize(f):
    #print(f)
    seed = np.zeros((1, MAX_FILE_SIZE))
    tmp = open(f, 'rb').read()
    ln = len(tmp)
    

    
    if ln < MAX_FILE_SIZE:
        tmp = tmp + (MAX_FILE_SIZE - ln) * b'\x00'
    seed[0] = [j for j in bytearray(tmp)]
    seed = seed.astype('float64') / 255     
    
    file_name = "./targets/" + target_dir + "/bitmaps/" + f.split('/')[-1] + ".npy"
    bitmap = np.load(file_name)
    return seed, bitmap

def vectorize_nb(f):
    seed = np.zeros((1, MAX_FILE_SIZE))
    tmp = open(f, 'rb').read()
    ln = len(tmp)
    if ln < MAX_FILE_SIZE:
        tmp = tmp + (MAX_FILE_SIZE - ln) * b'\x00'
    #print(tmp)
    seed[0] = [j for j in bytearray(tmp)]
    seed = seed.astype('float64') / 255     

    return seed

#prepare inputs in a range into usable vectors (dehex-ifying the file and storing it as a numerical array)
def vectorize_range(l, h):
    '''
    for i in range(len(seed_list)):
        print(seed_list[i] + '\n')
    
    print('-------------------------\n')
    print('-------------------------\n')
    print('-------------------------\n')        
    for i in range(len(runtime_seeds)):
        print(runtime_seeds[i] + '\n')
        
    print('-------------------------\n')
    print('-------------------------\n')
    print('-------------------------\n')    
    '''
    seeds = np.zeros((h - l, MAX_FILE_SIZE))
    bitmaps = np.zeros((h - l, MAX_BITMAP_SIZE))
    
    for i in range(l, h):
        seed, bitmap = vectorize(runtime_seeds[i])
        seeds[i-l] = seed
        bitmaps[i-l] = bitmap
    
    return seeds, bitmaps

#Splitting Data for NN
def prepare_data():
    
    X, y = [[],[]]
    for i in seed_list:
        tmp = vectorize(i)
        X.append([tmp[0]])
        y.append([tmp[1]])
        
    print(len(X))
    print(len(y))
    return X, y#train_test_split(X, y, test_size=0, random_state=1) #Want to preserve this split in case for live updates

def gen_model():
    batch_size = 32
    num_classes = MAX_BITMAP_SIZE
    epochs = 1

    mod = Sequential()
    mod.add(Dense(4096, input_dim=MAX_FILE_SIZE))
    mod.add(Activation('relu'))
    mod.add(Dense(num_classes))
    mod.add(Activation('sigmoid'))

    mod.compile(loss='binary_crossentropy', optimizer='adam', metrics=[jaccard])
    mod.summary()

    return mod

def train(mod):
   # X_train, X_val, y_train, y_val = prepare_data()
    print(MAX_FILE_SIZE)
    print(MAX_BITMAP_SIZE)
    X, y = prepare_data()
    X = np.array(X).reshape(-1, MAX_FILE_SIZE)
    y = np.array(y).reshape(-1, MAX_BITMAP_SIZE)
    print(X.shape)
    print(y.shape)
    #X_train = np.array(X_train).reshape(-1, MAX_FILE_SIZE)
    #y_train = np.array(y_train).reshape(-1, MAX_BITMAP_SIZE)
    #X_val = np.array(X_val).reshape(-1, MAX_FILE_SIZE)
    #y_val = np.array(y_val).reshape(-1, MAX_BITMAP_SIZE)
    
    es = EarlyStopping(monitor = 'val_loss', patience = 20, mode = 'min', verbose = 1)
    mc =  ModelCheckpoint('best_model.h5', monitor='val_loss', verbose=1, save_best_only=True)

    mod.summary()

    his = mod.fit(X, y, epochs = 50, batch_size = 64) #callbacks=[es, mc])
    
    #model.load_weights('best_model.h5')
    return mod, his

def build_model():
    
    print(MAX_FILE_SIZE)
    print(MAX_BITMAP_SIZE)
    
    mod, his = train(gen_model())
    
    return mod, his

def getSmallestPath():
    
    path = counter[-1]
    idx = label.index(int(path[0]))
    
    try:
        found = list(u_idx).index(idx)
    except ValueError:
        found = -1
        
    i = -2
    while(found == -1):
        path = counter[i]
        idx = label.index(int(path[0]))

        try:
            found = list(u_idx).index(idx)
        except ValueError:
            found = -1

        i -= 1
        
    return found


def getLargestPath(): 

    path = counter[0]
    idx = label.index(int(path[0]))
    try:
        found = list(u_idx).index(idx)
    except ValueError:
        found = -1
    
    i = 1
    while(found == -1):
        path = counter[i]
        idx = label.index(int(path[0]))
        try:
            found = list(u_idx).index(idx)
        except ValueError:
            found = -1
        
        i += 1
        
    return found


#Gradients of no input are removed from consideration because they are a) small and b) mean nothing to the fuzzer for mutating inputs
def compAbsIndexGradientMean(file_name, tar_index):
    
    curr_in, res = vectorize(file_name)

    #print(curr_in.shape)
    #Computing the Gradient
    #analyzer = innvestigate.create_analyzer("integrated_gradients", model, neuron_selection_mode="index")

    a = analyzer.analyze(curr_in, tar_index)
    print(a.shape)
    a = a[a != 0]
    print(a.shape)
    
    mean = np.mean(np.absolute(a))
    fname = file_name.split('/')[5]
    fname = "./targets/"  + target_dir + "/out/queue_grads/" + fname + "_" + str(tar_index) + ".txt"
    with open(fname, "w") as file:
       file.write(str(f"{mean:8f}"))
    
    return mean

#Computes average gradient of inputF
def compAbsGradientMean(file_name):

    curr_in = vectorize_nb(file_name)
    print(curr_in.flatten())
    #print(curr_in.shape)
    #Computing the Gradient
    analyzer = innvestigate.create_analyzer("integrated_gradients", model)
    
    print("created")
    with graph.as_default():
        a = analyzer.analyze(curr_in) #Its getting stuck here
        print(a)
        a = a[a != 0]
        
    mean = np.mean(np.absolute(a))
    print("Mean obtained")
    '''
    fname = file_name.split('/')[5]
    fname = "./targets/"  + target_dir + "/out/queue_grads/" + fname + ".txt"
    with open(fname, "w") as file:
       file.write(str(f"{mean:8f}"))
    '''
    return mean

def listen():
    while True:
        #data = client.recv(SIZE).decode(FORMAT)
        #try:
        data = client.recv(SIZE).decode('utf-8')
        print("received: " + data)
        if(data == 'input'):
            
            q.append(data)
        #except:
            #print("failed")
        #print(data)
        #print(data)
        if len(data) != 0 and data[-1] == "s":
            print("request_received")
            while(True):
                try:
                    out_data = send_data[data[:-1]].encode()
                    print(out_data)
                    client.send(out_data)
                    print("sent")
                    break
                except:
                    pass          
                  
def retrain(f, arg):
    global model
    global analyzer
    
    runtime_preprocessing(f)
    
    with open('targets/' + target_dir + '/' + f, 'w') as fn:
        fn.write(arg)
    
    print("written")
    
    X, y = vectorize_range(0, min(63, int(len(runtime_seeds)/16)))
    qX, qy = vectorize('targets/' + target_dir + '/' + f)
    
    X = np.append(X, qX).reshape(-1, MAX_FILE_SIZE)
    y = np.append(y, qy).reshape(-1, MAX_BITMAP_SIZE)
    
    model = gen_model()
    model.fit(X, y, epochs = 1, batch_size = min(63, int(len(runtime_seeds)/16)))                  

    with graph.as_default():
        analyzer = innvestigate.create_analyzer("integrated_gradients", model)


def compute_send():

    while True:
        #print(len(q))
        if len(q) > 0:
            print("remaining queue items: " + str(len(q)))
            #print(q) #args/BUFFER/ID/BUFFER/
            curr = q.popleft()
            curr = curr.split('/BUFFER/', 2)
            
            try:
                if len(curr[2]) != 0: #Checking if two entries got smashed into one
                    new_entry = "/BUFFER/" + curr[2]
                    q.append(new_entry)
            except:
                print(curr)
                print('unfortunate')

            seed = np.zeros((1, MAX_FILE_SIZE))

            in_s = curr[0] 
            in_s = bytes(in_s,'iso8859-1')
            
            ln = len(in_s)
            
            #print(MAX_FILE_SIZE)
            
            if ln < MAX_FILE_SIZE:
                in_s = in_s + (MAX_FILE_SIZE - ln) * b'\x00'
            seed[0] = [j for j in bytearray(in_s)]
            seed = seed.astype('float64') / 255     
            #print(seed.shape)\ 
            
            with graph.as_default():
                a = analyzer.analyze(seed)
                a = a[a != 0]
                                
            grad_data = str(np.mean(np.absolute(a)))
            send_data[curr[1]] = grad_data
            
            print("computed")
            #retrain(curr[1], curr[0]) #REMOVE THIS LINE TO RID OF INC LEARN CHANGES

cur_cycle = 0
    
dependencies = {
'jaccard': jaccard
}

init_seeds = preprocessing('out')
with graph.as_default():    
    model, his = build_model()

""" TCP Socket """
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(ADDR)
client.setblocking(1)
print("Connection Success")
i = 0

send_data = {
    
}

with graph.as_default():
    analyzer = innvestigate.create_analyzer("integrated_gradients", model)


print("waiting")

t1 = threading.Thread(target=compute_send)

t1.start()

data = client.recv(SIZE).decode(FORMAT)
print("first request: " + data)
grad_data = str(compAbsGradientMean("./targets/" + target_dir + "/out/queue/id:000000,orig:seed"))
print("computed")
first_input = client.recv(SIZE).decode(FORMAT)
print("recieved: " + first_input)
client.sendall(grad_data.encode())
print("first data sent: " + grad_data)
send_data["out_dir_v2/queue/id:000000,orig:seed"] = grad_data


for i in init_seeds:
    runtime_seeds.append(i)
    
for i in range(len(init_seeds)):
    if init_seeds[i].split('/')[0] != '.':
        init_seeds[i] = './targets/' + target_dir + '/out/queue/' + init_seeds[i]
    
for f in init_seeds:
   shutil.copy(f, './targets/' + target_dir + '/' + out_fol + '/queue/'+ f.split('/')[-1] + '_orig')



i = 0

while True:
    #Current issue, add_to_queue only sends in input args so it is difficult to decide what label gets assigned to the queue entry
    try:    
        #data = client.recv(SIZE).decode(FORMAT)
        #try:
        data = client.recv(SIZE).decode(FORMAT)
        print(data)
        if(data[:len(out_fol)] == out_fol):
            #input("Permission to send data for: " + data)
            while True:
                try:
                    out_data = send_data[data].encode(FORMAT)
                    client.sendall(out_data)
                    print("data_sent: " + str(i) + ':' + str(out_data))
                    i += 1
                    break
                    
                except KeyError:
                    print("stuck: " + data)
                    with open(data, "r", encoding=FORMAT) as fn:
                        inp = fn.read()
                        inp = inp + '/BUFFER/' + target_dir + '/queue/' + data.split('/')[-1] + '/BUFFER/'
                        #print(data)
                        q.append(inp)
                    pass
                
        elif data == "new cycle":
            keras.backend.clear_session()
            tf.reset_default_graph()

            preprocessing(out_fol)
            #prepare_data()
            with graph.as_default():
                model, his = build_model()
                analyzer = innvestigate.create_analyzer("integrated_gradients", model)
                cur_cycle += 1
                model.summary()
        
            for f in seed_list:    
                with open(f, "r", encoding=FORMAT) as fn:
                    data = fn.read()
                    data = data + '/BUFFER/' + target_dir + '/queue/' + f.split('/')[-1] + '/BUFFER/'
                    #print(data)
                    q.append(data)
                    
            print('done')
            client.sendall('done'.encode(FORMAT))

            
        else:
            entry = data
            q.append(entry)
    except KeyboardInterrupt:
        print(send_data)
        client.close()
        t1.join()
        sys.exit()
        


'''
while True:
    try:
        data = client.recv(SIZE).decode(FORMAT)
        if len(data) != 0: q.append(data)
        print(data)
    except:
        data = "none"
    #print(data)
    if len(data) != 0 and data[-1] == "s":
        print("request_received")
        client.send(send_data[data[:-1]].encode())
    if len(q) != 0:
        print(len(q))
        curr = q.popleft()
        curr = "./targets/" + target_dir + "/" + curr
        grad_data = str(compAbsGradientMean(curr))
        send_data[curr] = grad_data
'''


client.close()