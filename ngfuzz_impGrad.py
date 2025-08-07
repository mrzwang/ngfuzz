

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

import threading

import shutil

graph = tf.get_default_graph()

q = deque()
idx_on = []

target_dir = sys.argv[1] #Make this customizable for the future
out_fol = sys.argv[2]

retrain_dir = "./targets/" + target_dir + "/" + out_fol + "/update_queue"

IP = "127.0.0.1"
PORT = 4455
ADDR = (IP, PORT)
SIZE = 9048
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
    global MAX_BITMAP_SIZE
    global MAX_FILE_SIZE
    global seed_list
    global runtime_seeds
    global cond_init_seeds
    global counter
    global label
    global u_idx
    
    seed_list = glob.glob('./targets/' + target_dir + '/' + out + '/queue/id*')
    max_file_name = call(['ls', '-S', './targets/' + target_dir + '/' + out + '/queue/']).decode(FORMAT).split('\n')[0].rstrip('\n')
    MAX_FILE_SIZE = os.path.getsize('./targets/' + target_dir + '/' + out + '/queue/' + max_file_name)
    
    cond_init_seeds.append(max_file_name)
    
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

    model = Sequential()
    model.add(Dense(4096, input_dim=MAX_FILE_SIZE))
    model.add(Activation('relu'))
    model.add(Dense(num_classes))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[jaccard])
    model.summary()

    return model

def train(model):
   # X_train, X_val, y_train, y_val = prepare_data()
    X, y = prepare_data()
    X = np.array(X).reshape(-1, MAX_FILE_SIZE)
    y = np.array(y).reshape(-1, MAX_BITMAP_SIZE)
    #X_train = np.array(X_train).reshape(-1, MAX_FILE_SIZE)
    #y_train = np.array(y_train).reshape(-1, MAX_BITMAP_SIZE)
    #X_val = np.array(X_val).reshape(-1, MAX_FILE_SIZE)
    #y_val = np.array(y_val).reshape(-1, MAX_BITMAP_SIZE)
    
    #es = EarlyStopping(monitor = 'val_loss', patience = 20, mode = 'min', verbose = 1)
    #mc =  ModelCheckpoint('best_model.h5', monitor='val_loss', verbose=1, save_best_only=True)

    his = model.fit(X, y, epochs = 5, batch_size = 64) #callbacks=[es, mc])
    
    #model.load_weights('best_model.h5')
    return model, his

def build_model():
    
    model = gen_model()
    his = train(model)
    
    return model, his

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

def getSmallestPaths(fn, start, num_paths): #Changed input_vec to fn so that we can get the bitmap
    cnt = 0
    num_found = 0
    idxs = []
    path = counter[-1]
    #print(path)
    #print(u_idx)
    idx = label.index(int(path[0]))
    #print(idx)
    
    file_name = "./targets/" + target_dir + "/bitmaps/" + fn.split('/')[-1] + ".npy"
    bitmap = np.load(file_name)
    
    try:
        found = list(u_idx).index(idx)
        if(bitmap[found] == 0 and start < cnt):
            idxs.append(found)
            cnt += 1

    except ValueError:
        pass
        
    i = -2
    while(cnt < num_paths):
        #print(i)
        path = counter[i]
        idx = label.index(int(path[0]))
        try:
            found = list(u_idx).index(idx)
            if found != -1 and bitmap[found] == 0 and start <= num_found:
                idxs.append(found)
                cnt += 1
            if found != -1:
                num_found += 1

        except ValueError:
            pass

        #Figure out an alternative method for here because .index crashes the program if not found
        i -= 1
        
        if i <= -len(counter) or cnt >= num_paths:
            break
        #Add condition to break if i becomes greater than the size of the counter before num_paths is fufilled        
    return idxs


#Gradients of no input are removed from consideration because they are a) small and b) mean nothing to the fuzzer for mutating inputs
def compAbsIndexGradientMean(file_name, tar_index):
    
    curr_in, res = vectorize(file_name)

    #print(curr_in.shape)
    #Computing the Gradient
    analyzer = innvestigate.create_analyzer("integrated_gradients", model, neuron_selection_mode="index")
    
    with graph.as_default():
        a = analyzer.analyze(curr_in, tar_index)
        a = a[a != 0]
        
    mean = np.mean(np.absolute(a))
    fname = file_name.split('/')[5]
    fname = "./targets/"  + target_dir + "/out/queue_grads/" + fname + "_" + str(tar_index) + ".txt"
    with open(fname, "w") as file:
       file.write(str(f"{mean:8f}"))
    
    return mean

#Computes average gradient of input
def compAbsGradientMean(file_name):

    curr_in = vectorize_nb(file_name)
    #print(curr_in.shape)
    #Computing the Gradient
    #analyzer = innvestigate.create_analyzer("integrated_gradients", model)
    with graph.as_default():
        a = analyzer.analyze(curr_in)
        a = a[a != 0]
        
    mean = np.mean(np.absolute(a))
    '''
    fname = file_name.split('/')[5]
    fname = "./targets/"  + target_dir + "/out/queue_grads/" + fname + ".txt"
    with open(fname, "w") as file:
       file.write(str(f"{mean:8f}"))
    '''
    return mean

def compAbsGradient(file_name):

    curr_in = vectorize_nb(file_name)
    #print(curr_in.shape)
    #Computing the Gradient
    #analyzer = innvestigate.create_analyzer("integrated_gradients", model)
    with graph.as_default():
        a = analyzer.analyze(curr_in)
        a = a[a != 0]
        
    unsigned_a = np.absolute(a)
    grad_data = str(np.mean(np.absolute(a))) + '/'
    order = np.argsort(a)
    
    for i in unsigned_a:
        grad_data += str(i) + '/'

    grad_order = ""
    
    for i in order:
        grad_order += str(i) + '/'
        
                
    send_data[out_fol + "/queue/id:000000,orig:seed"] = [grad_data, grad_order]  #First entry is the average gradient, the rest are individual deliminated by '/'
        
    return np.mean(np.absolute(a))



# compute jaccard accuracy for multiple label
def adv_jaccard(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)
    y_true = tf.reshape(tf.round(y_true), [-1])
    y_pred = tf.reshape(tf.round(y_pred), [-1])
    
    y_pred = tf.gather(y_pred, indices=idx_on)
    y_true = tf.gather(y_true, indices=idx_on)
    
    matching = tf.cast(tf.math.count_nonzero(tf.equal(y_true,y_pred)), tf.float64)
    #print(matching)
    denom = tf.cast(tf.math.add(tf.size(y_true), tf.size(y_pred)), tf.float64)
    #print("mean")
    denom2 = tf.cast(tf.math.subtract(denom, matching), tf.float64)
    #print(K.mean(tf.cast(tf.divide(matching, denom2), tf.float64)))
    return tf.cast(tf.divide(matching, denom2), tf.float64)  
    
def adv_BinaryCrossEntropy(y_true, y_pred): 
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    y_pred = tf.gather(y_pred, indices=idx_on)
    y_true = tf.gather(y_true, indices=idx_on)
    
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    term_0 = (1 - y_true) * K.log(1 - y_pred + K.epsilon())  
    term_1 = y_true * K.log(y_pred + K.epsilon())
    return -K.mean(term_0 + term_1, axis=0)



#function for generating an adversarial example given a base p_input, adversarial class target, classifier, and regularization type
def generate_adversary(p_in,target_vector,model,regularization,loss_function):
    
    #input for base p_input
    p_input = Input(shape=(1, MAX_FILE_SIZE),name='p_input')#Consider trying to reshape each of these to its transpose (MAX_FILE_SIZE, 1)

    #unit input for adversarial noise
    one = Input(shape=(1,), name='unity')
    print("ran")
    #layer for learning adversarial noise to apply to p_input
    noise = Dense(MAX_FILE_SIZE,activation = None,use_bias=False,kernel_initializer='random_normal',
                  kernel_regularizer=regularization, name='adversarial_noise')(one)
    
    #reshape noise in shape of p_input
    noise = Reshape((1, MAX_FILE_SIZE),name='reshape')(noise)
    
    #add noise to p_input
    net = Add(name='add')([noise,p_input])
    #clip values to be within 0.0 and 1.0
    net = Activation('clip', name='clip_values')(net)
    
    #feed adversarial p_input to trained model
    outputs = model(net)

    adversarial_model = Model(inputs=[p_input,one], outputs=outputs)
    #freeze trained MNIST classifier layers
    adversarial_model.layers[-1].trainable = False
    print("ran2")
    adversarial_model.compile(optimizer='adam', loss=loss_function, metrics=[adv_jaccard]) #Categorical is most likely not great for this purpose

    #callback for saving weights with smallest loss
    checkpoint = ModelCheckpoint('./targets/' + target_dir + '/checkpoint/adversarial_weights.h5', monitor='loss', verbose=0, save_best_only=True, save_weights_only=True,
                                 mode='auto', period=1)
    print("ran3")
    print(adversarial_model.summary())
    #train adversarial p_input
    adversarial_model.fit(x=[p_in.reshape(-1, 1, MAX_FILE_SIZE),np.ones(shape=(1,1))],y=target_vector.reshape(-1, 1, MAX_BITMAP_SIZE),epochs=5000,verbose=1, callbacks=[checkpoint])
    print("ran4")
    #restore best weights
    adversarial_model.load_weights('./targets/' + target_dir + '/checkpoint/adversarial_weights.h5')
    
    #quantize adversarial noise
    quantized_weights = np.round(adversarial_model.get_weights()[0].reshape((1, MAX_FILE_SIZE)) * 255.) / 255.
    
    #add trained weights to original p_input and clip values to produce adversarial p_input
    adversarial_p_in = np.clip(p_in.reshape((1,MAX_FILE_SIZE)) + quantized_weights, 0., 1.)
    
    #display adversarial p_input
    #print(p_in)
    #print(adversarial_p_in) #check what kind of p_input is fed into the network
    #plt.imshow(adversarial_p_in,vmin=0., vmax=1.)
    #plt.show()
    #classify adversarial p_input
    #adversarial_prediction = fmodel.predict(adversarial_p_in.reshape((1,28,28,1)))
    #print(adversarial_prediction)
    
    return adversarial_p_in

#custom activation function for keeping adversarial values between 0.0 and 1.0
def clip(x):
    return K.clip(x, 0.0, 1.0)


def mut_adv(fn):
    '''
    Generates an aversarial example that targets less visited nodes and feeds it as input for the fuzzer to use as a seed
    '''
    #First Create a Target Vector of Values
    path = counter[0]
    idx = label.index(int(path[0]))
    zeros = []
    ones = []
    try:
        zeros.append(list(u_idx).index(idx))
    except ValueError:
        pass
    
    i = 1
    #First half of paths that are most frequented are set to 0 and second half are set to 1
    while(i <= len(counter)/2): #Thought to perhaps disclude nodes from items if execution path appears in every single generated case
        path = counter[i]
        idx = label.index(int(path[0]))
        try:
            zeros.append(list(u_idx).index(idx))
        except ValueError:
            pass
            
        #Figure out an alternative method for here because .index crashes the program if not found
        i += 1

    while(i < len(counter)):
        path = counter[i]
        idx = label.index(int(path[0]))
        try:
            ones.append(list(u_idx).index(idx))
        except ValueError:
            pass

        i += 1
        
    
    target_vector = np.zeros(MAX_BITMAP_SIZE)
    target_idx = getSmallestPaths(fn, 0, 10)
    
    for i in target_idx:
        target_vector[i] = 1

    #add custom objects to dictionary
    get_custom_objects().update({'clip': clip})
    #print(target_vector)
    #print(target_vector.reshape(1, -1))


    p_in = np.random.normal(.5, .3, (1, MAX_FILE_SIZE))
    p_in,bit = vectorize(fn)

    
    adv_jaccard(model.predict(p_in).flatten(), target_vector)
    
    #Instead of using a random vector pick one from a previous set that maximizes the gradient?
    
    print(p_in)
    #generate_adversary(img,9,model,l1(0.01),'categorical_crossentropy')
    
   # with CustomObjectScope({'clip': Activation(clip)}):
    adv = generate_adversary(p_in,target_vector,model,l2(0.001),adv_BinaryCrossEntropy) #greater the value the more subtle the noise
    #pred(generate_adversary(img,9,model,l1_l2(l1=0.01,l2=0.01),'categorical_crossentropy')
    
    return adv, target_vector
    
#Run with adv_in, target_vector = mut_adv(seed_list[0])

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
                    #print(out_data)
                    client.send(out_data)
                    #print("sent")
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
            #print("remaining queue items: " + str(len(q)))
            #print(q) #args/BUFFER/ID/BUFFER/
            curr = q.popleft()
            #print(curr)
            curr = curr.split('/BUFFER/', 2)
            
            if len(curr[2]) != 0: #Checking if two entries got smashed into one
                #print(curr)
                new_entry = curr[2]
                q.append(new_entry)
                #print(new_entry)
                

            seed = np.zeros((1, MAX_FILE_SIZE))

            in_s = curr[0] 
            #print(in_s)
            in_s = bytes(in_s, FORMAT)
            
            ln = len(in_s)
            
            #print(MAX_FILE_SIZE)
            
            if ln < MAX_FILE_SIZE:
                in_s = in_s + (MAX_FILE_SIZE - ln) * b'\x00'
            seed[0] = [j for j in bytearray(in_s)]
            seed = seed.astype('float64') / 255     
            #print(seed.shape)    

            with graph.as_default():
                a = analyzer.analyze(seed)
                a = a[a != 0]

            #print(a)
                
            unsigned_a = np.absolute(a)
            grad_data = str(np.mean(np.absolute(a))) + '/'
            order = np.argsort(a)
            
            for i in a:
                grad_data += str(i) + '/'
 
            grad_order = ""
            
            for i in order:
                grad_order += str(i) + '/'
                
                        
            send_data[curr[1]] = [grad_data, grad_order]  #First entry is the average gradient, the rest are individual deliminated by '/'
            #print(curr[1])
           ##print(a)
           #print("computed")
            #retrain(curr[1], curr[0]) #REMOVE THIS LINE TO RID OF INC LEARN CHANGES

curr_case = ""
    
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

#data = client.recv(SIZE).decode(FORMAT)
#print("first request: " + data)
grad_data = str(compAbsGradient("./targets/" + target_dir + "/out/queue/id:000000,orig:seed"))
first_input = client.recv(SIZE).decode(FORMAT)
print("received: " + first_input)

first_input = client.recv(SIZE).decode(FORMAT)
print("received" + first_input)

client.sendall('confirmed'.encode(FORMAT))

first_input = client.recv(SIZE).decode(FORMAT)
print("received" + first_input)

client.sendall(send_data[out_fol + "/queue/id:000000,orig:seed"][0].encode(FORMAT))
print("sent" + str(send_data[out_fol + "/queue/id:000000,orig:seed"][0]))

first_input = client.recv(SIZE).decode(FORMAT)
print("received" + first_input)

client.sendall(send_data[out_fol + "/queue/id:000000,orig:seed"][1].encode(FORMAT))
print("sent" + str(send_data[out_fol + "/queue/id:000000,orig:seed"][1]))
#print("first data sent: " + grad_data)
#send_data["out_dir_v2/queue/id:000000,orig:seed"] = grad_data

for i in init_seeds:
    runtime_seeds.append(i)
    
for i in range(len(init_seeds)):
    if init_seeds[i].split('/')[0] != '.':
        init_seeds[i] = './targets/' + target_dir + '/out/queue/' + init_seeds[i]
    
for f in init_seeds:
   shutil.copy(f, './targets/' + target_dir + '/' + out_fol + '/queue/'+ f.split('/')[-1] + '_orig')
   


t1 = threading.Thread(target=compute_send)


t1.start()
i = 0

while True:
    #Current issue, add_to_queue only sends in input args so it is difficult to decide what label gets assigned to the queue entry
    try:    
        #data = client.recv(SIZE).decode(FORMAT)
        #try:
        data = client.recv(SIZE).decode(FORMAT)
        #print(data)
        if(data[:len(out_fol)] == out_fol):
            print(data)
            while True:
                try:
                    #print("prepping payload")
                    out_data = send_data[data][0].encode(FORMAT)
                    client.sendall(out_data)
                    print("data_sent pt 1: " + str(out_data))
                    halt = client.recv(SIZE).decode(FORMAT)
                    out_data = send_data[data][1].encode(FORMAT)
                    #print("sent2")
                    client.sendall(out_data)
                    print("data_sent pt 2: " + str(out_data))
                    i += 1
                    curr_case = data
                    break
                    
                except KeyError:
                    #print("stuck: " + data)+
                    pass
                
        elif data == "new cycle":
            tf.reset_default_graph()
            preprocessing(out_fol)
            with graph.as_default():
                model, his = build_model()
                analyzer = innvestigate.create_analyzer("integrated_gradients", model)
            
            for f in seed_list:    
                with open(f, "r", encoding=FORMAT) as fn:
                    data = fn.read()
                    data = data + '/BUFFER/' + out_fol + '/queue/' + f.split('/')[-1] + '/BUFFER/'
                    #print(data)
                    q.append(data)
                    
            #print('done')
            client.sendall('done'.encode(FORMAT))

            
        else:
            entry = data
            q.append(entry)
    except KeyboardInterrupt:
        print(data)
        print(curr_case)
        for i in send_data:
            print(i)
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