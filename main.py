#! /anaconda3/envs/ex3/bin/python3

import numpy as np
import tensorflow as tf
import os

from shutil import rmtree
from data_preparation import read_images
from model import similarity, get_model, eval_model, batch, MODEL_PATH, CACHE_DIR

def main():
    tf.logging.set_verbosity(tf.logging.ERROR)
    print('Loading data...', end='')
    N_CLASSES = len(os.listdir('dataset/coarse/'))

    coarse = read_images('dataset/coarse/')
    fine = read_images('dataset/fine/')
    real = read_images('dataset/real/')
    train_idxs = ()
    with open('dataset/real/training_split.txt') as f:
        train_idxs = set(int(x) for x in f.read().split(', '))
    test_idxs = set(range(len(real))) - train_idxs
    Dataset = {
        'Sdb'    : coarse,
        'Strain' : fine + [real[i] for i in train_idxs],
        'Stest'  : [real[i] for i in test_idxs],
    }
   
    print('\rData loaded          ')

 
    rmtree(CACHE_DIR)
    os.makedirs(CACHE_DIR)
    os.makedirs(MODEL_PATH)

    print('Preparing model...', end='')
    cnn_descriptor = get_model()
    hists = []
    print('\rModel trained.')
    print('Executing model')
    with open('hists.csv', 'w+') as f:
        f.write('<10,<20,<40,<180\n')
        for h in hists:
            f.write(','.join(map(str,h)) + '\n')
    
    for x in range(10,100,10):
        print('\rIteration {}/{}    '.format(x,1000), end='')
        cnn_descriptor.train(
            input_fn=gen_train_input_fn(Dataset,300),
            max_steps=x)

        hist = get_hist(cnn_descriptor, Dataset)
        print(hist)
        with open('hists.csv', 'a+') as f:
            f.write(','.join(map(str,hist)) + '\n')
    
    with open('hists.csv', 'w+') as f:
        f.write('<10,<20,<40,<180')
        for h in hists:
            f.write(','.join(map(str,h)) + '\n')

if __name__ == '__main__':
    main()