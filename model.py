import tensorflow as tf
import numpy as np
import random
import os

CACHE_DIR = 'cache/'
MODEL_PATH = os.path.join(CACHE_DIR,'cnn_model')

def l2_squared(x, y):
    """
    Evaluate `∑|| x - y ||_{2}^{2}` 
    """
    return tf.reduce_sum(tf.square(x - y), 1)

def loss(descriptors, m = 0.01):
    print(descriptors.shape)
    diff_pos = l2_squared(descriptors[::3], descriptors[1::3])
    diff_neg = l2_squared(descriptors[::3], descriptors[2::3])
    tmp = 1 - (diff_neg / (diff_pos + m))
    L_trip = tf.reduce_sum(tf.maximum(tf.zeros_like(tmp),tmp), 0)
    L_pair = tf.reduce_sum(diff_pos)
    
    return L_trip + L_pair

def similarity(q1, q2):
    """
    Returns a measure of similarity between two quaternions
    """
    return 2 * np.arccos(min(1,np.abs(q1 @ q2)))

def batch(dataset):
    """
    Generates a batch of `n` elements
    """
    Strain, Sdb = dataset['Strain'], dataset['Sdb']
    def gen():
        while True:
            # Anchor: select random sample from Strain
            anchor = random.choice(Strain)
            # Puller: select most similar from Sdb
            puller = max( (x for x in Sdb if x.cls == anchor.cls)
                        , key = lambda x: similarity(x.quat,anchor.quat))
            # Pusher: same object different pose | random different object 
            if bool(random.randint(0,1)):
                pusher = random.choice([x for x in Sdb if x.cls != anchor.cls])
            else:
                pusher = random.choice([x for x in Sdb 
                if x.cls == anchor.cls and x.idx != anchor.idx])
            yield anchor.img
            yield puller.img
            yield pusher.img
    return gen

def _cnn_model_fn(features, mode):
    input_layer = tf.convert_to_tensor(features)
    C = features.shape[-1]
    conv1 = tf.layers.conv2d(
        inputs = input_layer,
        filters = 16,
        kernel_size = [8,8],
        activation = tf.nn.relu
    )

    pool1 = tf.layers.max_pooling2d(
        inputs = conv1,
        pool_size = [2,2],
        strides = 2
    )

    conv2 = tf.layers.conv2d(
        inputs = pool1,
        filters = 7,
        kernel_size = [5,5],
        activation = tf.nn.relu
    )
    pool2 = tf.layers.max_pooling2d(
        inputs = conv2,
        pool_size = [2,2],
        strides = 2
    )

    pool2_flat = tf.reshape(pool2, [-1, 12*12*7])
    dense1 = tf.layers.dense(
        inputs = pool2_flat,
        units = 256,
        activation = tf.nn.relu
    )

    dense2 = tf.layers.dense(
        inputs = dense1,
        units = 16,
        activation = tf.nn.relu
    )
    
    ## Mode selection
    # Loss selection
    if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
        L = loss(dense2)
        tf.summary.scalar('loss', L)
    else:
        L = None

    # train_op selection
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(
            loss=L,
            global_step=tf.train.get_global_step())
    else:
        train_op = None
    
    # Prediction selection
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = dense2 # ?
    else:
        predictions = None
    
    return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=L,
      train_op=train_op)

def get_model():
    return tf.estimator.Estimator(
        model_fn=_cnn_model_fn, model_dir=MODEL_PATH)

def eval_model(model, dataset):
    """
    Evaluates model for a given Sdb and Stest
    """
    Sdb_img = np.array([x.img for x in dataset['Sdb']])
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        Sdb_img,
        shuffle=False,
    )

    Sdb_descriptors = list(model.predict(input_fn=eval_input_fn))
    Stest_img = np.array([x.img for x in dataset['Stest']])
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        Stest_img,
        shuffle=False,
    )

    Stest_descriptors = list(model.predict(input_fn=eval_input_fn))
    return Sdb_descriptors, Stest_descriptors