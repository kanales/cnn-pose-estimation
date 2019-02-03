import tensorflow as tf
import numpy as np
import random
import os

from sklearn.neighbors import KNeighborsClassifier

CACHE_DIR = 'cache/'
MODEL_PATH = os.path.join(CACHE_DIR,'cnn_model')

def l2_squared(x, y):
    """
    Evaluate `∑|| x - y ||_{2}^{2}` 
    """
    return tf.reduce_sum(tf.square(x - y), 1)

def loss(descriptors, m = 0.01):
    diff_pos = l2_squared(descriptors[::3], descriptors[1::3])
    diff_neg = l2_squared(descriptors[::3], descriptors[2::3])
    tmp = 1 - (diff_neg / (diff_pos + m))
    L_trip = tf.reduce_sum(tf.maximum(tf.zeros_like(tmp),tmp), 0, name='triplet_loss')
    L_pair = tf.reduce_sum(diff_pos, name='pair_loss')
    
    return (L_trip, L_pair)

def similarity(q1, q2):
    """
    Returns a measure of similarity between two quaternions
    """
    return 2 * np.arccos(min(1,np.abs(q1 @ q2)))

def get_triplets(Sdb, Strain):
    """
    Generates a batch of `n` elements
    """
    N = len(Strain)
    def gen():
        anchors = random.sample(Strain, N)
        for anchor in anchors:
            # Puller: select most similar from Sdb
            puller = min( (x for x in Sdb if x.cls == anchor.cls)
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

def get_batch(Sdb, Strain, verbose = True):
    """
    Generates a batch of `n` elements
    """
    def gen():
        anchors = Strain[:]
        while True:
            if verbose: print('Shuffling dataset')
            random.shuffle(anchors)
            
            for anchor in anchors:
                # Anchor: select random sample from Strain
                #anchor = random.choice(Strain)
                # Puller: select most similar from Sdb
                puller = min( (x for x in Sdb if x.cls == anchor.cls)
                            , key = lambda x: similarity(x.quat,anchor.quat))
                # Pusher: same object different pose | random different object 
                pusher = random.choice([x for x in Sdb if (x.cls != anchor.cls) or (x.cls == anchor.cls and x.idx != anchor.idx) ])
                yield (anchor.img, anchor.cls)
                yield (puller.img, puller.cls)
                yield (pusher.img, pusher.cls)
    return gen

def _cnn_model_fn(features, labels, mode):
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

    # Prediction selection
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = dense2 # ?

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions)

    # Loss selection
    if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
        Lt, Lp = loss(dense2)
        L = tf.add(Lt, Lp, name='full_loss')
        tf.summary.scalar('loss',L)

    # train_op selection
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(
            loss=L,
            global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=L,
            train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        # Sdb_img = np.array([x.img for x in dataset['Sdb']])

        # Sdb_descriptors = list(model.predict(input_fn=eval_input_fn))
        # Stest_img = np.array([x.img for x in dataset['Stest']])

        # Stest_descriptors = list(model.predict(input_fn=eval_input_fn))

        eval_metric_ops = {
             "histogram": []
        }
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=L,
            eval_metric_ops=eval_metric_ops)

# train input
def gen_train_input_fn(Sdb, Strain,batch_size):
    #Sdb    = features['Sdb']
    #Strain = features['Strain']
    #dataset = tf.data.Dataset.from_tensoxr_slices(tf.convert_to_tensor(batch(Sdb,Strain,batch_size)))
    assert batch_size % 3 == 0
    def inner():
        #b = batch(Sdb,Strain,batch_size)
        #dataset = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(b))
        return tf.data.Dataset.from_generator(
            get_batch(Sdb, Strain),
            output_types=(tf.float32,tf.int8),
            output_shapes=(tf.TensorShape([64, 64, 3]),tf.TensorShape([]))
        ).batch(batch_size)
    return inner


def gen_eval_input_fn(Sdb, Stest):
    return lambda:0

def get_model(Dataset):
    return tf.estimator.Estimator(
        model_fn=_cnn_model_fn, model_dir=MODEL_PATH)

def eval_model(model, Sdb, Stest):
    """
    Evaluates model for a given Sdb and Stest
    """
    Sdb_img = np.array([x.img for x in Sdb])
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        Sdb_img,
        shuffle=False,
    )

    Sdb_descriptors = list(model.predict(input_fn=eval_input_fn))
    Stest_img = np.array([x.img for x in Stest])
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        Stest_img,
        shuffle=False,
    )

    Stest_descriptors = list(model.predict(input_fn=eval_input_fn))
    return Sdb_descriptors, Stest_descriptors

def get_hist(model, Sdb, Stest):
    neigh = KNeighborsClassifier(n_neighbors=1)

    Sdb_descriptors, Stest_descriptors = eval_model(model, Sdb, Stest)
    X, y = Sdb_descriptors, [x.cls for x in Sdb]
    neigh.fit(X,y)

    idxs = neigh.kneighbors(Stest_descriptors)[1]

    hist = np.array([0,0,0,0])
    N = len(Stest_descriptors)
    for i in range(N):
        db = Sdb[idxs[i][0]]
        test = Stest[i]

        if db.cls != test.cls:
            continue

        theta = np.rad2deg(similarity(db.quat, test.quat))
        if theta < 10:
            hist[0] += 1
        if theta < 20:
            hist[1] += 1
        if theta < 40:
            hist[2] += 1
        hist[3] += 1
    return hist
