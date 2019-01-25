import tensorflow as tf

def l2_squared(x, y):
    """
    Evaluate `∑|| x - y ||_{2}^{2}` 
    """
    return tf.reduce_sum(tf.square(x - y), 1)

def loss(descriptors, m = 0.01):
    diff_pos = l2_squared(descriptors[::3], descriptors[1::3])
    diff_neg = l2_squared(descriptors[::3], descriptors[2::3])
    tmp = 1 - (diff_neg / (diff_pos + m))
    L_trip = tf.reduce_sum(tf.maximum(tf.zeros_like(tmp),tmp), 0)
    L_pair = tf.reduce_sum(diff_pos)
    
    return L_trip + L_pair

def cnn_model(features, labels, mode):
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
    

    # Mode selection

    # Loss selection
    if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
        L = loss(dense2)
    else:
        L = None

    # train_op selection
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(L)
    else:
        train_op = None
    
    # Prediction selection
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = ...
    else:
        predictions = None
    
    return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=L,
      train_op=train_op)