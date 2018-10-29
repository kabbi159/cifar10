import tensorflow as tf
import numpy as np


def weight_variable(shape):
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)

def bias_variable(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)

def conv_layer(input, shape):
    W = weight_variable(shape)
    b = bias_variable([shape[3]])
    return tf.nn.relu(tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding= 'SAME') + b)

def full_layer(input, size):
    input_size = int(input.get_shape()[1])
    W = weight_variable([input_size, size])
    b = bias_variable([size])
    return tf.matmul(input, W) + b

def next_batch(num, data, labels):
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def model(x):

    conv1_1 = conv_layer(x, shape=[3, 3, 3, 30])
    conv1_2 = conv_layer(conv1_1, shape=[3, 3, 30, 30])
    conv1_3 = conv_layer(conv1_2, shape=[3, 3, 30, 30])
    conv1_pool = tf.nn.max_pool(conv1_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding= 'SAME')
    conv1_drop = tf.nn.dropout(conv1_pool, keep_prob=keep_prob)

    conv2_1 = conv_layer(conv1_drop, shape=[3, 3, 30, 50])
    conv2_2 = conv_layer(conv2_1, shape=[3, 3, 50, 50])
    conv2_3 = conv_layer(conv2_2, shape=[3, 3, 50, 50])
    conv2_pool = tf.nn.max_pool(conv2_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2_drop = tf.nn.dropout(conv2_pool, keep_prob=keep_prob)

    conv3_1 = conv_layer(conv2_drop, shape=[3, 3, 50, 80])
    conv3_2 = conv_layer(conv3_1, shape=[3, 3, 80, 80])
    conv3_3 = conv_layer(conv3_2, shape=[3, 3, 80, 80])
    conv3_pool = tf.nn.max_pool(conv3_3, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')

    conv3_flat = tf.reshape(conv3_pool, [-1, 80])
    conv3_drop = tf.nn.dropout(conv3_flat, keep_prob=keep_prob)

    full1 = tf.nn.relu(full_layer(conv3_drop, 500))
    full1_drop = tf.nn.dropout(full1, keep_prob=keep_prob)

    logits = full_layer(full1_drop, 10)
    y_pred = tf.nn.softmax(logits)

    return y_pred, logits


x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 10), axis=1)
y_test_one_hot = tf.squeeze(tf.one_hot(y_test, 10), axis=1)

y_pred, logits = model(x)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
train_step = tf.train.RMSPropOptimizer(1e-3).minimize(loss)
# train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

is_correct = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))


with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for i in range(10000):
        batch = next_batch(128, x_train, y_train_one_hot.eval())

        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
            loss_print = loss.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})

            print("Epoch: %d, Training Accuracy: %f, Loss: %f" % (i, train_accuracy, loss_print))

        sess.run(train_step, feed_dict={x: batch[0], y: batch[1], keep_prob: 0.7})

    test_accuracy = 0.0
    for i in range(10):
        test_batch = next_batch(1000, x_test, y_test_one_hot.eval())
        test_accuracy = test_accuracy + accuracy.eval(feed_dict={x: test_batch[0], y: test_batch[1], keep_prob: 1.0})
    test_accuracy = test_accuracy / 10;
    print("Test Accuracy: %f" % test_accuracy)



