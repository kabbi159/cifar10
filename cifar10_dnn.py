import tensorflow as tf
import numpy as np


def weight_variable(shape):
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)

def bias_variable(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)

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

    flat_inputs = tf.contrib.layers.flatten(x)
    full1 = full_layer(flat_inputs, 28 * 28 * 3)
    full1_drop = tf.nn.dropout(full1, keep_prob=keep_prob)

    full2 = full_layer(full1_drop, 20 * 20 * 3)
    full2_drop = tf.nn.dropout(full2, keep_prob=keep_prob)

    full3 = full_layer(full2_drop, 16 * 16 * 3)
    full3_drop = tf.nn.dropout(full3, keep_prob=keep_prob)

    full4 = full_layer(full3_drop, 10)
    y_pred = tf.nn.softmax(full4)

    return y_pred, full4


x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 10), axis=1)
y_test_one_hot = tf.squeeze(tf.one_hot(y_test, 10), axis=1)

y_pred, logits = model(x)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
train_step = tf.train.RMSPropOptimizer(1e-3).minimize(loss)

is_correct = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))


with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for i in range(30000): # 10000, 30000, 50000
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


# training step 10000: test acc - 0.2014
# training step 30000: test acc - 0.2803
# training step 50000: test acc - 0.1962
