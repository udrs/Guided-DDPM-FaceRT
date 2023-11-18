import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from utils.constants import *
from utils.NeuralNetwork import *
from utils.Plot import *
from utils.basic import *

input_height = 1
kernel_size = 60
depth = 60
num_hidden = 1000
learning_rate = 0.0001
training_epochs = 5

def read_data(file_path):
    column_names = ['user-id','activity','timestamp', 'x-axis', 'y-axis', 'z-axis']
    data = pd.read_csv(file_path, header = None, names = column_names)
    return data

def feature_normalize(dataset):
    mu = np.mean(dataset,axis = -1)
    sigma = np.std(dataset,axis = -1)
    mu = mu[:,:,None]
    sigma = sigma[:, :,None]
    return (dataset - mu)/sigma

# def plot_axis(ax, x, y, title):
#     ax.plot(x, y)
#     ax.set_title(title)
#     ax.xaxis.set_visible(False)
#     ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
#     ax.set_xlim([min(x), max(x)])
#     ax.grid(True)

def plot_activity2(activity,data):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows = 3, figsize = (15, 10), sharex = True)
    plot_axis(ax0, data['timestamp'], data['x-axis'], 'x-axis')
    plot_axis(ax1, data['timestamp'], data['y-axis'], 'y-axis')
    plot_axis(ax2, data['timestamp'], data['z-axis'], 'z-axis')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()

def windows(data, size):
    start = 0
    while start < data.count():
        yield int(start), int(start + size)
        start += (size / 2)

def segment_signal(data,window_size = 90):
    segments = np.empty((0,window_size,3))
    labels = np.empty((0))
    for (start, end) in windows(data["timestamp"], window_size):
        x = data["x-axis"][start:end]
        y = data["y-axis"][start:end]
        z = data["z-axis"][start:end]
        if(len(dataset["timestamp"][start:end]) == window_size):
            segments = np.vstack([segments,np.dstack([x,y,z])])
            labels = np.append(labels,stats.mode(data["activity"][start:end])[0][0])

    return segments, labels

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.0, shape = shape)
    return tf.Variable(initial)

def depthwise_conv2d(x, W):
    return tf.nn.depthwise_conv2d(x,W, [1, 1, 1, 1], padding='VALID')

def apply_depthwise_conv(x,kernel_size,num_channels,depth):
    weights = weight_variable([1, kernel_size, num_channels, depth])
    biases = bias_variable([depth * num_channels])
    return tf.nn.relu(tf.add(depthwise_conv2d(x, weights),biases))

def apply_max_pool(x,kernel_size,stride_size):
    return tf.nn.max_pool(x, ksize=[1, 1, kernel_size, 1],
                          strides=[1, 1, stride_size, 1], padding='VALID')


# dataset = read_data("/Users/zhaoleixiao/HAR/myHAR/data/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt")
# dataset.dropna(axis=0, how='any', inplace= True)
# dataset['x-axis'] = feature_normalize(dataset['x-axis'])
# dataset['y-axis'] = feature_normalize(dataset['y-axis'])
# dataset['z-axis'] = feature_normalize(dataset['z-axis'])

train_loader = data_loader('train',True)
test_loader = data_loader("test", 0, 0)
#plot_hist(train_loader.dataset.y_data)
X_train=feature_normalize(np.array(train_loader.dataset.x_data)) #(-1,9,128)
X_train=X_train.transpose((0,2,1)) #(-1,128,9)
train_x =X_train[:,None,:,:] #(-1,1,128,9)
train_y= np.asarray(pd.get_dummies(np.squeeze(np.array(train_loader.dataset.y_data))), dtype = np.float32)

X_test=feature_normalize(np.array(test_loader.dataset.x_data))
X_test=X_test.transpose((0,2,1))
test_x=X_test[:,None,:,:] #(-1,1,128,9)
test_y= np.asarray(pd.get_dummies(np.squeeze(np.array(test_loader.dataset.y_data))), dtype = np.float32)
#
# for activity in np.unique(train_y):
#     subset = X_train[train_y == activity][:180]
#     plot_activity(activity,subset)


total_batchs = X_train.shape[0] // BATCH_SIZE
X = tf.placeholder(tf.float32, shape=[None,input_height,WINDOW_WIDTH,num_channels])
Y = tf.placeholder(tf.float32, shape=[None,NUM_CLASSES])

c = apply_depthwise_conv(X,kernel_size,num_channels,depth)
p = apply_max_pool(c,20,2)
c = apply_depthwise_conv(p,6,depth*num_channels,depth//10)

shape = c.get_shape().as_list()
c_flat = tf.reshape(c, [-1, shape[1] * shape[2] * shape[3]])

f_weights_l1 = weight_variable([shape[1] * shape[2] * depth * num_channels * (depth//10), num_hidden])
f_biases_l1 = bias_variable([num_hidden])
f = tf.nn.tanh(tf.add(tf.matmul(c_flat, f_weights_l1),f_biases_l1))

out_weights = weight_variable([num_hidden, NUM_CLASSES])
out_biases = bias_variable([NUM_CLASSES])
y_ = tf.nn.softmax(tf.matmul(f, out_weights) + out_biases)

loss = -tf.reduce_sum(Y * tf.log(y_))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
with tf.Session() as session:
    tf.global_variables_initializer().run()
    for epoch in range(training_epochs):
        cost_history = np.empty(shape=[1],dtype=float)
        for b in range(total_batchs):
            offset = (b * BATCH_SIZE) % (train_y.shape[0] - BATCH_SIZE)
            batch_x = train_x[offset:(offset + BATCH_SIZE), :, :, :]
            batch_y = train_y[offset:(offset + BATCH_SIZE), :]
            _, c = session.run([optimizer, loss],feed_dict={X: batch_x, Y : batch_y})
            cost_history = np.append(cost_history,c)
        print ("Epoch: ",epoch," Training Loss: ",np.mean(cost_history)," Training Accuracy: ",
              session.run(accuracy, feed_dict={X: train_x, Y: train_y}))

    print ("Testing Accuracy:", session.run(accuracy, feed_dict={X: test_x, Y: test_y}))