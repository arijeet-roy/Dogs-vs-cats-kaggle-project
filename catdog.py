import tensorflow as tf
import pandas as pd
import numpy as np
import dataset
import random
import cnn_lib as cnn
import time

from datetime import timedelta

################## Label information ##################
classes = ['dogs', 'cats']
num_classes = len(classes)
early_stopping = None # use None if you do not want to implement early stop

####################################################################
############# Gaussian Noise with 0.01 variance   ##################
############# and normalized mean over all pixels ##################
#############            Image train_path         ##################
train_path = '/Users/kwanghoonan/Documents/MLProject/catdog/noisytrain1/'
####################################################################

#train_path = '/Users/kwanghoonan/Documents/MLProject/catdog/smalltrain/'

################## HyperParameters ##################
##### Convolutional Layer 1.
filter_size1 = 5
num_filters1 = 32
##### Convolutional Layer 2.
filter_size2 = 5
num_filters2 = 32
##### Convolutional Layer 3.
filter_size3 = 5
num_filters3 = 64
##### Convolutional Layer 3.
filter_size4 = 5
num_filters4 = 64
##### Fully-connected layer.
fc_size = 128
fc_size2 = 128
##### Number of color channels for the images: 1 channel for gray-scale.
num_channels = 3
img_size = 128
img_size_flat = img_size * img_size * num_channels
img_shape = (img_size, img_size)
batch_size = 500
#batch_size = 2
# validation split
validation_size = .2
dropRate = 0.5
# how long to wait after validation loss stops improving before terminating
############################################################################################
############################################################################################


############################################################################################
################## Retrieve training image sets and Load it in the memory ##################
data = dataset.read_train_sets(train_path, img_size, classes, validation_size)

print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Validation-set:\t{}".format(len(data.valid.labels)))

images, cls_true = data.train.images, data.train.cls
############################################################################################
############################################################################################


################ Create variables for tensorflows ################
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)
##################################################################
##################################################################

################ Convolutional Neural Network Layers ################
layer_conv1, weights_conv1 = cnn.new_conv_layer(input = x_image, num_input_channels=num_channels, filter_size=filter_size1, num_filters=num_filters1, use_pooling=True)
layer_conv2, weights_conv2 = cnn.new_conv_layer(input = layer_conv1, num_input_channels = num_filters1, filter_size=filter_size2, num_filters=num_filters2, use_pooling=True)
layer_conv3, weights_conv3 = cnn.new_conv_layer(input = layer_conv2, num_input_channels = num_filters2, filter_size=filter_size3, num_filters=num_filters3, use_pooling=True)
layer_conv4, weights_conv4 = cnn.new_conv_layer(input = layer_conv3, num_input_channels = num_filters3, filter_size=filter_size4, num_filters=num_filters4, use_pooling=True)
#####################################################################

print(x_image)
print(layer_conv1)
print(layer_conv2)
print(layer_conv3)
print(layer_conv4)
################ First Fully Connected Layer ################
layer_flat, num_features = cnn.flatten_layer(layer_conv4)

layer_fc1 = cnn.new_fc_layer(input=layer_flat, num_inputs=num_features, num_outputs=fc_size, use_relu=True)
dropProb = tf.placeholder(tf.float32)
layer_fc1_dropout = tf.nn.dropout(layer_fc1, dropProb)
#####################################################################
#####################################################################

################ Second Fully Connected Layer ################
###### This layer Will be connected to Softmax function ######
layer_fc2 = cnn.new_fc_layer(input=layer_fc1_dropout, num_inputs=fc_size, num_outputs=fc_size2, use_relu=True)
layer_fc2_dropout = tf.nn.dropout(layer_fc2, dropProb)

########### What is most likely probability in vector? - cats or dogs?
weights = cnn.new_weights(shape=[fc_size2, num_classes])
biases = cnn.new_biases(length=num_classes)
    
y_pred = tf.matmul(layer_fc2_dropout, weights) + biases
y_pred_cls = tf.argmax(y_pred, dimension=1)
############################################################################################
############################################################################################

########################### Loss function  & Optimizer & Session ###########################
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session()
session.run(tf.global_variables_initializer())
############################################################################################
############################################################################################

train_batch_size = batch_size

cnn.optimize(num_iterations=900, train_batch_size = train_batch_size, data = data, x=x, y_true=y_true, img_size_flat=img_size_flat, session = session, optimizer = optimizer, accuracy = accuracy, cost=cost,dropProb = dropProb,prob = dropRate)


wrong_image, correct_image = cnn.print_validation_accuracy(data = data, x = x, y_true = y_true, session = session, y_pred_cls = y_pred_cls, validation_batch_size = batch_size, img_size_flat=img_size_flat, dropProb = dropProb,show_example_errors=True, show_examples_correct=True,show_confusion_matrix=False)

#cnn.plot_image(wrong_image)
#cnn.plot_image(correct_image)

#cnn.plot_conv_layer(layer=layer_conv1, image=wrong_image, x=x,session = session)
#cnn.plot_conv_layer(layer=layer_conv2, image=wrong_image, x=x,session = session)
#cnn.plot_conv_layer(layer=layer_conv1, image=correct_image, x=x,session = session)
#cnn.plot_conv_layer(layer=layer_conv2, image=correct_image, x=x,session = session)






