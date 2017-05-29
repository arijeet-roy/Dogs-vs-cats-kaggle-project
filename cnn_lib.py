# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import random



early_stopping = None # use None if you do not want to implement early stop

img_size = 128
img_shape = (img_size, img_size)
classes = ['dogs', 'cats']
num_classes = len(classes)
num_channels = 3

img_size_flat = img_size * img_size * num_channels
###############################################
# Plot image
###############################################
def plot_images(images, cls_true,img_size,cls_pred=None):
    
    print("plot_images called")
    random_indices = random.sample(range(len(images)), min(len(images), 9))
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    
    images, cls_true = zip(*[(images[i], cls_true[i]) for i in random_indices])

    for i, ax in enumerate(axes.flat):
        # Plot image.

        ax.imshow(images[i].reshape(img_size, img_size, num_channels))

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
###############################################    


###############################################
# Weight, Biase Creation
###############################################

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


def new_conv_layer(input,
                   num_input_channels,
                   filter_size,
                   num_filters,
                   use_pooling=True):
    
    shape = [filter_size, filter_size, num_input_channels, num_filters]    
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)
    
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1,1,1,1],
                         padding='SAME')
    
    layer = layer + biases
    print(layer)
    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1,2,2,1],
                               strides=[1,2,2,1],
                               padding='SAME')
    layer = tf.nn.relu(layer)
    print(layer)
    return layer, weights

###############################################
# flatten layer
###############################################
def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer,[-1,num_features])
    
    return layer_flat, num_features

###############################################
# FC layer
###############################################
def new_fc_layer(input, num_inputs, num_outputs, use_relu=True):
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    
    layer = tf.matmul(input, weights) + biases
    
    if use_relu:
        layer = tf.nn.relu(layer)
        
    return layer
###############################################




#total_iterations = 0

###############################################
# Print Progress
###############################################
def print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss, accuracy, session):
    # Calculate the accuracy on the training-set.
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)

    msg = "Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}, Validation Loss : {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))

###############################################
# Optimize
###############################################
def optimize (num_iterations, train_batch_size, data, x, y_true, img_size_flat, session, optimizer, accuracy, cost, dropProb, prob):
    
    #global total_iterations
    total_iterations = 0
    
    start_time = time.time()
    
    best_val_loss = float("inf")
    patience = 0
    
    for i in range(total_iterations,
                   total_iterations + num_iterations):

        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(train_batch_size)
        #x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(train_batch_size)
        
        x_batch = x_batch.reshape(train_batch_size, img_size_flat)
        #x_valid_batch = x_valid_batch.reshape(train_batch_size, img_size_flat)

        #print(x.shape)
        #feed_dict_train = {x: x_batch,
        #                   y_true: y_true_batch, dropProb:prob}
        #feed_dict_validate = {x: x_valid_batch, y_true:y_valid_batch}
        
        session.run(optimizer, feed_dict = {x: x_batch, y_true: y_true_batch, dropProb:prob})
        # Print status at end of each epoch (defined as full pass through)
        #if i % int(data.train.num_examples/train_batch_size) ==0:
        if i % 100 == 0:
            #val_loss = session.run(cost, feed_dict = feed_dict_validate)
            acc = session.run(accuracy, feed_dict={x: x_batch,
                                                   y_true: y_true_batch, dropProb:1.0})
            #epoch = int(i / int(data.train.num_examples/train_batch_size))
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
            #print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss, accuracy, session)
            print(msg.format(i + 1, acc))
            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience = 0
                else:
                    patience += 1

                if patience == early_stopping:
                    break

    
    total_iterations += num_iterations
    end_time = time.time()
    
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
    
###############################################
# Test and Prediction
###############################################
    
def print_validation_accuracy(data,x, y_true, session,y_pred_cls,validation_batch_size, img_size_flat, dropProb,
                        show_example_errors=False,
                        show_examples_correct=False, 
                        show_confusion_matrix=False):

    # Number of images in the valid-set.
    num_valid = len(data.valid.images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_valid, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_valid:
        # The ending index for the next batch is denoted j.
        j = min(i + validation_batch_size, num_valid)

        # Get the images from the test-set between index i and j.
        images = data.valid.images[i:j, :]
        #print(images, validation_batch_size)
        print(images.shape)
        images = images.reshape(validation_batch_size, img_size_flat)
        
        # Get the associated labels.
        labels = data.valid.labels[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images,
                     y_true: labels, dropProb:1.0}
        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = np.array(data.valid.cls)
    cls_pred = np.array([classes[x] for x in cls_pred])

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_valid

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_valid))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        wrong_image = plot_example_errors(data = data, cls_pred=cls_pred, correct=correct)
    if show_examples_correct:
        print("Correct examples")
        correct_image = plot_example_correct(data = data, cls_pred=cls_pred, correct=correct)
    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(data = data, cls_pred=cls_pred)
    return (wrong_image, correct_image)
    
    
def plot_example_errors(data, cls_pred, correct):
    # This function is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == False)
    
    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data.valid.images[incorrect]
    
    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]
    # Get the true classes for those images.
    cls_true = data.valid.cls[incorrect]
    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                img_size = 128,
                cls_pred=cls_pred[0:9])
    return images[0]

def plot_example_correct(data, cls_pred, correct):
    # This function is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == True)
    
    # Get the images from the test-set that have been
    # incorrectly classified.
    
    images = data.valid.images[incorrect]
    
    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = data.valid.cls[incorrect]
    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                img_size = 128,
                cls_pred=cls_pred[0:9])
    return images[0]
    
    
def plot_confusion_matrix(data, cls_pred):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the true classifications for the test-set.
    cls_true = data.valid.cls
    
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
    
        
        

def plot_conv_weights(session, weights, input_channel=0 ):
    # Assume weights are TensorFlow ops for 4-dim variables
    # e.g. weights_conv1 or weights_conv2.
    
    # Retrieve the values of the weight-variables from TensorFlow.
    # A feed-dict is not necessary because nothing is calculated.
    w = session.run(weights)

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Number of filters used in the conv. layer.
    num_filters = w.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids =  int(math.ceil(math.sqrt(num_filters)))
    
    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i<num_filters:
            # Get the weights for the i'th filter of the input channel.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = w[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

def plot_conv_layer(layer, image,x, session):
    # Assume layer is a TensorFlow op that outputs a 4-dim tensor
    # which is the output of a convolutional layer,
    # e.g. layer_conv1 or layer_conv2.

    # Create a feed-dict containing just one image.
    # Note that we don't need to feed y_true because it is
    # not used in this calculation.
    image = image.reshape(img_size_flat)
    feed_dict = {x: [image]}

    # Calculate and retrieve the output values of the layer
    # when inputting that image.
    values = session.run(layer, feed_dict=feed_dict)

    # Number of filters used in the conv. layer.
    num_filters = values.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    #num_grids = int(math.ceil(math.sqrt(num_filters)))
    
    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(3, 3)
    #fig, axes = plt.subplots(num_grids, num_grids)


    # Plot the output images of all the filters.
    for i, ax in enumerate(axes.flat):
        # Only plot the images for valid filters.
        if i<num_filters:
            # Get the output image of using the i'th filter.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = values[0, :, :, i]

            # Plot image.
            ax.imshow(img, interpolation='nearest', cmap='Greys_r')
            #ax.imshow(img, interpolation='nearest', cmap='binary')
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
    
    
def plot_image(image):
    plt.imshow(image.reshape(img_size,img_size,num_channels),
               interpolation='nearest',
               cmap='binary')

    plt.show()






    
