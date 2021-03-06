import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import ImageGrid


# Model construction utilities below adapted from
# https://www.tensorflow.org/versions/r0.8/tutorials/mnist/pros/index.html#deep-mnist-for-experts

def dense_variable(shape):
    initializer = tf.contrib.layers.xavier_initializer()
    initial = initializer(shape)
    return tf.Variable(initial)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    # initial = tf.constant(0.1, shape=shape)
    initializer = tf.constant_initializer(0.0)
    initial = initializer(shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def shuffle_aligned_list(data):
    """Shuffle arrays in a list by shuffling each array identically."""
    num = data[0].shape[0]
    p = np.random.permutation(num)
    return [d[p] for d in data]


def batch_generator(data, batch_size, shuffle=True):
    """Generate batches of data.
    
    Given a list of array-like objects, generate batches of a given
    size by yielding a list of array-like objects corresponding to the
    same slice of each input.
    """
    if shuffle:
        data = shuffle_aligned_list(data)

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= len(data[0]):
            batch_count = 0

            if shuffle:
                data = shuffle_aligned_list(data)

        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        start = (int)(start)
        end = (int)(end)
        yield [d[start:end] for d in data]


def imshow_grid(images, shape=[2, 8]):
    """Plot images in a grid of a given shape."""
    fig = plt.figure(1)
    grid = ImageGrid(fig, 111, nrows_ncols=shape, axes_pad=0.05)

    size = shape[0] * shape[1]
    for i in range(size):
        grid[i].axis('off')
        grid[i].imshow(images[i])  # The AxesGrid object work as a list of axes.

    plt.show()


def plot_embedding(X, y, d, title=None):
    """Plot an embedding X with the class label y colored by the domain d."""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.figure(figsize=(10,10))
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        # plot colored number
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.bwr(d[i] / 1.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 16:27:26 2016

@author: alvmu
"""

import numpy as np
import pdb


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    batchsize = int(batchsize)
    last_idx = batchsize * (len(inputs) / batchsize)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    # pdb.set_trace()
    for start_idx in range(0, int(last_idx + batchsize), int(batchsize)):
        if shuffle:
            if start_idx != last_idx:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = indices[start_idx:]
        else:
            if start_idx != last_idx:
                excerpt = slice(start_idx, start_idx + batchsize)
            else:
                excerpt = slice(start_idx, len(inputs))
        yield inputs[excerpt], targets[excerpt]


def iterate_minibatches_dann(inputs_phys, targets_phys, inputs_hosp, targets_hosp, batchsize, shuffle=False):
    assert len(inputs_phys) == len(targets_phys)
    assert len(inputs_hosp) == len(targets_hosp)
    last_idx_phys = (batchsize / 2) * (
        (len(inputs_phys) / (batchsize / 2)) - 1)  # Correction to avoid the last batch with different sizes
    last_idx_hosp = (batchsize / 2) * ((len(inputs_hosp) / (batchsize / 2)) - 1)
    last_idx = min(last_idx_phys, last_idx_hosp)
    if shuffle:
        indices_phys = np.arange(len(inputs_phys))
        indices_hosp = np.arange(len(inputs_hosp))
        np.random.shuffle(indices_phys)
        np.random.shuffle(indices_hosp)
    for start_idx in range(0, int(last_idx + batchsize / 2), int(batchsize / 2)):
        if shuffle:
            excerpt_phys = indices_phys[start_idx:start_idx + int(batchsize / 2)]
            excerpt_hosp = indices_hosp[start_idx:start_idx + int(batchsize / 2)]
        else:
            excerpt_phys = slice(start_idx, start_idx + int(batchsize / 2))
            excerpt_hosp = slice(start_idx, start_idx + int(batchsize / 2))
        inputs = np.concatenate((inputs_phys[excerpt_phys], inputs_hosp[excerpt_hosp]), axis=0)
        targets = np.concatenate((targets_phys[excerpt_phys], targets_hosp[excerpt_hosp]), axis=0)
        yield inputs, targets


def get_data_complete(idx_tmp, idx_test, data,
                      case):  # This function has to be called inside a for loop related to the loo scheme
    # Differentiale the number of subjects depending on the case considered
    subject_list = []
    if case == 'physionet':
        num_subjects = 20
        for i in range(1, num_subjects + 1):
            if (i == 1):
                current_inputs = data['Images'][0:data['FilesPerPatient'][i * 2 - 1]]
                current_targets = data['ClassLabels'][0:data['FilesPerPatient'][i * 2 - 1]]
            elif (i == num_subjects):
                current_inputs = data['Images'][data['FilesPerPatient'][i * 2 - 3]:data['FilesPerPatient'][i * 2 - 2]]
                current_targets = data['ClassLabels'][
                                  data['FilesPerPatient'][i * 2 - 3]:data['FilesPerPatient'][i * 2 - 2]]
            else:
                current_inputs = data['Images'][data['FilesPerPatient'][i * 2 - 3]:data['FilesPerPatient'][i * 2 - 1]]
                current_targets = data['ClassLabels'][
                                  data['FilesPerPatient'][i * 2 - 3]:data['FilesPerPatient'][i * 2 - 1]]
            subject_list.append([current_inputs, current_targets])

    elif case == 'hospital':
        num_subjects = 17
        for i in range(1, num_subjects + 1):
            if i == 1:
                current_inputs = data['Images'][0:data['FilesPerPatient'][i - 1]]
                current_targets = data['ClassLabels'][0:data['FilesPerPatient'][i - 1]]
            else:
                current_inputs = data['Images'][data['FilesPerPatient'][i - 2]:data['FilesPerPatient'][i - 1]]
                current_targets = data['ClassLabels'][data['FilesPerPatient'][i - 2]:data['FilesPerPatient'][i - 1]]
            subject_list.append([current_inputs, current_targets])

    # Randomly select 15 subjects for train and 4 subjects for validation
    # random.shuffle(idx_tmp)
    if case == 'physionet':
        idx_train = idx_tmp[0:14]
        idx_val = idx_tmp[14:17]
    elif case == 'hospital':
        idx_train = idx_tmp[0:11]
        idx_val = idx_tmp[11:14]

    num_subjects_train = np.size(idx_train)
    num_subjects_val = np.size(idx_val)
    num_subjects_test = np.size(idx_test)
    # Move training inputs and targets from list to numpy array
    # if case == 'hospital':
    #     pdb.set_trace()
    train_data = [subject_list[i] for i in idx_train]
    inputs_train = np.empty((0, 224, 224, 3), dtype='uint8')
    targets_train = np.empty((0, 5), dtype='uint8')
    # pdb.set_trace()
    for item in train_data:
        inputs_train = np.concatenate((inputs_train, item[0]), axis=0)
        targets_train = np.concatenate((targets_train, item[1]), axis=0)

    # Move validation inputs and targets from list to numpy array
    val_data = [subject_list[i] for i in idx_val]
    inputs_val = np.empty((0, 224, 224, 3), dtype='uint8')
    targets_val = np.empty((0, 5), dtype='uint8')
    for item in val_data:
        inputs_val = np.concatenate((inputs_val, item[0]), axis=0)
        targets_val = np.concatenate((targets_val, item[1]), axis=0)

    # Move test inputs and targets from list to numpy array
    test_data = [subject_list[i] for i in idx_test]
    inputs_test = np.empty((0, 224, 224, 3), dtype='uint8')
    targets_test = np.empty((0, 5), dtype='uint8')
    for item in test_data:
        inputs_test = np.concatenate((inputs_test, item[0]), axis=0)
        targets_test = np.concatenate((targets_test, item[1]), axis=0)

    return inputs_train, targets_train, inputs_val, targets_val, inputs_test, targets_test


def get_data_training_epoch(inputs_train,
                            targets_train):  # This function is used to organize the train data inside each epoch

    # Convert FROM onehot
    targets_train_int = [np.where(r == 1)[0][0] for r in targets_train]
    # Select only N3, N2, N1, REM, W
    idx0 = np.where(np.array(targets_train_int) == 0)
    idx1 = np.where(np.array(targets_train_int) == 1)
    idx2 = np.where(np.array(targets_train_int) == 2)
    idx3 = np.where(np.array(targets_train_int) == 3)
    idx4 = np.where(np.array(targets_train_int) == 4)
    # Split inputs according to class labels
    inputs_tr0 = inputs_train[idx0,]
    inputs_tr1 = inputs_train[idx1,]
    inputs_tr2 = inputs_train[idx2,]
    inputs_tr3 = inputs_train[idx3,]
    inputs_tr4 = inputs_train[idx4,]

    # Calculate class with fewest number of instances
    num_samples0 = len(idx0[0])
    num_samples1 = len(idx1[0])
    num_samples2 = len(idx2[0])
    num_samples3 = len(idx3[0])
    num_samples4 = len(idx4[0])
    min_samples = np.min((num_samples0, num_samples1, num_samples2, num_samples3, num_samples4))

    # Balance all classes to have the same number of inputs by downsampling
    idx0 = np.random.choice(range(num_samples0), min_samples, replace=False)
    idx1 = np.random.choice(range(num_samples1), min_samples, replace=False)
    idx2 = np.random.choice(range(num_samples2), min_samples, replace=False)
    idx3 = np.random.choice(range(num_samples3), min_samples, replace=False)
    idx4 = np.random.choice(range(num_samples4), min_samples, replace=False)

    del targets_train_int, targets_train, inputs_train

    # pdb.set_trace()
    # inputs_tr0 = inputs_tr0[idx0,]
    # inputs_tr1 = inputs_tr1[idx1,]
    # inputs_tr2 = inputs_tr2[idx2,]
    # inputs_tr3 = inputs_tr3[idx3,]
    # inputs_tr4 = inputs_tr4[idx4,]
    inputs_tr0 = inputs_tr0[0][idx0]
    inputs_tr1 = inputs_tr1[0][idx1]
    inputs_tr2 = inputs_tr2[0][idx2]
    inputs_tr3 = inputs_tr3[0][idx3]
    inputs_tr4 = inputs_tr4[0][idx4]

    inputs_tr = np.concatenate((inputs_tr0, inputs_tr1, inputs_tr2, inputs_tr3, inputs_tr4))
    targets_tr = np.uint8(np.vstack([np.tile([1., 0., 0., 0., 0.], [min_samples, 1]),
                                     np.tile([0., 1., 0., 0., 0.], [min_samples, 1]),
                                     np.tile([0., 0., 1., 0., 0.], [min_samples, 1]),
                                     np.tile([0., 0., 0., 1., 0.], [min_samples, 1]),
                                     np.tile([0., 0., 0., 0., 1.], [min_samples, 1])]))

    # targets_tr = np.uint8(np.concatenate((np.zeros((min_samples,),dtype='uint8'),np.ones((min_samples,),dtype='uint8'),
    #                                      np.repeat(2,(min_samples,)),np.repeat(3,(min_samples,)),np.repeat(4,(min_samples,)))))

    return inputs_tr, targets_tr


def get_data_validation_epoch(inputs_val,
                              targets_val):  # This function is used to organize the validation data inside each epoch
    # Select only N3, N2, N1, REM, W
    idx0 = (targets_val == 0)
    idx1 = (targets_val == 1)
    idx2 = (targets_val == 2)
    idx3 = (targets_val == 3)
    idx4 = (targets_val == 4)

    # Split inputs according to class labels
    inputs_v0 = inputs_val[idx0,]
    inputs_v1 = inputs_val[idx1,]
    inputs_v2 = inputs_val[idx2,]
    inputs_v3 = inputs_val[idx3,]
    inputs_v4 = inputs_val[idx4,]

    # Create appropriate inputs and targets
    num_samples0 = np.sum(idx0 == True)
    num_samples1 = np.sum(idx1 == True)
    num_samples2 = np.sum(idx2 == True)
    num_samples3 = np.sum(idx3 == True)
    num_samples4 = np.sum(idx4 == True)
    inputs_v = np.concatenate((inputs_v0, inputs_v1, inputs_v2, inputs_v3, inputs_v4))
    targets_v = np.uint8(np.vstack([np.tile([1., 0., 0., 0., 0.], [num_samples0, 1]),
                                    np.tile([0., 1., 0., 0., 0.], [num_samples1, 1]),
                                    np.tile([0., 0., 1., 0., 0.], [num_samples2, 1]),
                                    np.tile([0., 0., 0., 1., 0.], [num_samples3, 1]),
                                    np.tile([0., 0., 0., 0., 1.], [num_samples4, 1])]))
    # targets_v = np.uint8(np.concatenate((np.zeros((num_samples0,),dtype='uint8'),np.ones((num_samples1,),dtype='uint8'),
    #                                     np.repeat(2,(num_samples2)),np.repeat(3,(num_samples3)),np.repeat(4,(num_samples4)))))

    return inputs_v, targets_v


def get_data_testing_epoch(inputs_test,
                           targets_test):  # This function is used to organize the test data inside each epoch
    # Select only N3, N2, N1, REM, W balanced
    idx0 = (targets_test == 0)
    idx1 = (targets_test == 1)
    idx2 = (targets_test == 2)
    idx3 = (targets_test == 3)
    idx4 = (targets_test == 4)

    # Split inputs according to class labels
    inputs_te0 = inputs_test[idx0,]
    inputs_te1 = inputs_test[idx1,]
    inputs_te2 = inputs_test[idx2,]
    inputs_te3 = inputs_test[idx3,]
    inputs_te4 = inputs_test[idx4,]

    # Create appropriate inputs and targets
    num_samples0 = np.sum(idx0 == True)
    num_samples1 = np.sum(idx1 == True)
    num_samples2 = np.sum(idx2 == True)
    num_samples3 = np.sum(idx3 == True)
    num_samples4 = np.sum(idx4 == True)
    inputs_te = np.concatenate((inputs_te0, inputs_te1, inputs_te2, inputs_te3, inputs_te4))
    targets_te = np.uint8(np.vstack([np.tile([1., 0., 0., 0., 0.], [num_samples0, 1]),
                                     np.tile([0., 1., 0., 0., 0.], [num_samples1, 1]),
                                     np.tile([0., 0., 1., 0., 0.], [num_samples2, 1]),
                                     np.tile([0., 0., 0., 1., 0.], [num_samples3, 1]),
                                     np.tile([0., 0., 0., 0., 1.], [num_samples4, 1])]))
    # targets_te = np.uint8(np.concatenate((np.zeros((num_samples0,),dtype='uint8'),np.ones((num_samples1,),dtype='uint8'),
    #                                      np.repeat(2,(num_samples2)),np.repeat(3,(num_samples3)),np.repeat(4,(num_samples4)))))

    return inputs_te, targets_te