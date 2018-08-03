import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy


def convolutional_block(input_tensor,
                        is_training,
                        name="",
                        batch_norm=True,
                        dropout=True,
                        kernel_size=[3,3,3],
                        strides=[1,1,1],
                        n_filters=None,
                        reuse=False):

    with tf.variable_scope(name):

        # Apply convolution, relu, batchnorm, and dropout to a tensor and return.
        # This does not change the dimensions of the tensor.

        # This block applies a regularizer to the weights in the convolution,
        # which ought to help with overfitting.  To add it to the loss,
        # Use the following code to collect the regularization loss at the
        # stage where the optimizer is done:
        #
        #   reg_loss = tf.losses.get_regularization_loss()
        #

        x = input_tensor
        if n_filters is None:
            n_filters = x.shape[-1]

        # Convolutional layer:
        x = tf.layers.conv3d(x, n_filters,
                             kernel_size=[3, 3, 3],
                             strides=[1, 1, 1],
                             padding='same',
                             activation=None,
                             use_bias=False,
                             trainable=is_training,
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                             name="Conv3D",
                             reuse=reuse)


        # Activation:
        x = tf.nn.relu(x)

        # Batch normalization, if needed:
        if batch_norm:
            with tf.variable_scope("batch_norm") as scope:
                # updates_collections=None is important here
                # it forces batchnorm parameters to be saved immediately,
                # rather than have to be saved into snapshots manually.
                x = tf.contrib.layers.batch_norm(x,
                                                 updates_collections=None,
                                                 decay=0.9,
                                                 is_training=is_training,
                                                 trainable=is_training,
                                                 scope=scope,
                                                 # name="BatchNorm",
                                                 reuse=reuse)
        # Take the

        # Apply dropout, as needed:
        if dropout:
            x = tf.layers.dropout(x, rate = 0.2)

    return x






def downsample_block(input_tensor,
                     is_training,
                     batch_norm=False,
                     dropout=False,
                     name="",
                     reuse = False):
    """
    @brief      Downsample the input tensor with strided convolutions, increase n_filters by x2

    @param      input_tensor  The input tensor
    @param      kernel        Size of convolutional kernel to apply
    @param      n_filters     Number of output filters

    @return     { Tensor with the residual network applied }
    """

    # Residual block has the identity path summed with the output of
    # BN/Relu/Conv2d applied twice
    with tf.variable_scope(name):

        x = input_tensor

        # Assuming channels last here:
        n_filters = 2*x.get_shape().as_list()[-1]


        # Convolutional layer:
        x = tf.layers.conv3d(x, n_filters,
                             kernel_size=[3, 3, 3],
                             strides=[2, 2, 2],
                             padding='same',
                             activation=None,
                             use_bias=False,
                             trainable=is_training,
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                             name="Conv3D_downsample",
                             reuse=reuse)

        # Activation:
        x = tf.nn.relu(x)

        # Batch normalization, if needed:
        if batch_norm:
            with tf.variable_scope("batch_norm") as scope:
                # updates_collections=None is important here
                # it forces batchnorm parameters to be saved immediately,
                # rather than have to be saved into snapshots manually.
                x = tf.contrib.layers.batch_norm(x,
                                                 updates_collections=None,
                                                 decay=0.9,
                                                 is_training=is_training,
                                                 trainable=is_training,
                                                 scope=scope,
                                                 # name="BatchNorm",
                                                 reuse=reuse)
        # Take the

        # Apply dropout, as needed:
        if dropout:
            x = tf.layers.dropout(x, rate = 0.2)

    return x


def upsample_block(input_tensor,
                   is_training,
                   batch_norm=False,
                   dropout=False,
                   n_output_filters=0,
                   name="",
                   reuse=False):

    """Upsample a tensor using transposed convolutions
    If you want to just upsample by splitting pixels, do that by hand

    By default, this halves (rounding up if needed) the number of filters.
    You can override this with n_output_filters
    """

    with tf.variable_scope(name):

        x = input_tensor

        # Assuming channels last here:
        n_filters = int(0.5*input_tensor.get_shape().as_list()[-1])
        if n_filters == 0:
            n_filters = 1

        if n_output_filters == 0:
            n_output_filters = n_filters

        # Convolutional layer:
        x = tf.layers.conv3d_transpose(x, n_filters,
                             kernel_size=[3, 3, 3],
                             strides=[2, 2, 2],
                             padding='same',
                             activation=None,
                             use_bias=False,
                             trainable=is_training,
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                             name="Conv3D_upsample",
                             reuse=reuse)

        # Activation:
        x = tf.nn.relu(x)

        # Batch normalization, if needed:
        if batch_norm:
            with tf.variable_scope("batch_norm") as scope:
                # updates_collections=None is important here
                # it forces batchnorm parameters to be saved immediately,
                # rather than have to be saved into snapshots manually.
                x = tf.contrib.layers.batch_norm(x,
                                                 updates_collections=None,
                                                 decay=0.9,
                                                 is_training=is_training,
                                                 trainable=is_training,
                                                 scope=scope,
                                                 # name="BatchNorm",
                                                 reuse=reuse)
        # Take the

        # Apply dropout, as needed:
        if dropout:
            x = tf.layers.dropout(x, rate = 0.2)

    return x


def residual_block(input_tensor,
                   is_training,
                   batch_norm=False,
                   name="",
                   reuse=False):
    """
    @brief      Create a residual block and apply it to the input tensor

    @param      input_tensor  The input tensor
    @param      kernel        Size of convolutional kernel to apply
    @param      n_filters     Number of output filters

    @return     { Tensor with the residual network applied }
    """

    # Residual block has the identity path summed with the output of
    # BN/Relu/Conv2d applied twice
    with tf.variable_scope(name):
        x = input_tensor

        # Assuming channels last here:
        n_filters = x.shape[-1]

        with tf.variable_scope(name + "_0"):
            x = convolutional_block(x,
                                    is_training,
                                    name='conv_block',
                                    batch_norm=batch_norm,
                                    dropout=False,
                                    reuse=reuse)

        # Apply everything a second time:
        with tf.variable_scope(name + "_1"):
            x = convolutional_block(x,
                                    is_training,
                                    name='conv_block',
                                    batch_norm=batch_norm,
                                    dropout=False,
                                    reuse=reuse)

        # Sum the input and the output:
        with tf.variable_scope(name+"_add"):
            x = tf.add(x, input_tensor, name="Add")

    return x
