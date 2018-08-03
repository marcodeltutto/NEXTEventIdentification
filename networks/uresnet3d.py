import sys
import time


import tensorflow as tf

from utils3d import residual_block, downsample_block, upsample_block, convolutional_block


from uresnetcore import uresnetcore

class uresnet3d(uresnetcore):

    '''
    U resnet implementation
    '''
    def __init__(self):
        '''initialization

        Requires a list of parameters as python dictionary

        Arguments:
            params {dict} -- Network parameters

        Raises:
            ConfigurationException -- Missing a required parameter
        '''

        # Call the base class to initialize _core_network_params:
        super(uresnet3d, self).__init__()

        # Extend the parameters to include the needed ones:

        self._core_network_params += [
            'N_INITIAL_FILTERS',
            'RESIDUAL_BLOCKS_PER_LAYER',
            'RESIDUAL_BLOCKS_DEEPEST_LAYER',
            'NETWORK_DEPTH',
        ]


        return








    def _create_softmax(self, logits):
        '''Must return a dict type

        [description]

        Arguments:
            logits {[type]} -- [description]

        Raises:
            NotImplementedError -- [description]
        '''

        # For the logits, we compute the softmax and the predicted label


        output = dict()

        output['softmax'] = tf.nn.softmax(logits)
        output['prediction'] = tf.argmax(logits, axis=-1)

        return output





    def _calculate_loss(self, inputs, outputs):
        ''' Calculate the loss.

        returns a single scalar for the optimizer to use.
        '''


        with tf.name_scope('cross_entropy'):
            label = tf.squeeze(inputs['label'], axis=-1)


            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label,
                                                                  logits=outputs)


            if self._params['BALANCE_LOSS']:
                weight = tf.squeeze(inputs['weight'], axis=-1)
                loss = tf.multiply(loss, weight)

            loss = tf.reduce_sum(loss)


            # If desired, add weight regularization loss:
            if 'REGULARIZE_WEIGHTS' in self._params:
                reg_loss = tf.losses.get_regularization_loss()
                loss += reg_loss


            # Total summary:
            tf.summary.scalar("Total Loss",loss)

            return loss

    def _calculate_accuracy(self, inputs, outputs):
        ''' Calculate the accuracy.

        '''

        # Compare how often the input label and the output prediction agree:

        with tf.name_scope('accuracy'):

            if isinstance(outputs['prediction'], dict):
                accuracy = dict()

                for key in outputs['prediction'].keys():
                    correct_prediction = tf.equal(tf.argmax(inputs['label'][key], -1),
                                                  outputs['prediction'][key])
                    accuracy.update({
                        key : tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    })


                    # Add the accuracies to the summary:
                    tf.summary.scalar("{0}_Accuracy".format(key), accuracy[key])

            else:
                correct_prediction = tf.equal(tf.argmax(inputs['label'], -1),
                                              outputs['prediction'])
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                tf.summary.scalar("Accuracy", accuracy)

        return accuracy

    def _build_network(self, inputs, verbosity = 0):

        if verbosity > 1:
            print inputs

        x = inputs['image']

        # This is a straightforward UResNet architecture.
        # The downsampling and upsampling steps are not complicated
        # residual steps, just normal convolutional layers


        # Initial convolution to get to the correct number of filters:

        x = convolutional_block(x,
                        is_training=self._params['TRAINING'],
                        name="Conv2DInitial",
                        batch_norm=True,
                        dropout=False,
                        kernel_size=[5,5,5],
                        n_filters=self._params['N_INITIAL_FILTERS'])

        if verbosity > 1:
            print x.get_shape()



        # Need to keep track of the outputs of the residual blocks before downsampling, to feed
        # On the upsampling side

        network_filters = []

        # Begin the process of residual blocks and downsampling:
        for i in xrange(self._params['NETWORK_DEPTH']):

            for j in xrange(self._params['RESIDUAL_BLOCKS_PER_LAYER']):
                name = "resblock_down"
                name += "_{0}_{1}".format(i, j)


                x = residual_block(x, self._params['TRAINING'],
                                   batch_norm=self._params['BATCH_NORM'],
                                   name=name)

            name = "downsample"
            name += "_{0}".format(i)

            network_filters.append(x)
            x = downsample_block(x, self._params['TRAINING'],
                                    batch_norm=self._params['BATCH_NORM'],
                                    name=name)

            if verbosity > 1:
                print "Layer {i}: x.get_shape(): {s}".format(
                    i=i, s=x.get_shape())

        if verbosity > 0:
            print "Reached the deepest layer."

        # At the bottom, do another residual block:
        for j in xrange(self._params['RESIDUAL_BLOCKS_DEEPEST_LAYER']):
            x = residual_block(x, self._params['TRAINING'],
                batch_norm=self._params['BATCH_NORM'], name="deepest_block_{0}".format(j))


        # Come back up the network:
        for i in xrange(self._params['NETWORK_DEPTH']-1, -1, -1):

            # How many filters to return from upsampling?
            n_filters = network_filters[-1].get_shape().as_list()[-1]

            if verbosity > 1:
                print "Layer {i}: x.get_shape(): {s}".format(
                    i=i, s=x.get_shape())

            name = "upsample"
            name += "_{0}".format(i)

            # Upsample:
            x = upsample_block(x,
                               self._params['TRAINING'],
                               batch_norm=self._params['BATCH_NORM'],
                               n_output_filters=n_filters,
                               name=name)


            x = tf.concat([x, network_filters[-1]],
                              axis=-1, name='up_concat_{0}'.format(i))

            # Remove the recently concated filters:
            network_filters.pop()
            # with tf.variable_scope("bottleneck_plane{0}_{1}".format(p,i)):

            name = "BottleneckUpsample"
            name += "_{0}".format(i)


            # Include a bottleneck to reduce the number of filters after upsampling:
            x = convolutional_block(x,
                        is_training=self._params['TRAINING'],
                        name=name,
                        batch_norm=True,
                        dropout=False,
                        kernel_size=[1,1,1],
                        n_filters=n_filters)


            # Residual
            for j in xrange(self._params['RESIDUAL_BLOCKS_PER_LAYER']):
                name = "resblock_up"
                name += "_{0}_{1}".format(i, j)

                x = residual_block(x, self._params['TRAINING'],
                                   batch_norm=self._params['BATCH_NORM'],
                                   name=name)



        name = "FinalResidualBlock"

        x = residual_block(x,
                self._params['TRAINING'],
                batch_norm=self._params['BATCH_NORM'],
                name=name)

        name = "BottleneckConv2D"

        # At this point, we ought to have a network that has the same shape as the initial input
        #, but with more filters.

        x = tf.layers.conv3d(x,
                             self._params['NUM_LABELS'],
                             kernel_size=[1,1,1],
                             strides=[1,1,1],
                             padding='same',
                             activation=None,
                             use_bias=False,
                             trainable=self._params['TRAINING'],
                             name=name)

        if verbosity > 0:
            print "Final output shape: " + str(x.get_shape())

        return x

