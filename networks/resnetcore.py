import sys
import time

from networkcore import networkcore

import tensorflow as tf

# Main class
class resnetcore(networkcore):
    '''Define a network model and run training

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
        super(resnetcore,self).__init__()

        # Extend the parameters to include the needed ones:

        # As of this moment, resnetcore doesn't actually add any params
        # self._core_network_params.append([
        #     # 'MINIBATCH_SIZE',
        #     # 'SAVE_ITERATION',
        #     # 'NUM_LABELS',
        #     # 'LOGDIR',
        #     # 'RESTORE',
        #     # 'ITERATIONS',
        # ])

        return


    def _initialize_input(self, dims):
        '''Initialize input parameters of the network.  Must return a dict type

        For exampe, paremeters of the dict can be 'image', 'label', 'weights', etc

        Arguments:
            dims {[type]} -- [description]

        Keyword Arguments:
            label_dims {[type]} -- [description] (default: {None})

        Raises:
            NotImplementedError -- [description]
        '''

        inputs = dict()

        inputs.update({
            'image' :  tf.placeholder(tf.float32, dims['image'], name="input_image")
        })


        if 'label' in dims:
            label_dims = dims['label']
            if isinstance(label_dims, dict):
                inputs['label'] = dict()
                for key in label_dims:
                    inputs['label'].update({
                        key : tf.placeholder(tf.int64, label_dims[key], name="label_{}".format(key))
                    })
            else:
                inputs['label'] =  tf.placeholder(tf.int64, label_dims, name="label")#.format(key))

        return inputs

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

        if isinstance(logits, dict):
            output['softmax']    = dict()
            output['prediction'] = dict()
            for key in logits.keys():
                output['softmax'][key]    = tf.nn.softmax(logits[key])
                output['prediction'][key] = tf.argmax(logits[key], axis=-1)


        else:
            output['softmax'] = tf.nn.softmax(logits)
            output['prediction'] = tf.argmax(logits, axis=-1)

        return output





    def _calculate_loss(self, inputs, outputs):
        ''' Calculate the loss.

        returns a single scalar for the optimizer to use.
        '''


        with tf.name_scope('cross_entropy'):


            # If logits is a dictionary form, create a loss against each label name:
            if isinstance(outputs, dict):
                losses = []

                for key in outputs:
                    losses.append(
                        tf.reduce_mean(
                            tf.nn.softmax_cross_entropy_with_logits(labels=inputs['label'][key],
                                                                    logits=outputs[key])
                        )
                    )
                    # Individual summaries:
                    tf.summary.scalar("{}_loss".format(key), losses[-1])

                # Compute the total loss:
                loss = sum(losses)


            else:
                #otherwise, just one set of logits, against one label:
                loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=inputs['label'],
                                                            logits=outputs))



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
