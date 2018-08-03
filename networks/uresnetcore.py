import sys
import time


import tensorflow as tf

from networkcore import networkcore



# Main class
class uresnetcore(networkcore):
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
        super(uresnetcore,self).__init__()

        # Extend the parameters to include the needed ones:

        self._core_network_params += [
            'BALANCE_LOSS',
            'NUM_LABELS',
        ]

        return


    def _initialize_input(self, dims):
        '''Initialize input parameters of the network.  Must return a dict type

        For example, paremeters of the dict can be 'image', 'label', 'weights', etc

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
            inputs['label'] =  tf.placeholder(tf.int64, dims['label'], name="label")

        if self._params['BALANCE_LOSS']:
            if 'weight' in dims:
                inputs['weight'] = tf.placeholder(tf.float32, dims['weight'], name='weight')
            else:
                raise Exception("Weights requested for loss balance, but dimension not provided.")

        return inputs
