import sys
import time


import tensorflow as tf


# Declaring exception names:
class ConfigurationException(Exception): pass
class IncompleteFeedDict(Exception): pass



# Main class
class networkcore(object):
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
        super(networkcore, self).__init__()
        # This defines a core set of parameters needed to define the network model
        # It gets extended for each network that may inherit this class.
        self._core_network_params = [
            'BASE_LEARNING_RATE',
            'TRAINING',
        ]


    def set_params(self, params):
        ''' Check the parameters, if passes set the paramers'''
        if self._check_params(params):
            self._params = params

    def _check_params(self, params):
        for param in self._core_network_params:
            if param not in params:
                raise ConfigurationException("Missing paragmeter "+ str(param))
        return True
        # self._params = params

    def construct_network(self, dims):
        '''Build the network model

        Initializes the tensorflow model according to the parameters
        '''

        if not hasattr(self,'_params'):
            raise Exception('Missing _params member object, something has gone wrong.')


        tf.reset_default_graph()


        # Input placeholders:
        start = time.time()
        self._input = self._initialize_input(dims)
        sys.stdout.write(" - Finished input placeholders [{0:.2}s]\n".format(time.time() - start))

        # Actual Network implementation:
        start = time.time()
        self._logits = self._build_network(self._input)
        sys.stdout.write(" - Finished Network graph [{0:.2}s]\n".format(time.time() - start))

        # Create softmax and prediction objects:
        start = time.time()
        self._output = self._create_softmax(self._logits)
        sys.stdout.write(" - Finished Softmax creation [{0:.2}s]\n".format(time.time() - start))



        start = time.time()
        # Keep a list of trainable variables for minibatching:
        with tf.variable_scope('gradient_accumulation'):
            self._accum_vars = [tf.Variable(tv.initialized_value(),
                                trainable=False) for tv in tf.trainable_variables()]

        sys.stdout.write(" - Finished gradient accumulation [{0:.2}s]\n".format(time.time() - start))

        start = time.time()
        self._loss = self._calculate_loss(self._input, self._logits)
        sys.stdout.write(" - Finished loss calculation [{0:.2}s]\n".format(time.time() - start))


        start = time.time()
        self._accuracy = self._calculate_accuracy(self._input, self._output)
        sys.stdout.write(" - Finished accuracy calculation [{0:.2}s]\n".format(time.time() - start))


        # Optimizer:
        start = time.time()
        self._create_optimizer(self._loss)
        sys.stdout.write(" - Finished optimizer [{0:.2}s]\n".format(time.time() - start))


        # Merge the summaries:
        start = time.time()
        self._make_snapshots(self._input, self._output)
        self._merged_summary = tf.summary.merge_all()
        sys.stdout.write(" - Finished snapshotting [{0:.2}s]\n".format(time.time() - start))

        # All done building the network model, return
        return

    def _create_optimizer(self, trainable_loss):

        # Only make an optimizer if it is training:
        if self._params['TRAINING']:

            with tf.name_scope("training"):
                # Create a global step:
                self._global_step = tf.Variable(0, dtype=tf.int32,
                    trainable=False, name='global_step')

                if self._params['BASE_LEARNING_RATE'] <= 0:
                    opt = tf.train.AdamOptimizer()
                else:
                    opt = tf.train.AdamOptimizer(self._params['BASE_LEARNING_RATE'])

                # Variables for minibatching:
                self._zero_gradients =  [tv.assign(tf.zeros_like(tv)) for tv in self._accum_vars]
                self._accum_gradients = [self._accum_vars[i].assign_add(gv[0]) for
                                         i, gv in enumerate(opt.compute_gradients(trainable_loss))]
                self._apply_gradients = opt.apply_gradients(zip(self._accum_vars, tf.trainable_variables()),
                    global_step = self._global_step)

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
        raise NotImplementedError("Must implement _initialize_input")

    def _create_softmax(self, logits):
        '''Must return a dict type

        [description]

        Arguments:
            logits {[type]} -- [description]

        Raises:
            NotImplementedError -- [description]
        '''
        raise NotImplementedError("Must implement _create_softmax")


    def _make_snapshots(self, inputs, outputs):
        '''Create snapshots of inputs or outputs, as desired.
        This function doesn't raise a NotImplementedError, since it's not
        strictly necessary.
        '''
        pass

    def _calculate_loss(self, inputs, outputs):
        ''' Calculate the loss.

        returns a single scalar for the optimizer to use.
        '''

        raise NotImplementedError("Must implement _calculate_loss")



    def _calculate_accuracy(self, inputs, outputs):
        ''' Calculate the accuracy.

        '''

        raise NotImplementedError("Must implement _calculate_accuracy")


    def apply_gradients(self,sess):

        return sess.run( [self._apply_gradients], feed_dict = {})


    def feed_dict(self, inputs):
        '''Build the feed dict

        Take input images, labels and match
        to the correct feed dict tensorrs

        This is probably overridden in the subclass, but here you see the idea

        Arguments:
            images {dict} -- Dictionary containing the input tensors

        Returns:
            [dict] -- Feed dictionary for a tf session run call

        '''
        fd = dict()

        for key in inputs:
            if inputs[key] is not None:
                if isinstance(inputs[key], dict):
                    for secondard_key in inputs[key].keys():
                        fd.update({self._input[key][secondard_key] : inputs[key][secondard_key]})
                else:
                    fd.update({self._input[key] : inputs[key]})

        # for key in fd:
        #     print key
        #     print fd[key].shape
        #     print ''

        return fd

    def make_summary(self, sess, inputs):
        fd = self.feed_dict(inputs)
        return sess.run(self._merged_summary, feed_dict=fd)

    def zero_gradients(self, sess):
        sess.run(self._zero_gradients)

    def accum_gradients(self, sess, inputs):

        feed_dict = self.feed_dict(inputs)

        ops = [self._accum_gradients]
        doc = ['']


        # classification
        ops += [self._loss]
        doc += ['loss']

        if isinstance(self._accuracy, dict):
            for label_name in self._accuracy.keys():
                ops += [self._accuracy[label_name]]
                doc += ["acc. {0}".format(label_name)]
        else:
            ops += [self._accuracy]
            doc += ["acc. "]

        res = sess.run(ops, feed_dict = feed_dict )

        return res, doc


    def run_test(self,sess, inputs):
        feed_dict = self.feed_dict(inputs)

        ops = [self._loss]
        doc = ['loss']

        ops_metrics, doc_metrics = self.metrics(inputs)

        ops += ops_metrics
        doc += doc_metrics

        return sess.run(ops, feed_dict = feed_dict ), doc

    def inference(self,sess,inputs):

        feed_dict = self.feed_dict(inputs)

        ops = self._output['softmax']
        doc = ['softmax']

        ops_metrics, doc_metrics = self.metrics(inputs)

        ops += ops_metrics
        doc += doc_metrics

        return sess.run( ops, feed_dict = feed_dict ), doc

    def metrics(self, inputs):

        ops = []
        doc = []

        if 'label' in inputs.keys():
            if isinstance(self._accuracy, dict):
                for label_name in self._accuracy.keys():
                    ops += [self._accuracy[label_name]]
                    doc += ["acc. {0}".format(label_name)]
            else:
                ops += [self._accuracy]
                doc += ["acc. "]

        return ops, doc

    def global_step(self, sess):
        return sess.run(self._global_step)

    def _build_network(self, inputs):
        raise NotImplementedError("_build_network must be implemented by the child class")
