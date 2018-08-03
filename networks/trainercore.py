import os
import sys
import time

import numpy

# Larcv imports:
from larcv import larcv
from larcv.dataloader2 import larcv_threadio

import tensorflow as tf

class trainercore(object):
    '''
    This class is the core interface for training.  Each function to
    be overridden for a particular interface is marked and raises
    a NotImplemented error.

    '''
    def __init__(self, config):
        self._config        = config
        self._dataloaders   = dict()
        self._iteration     = 0
        self._batch_metrics = None
        self._output        = None

        self._core_training_params = [
            'MINIBATCH_SIZE',
            'SAVE_ITERATION',
            'LOGDIR',
            'RESTORE',
            'ITERATIONS',
            'IO',
            'TRAINING',
            'NETWORK'
        ]

        # Make sure that 'BASE_LEARNING_RATE' and 'TRAINING'
        # are in net network parameters:


        config['NETWORK']['BASE_LEARNING_RATE'] = config['BASE_LEARNING_RATE']
        config['NETWORK']['TRAINING'] = config['TRAINING']



    def check_params(self):
        for param in self._core_training_params:
            if param not in self._config:
                raise Exception("Missing paragmeter "+ str(param))
        return True


    def _report(self,metrics,descr):
        msg = ''
        for i,desc in enumerate(descr):
          if not desc: continue
          msg += '%s=%6.6f   ' % (desc,metrics[i])
        msg += '\n'
        sys.stdout.write(msg)
        sys.stdout.flush()

    def delete(self):
        for key, manager in self._dataloaders.iteritems():
            manager.stop_manager()

    def prepare_manager(self, mode):

        if mode not in self._config['IO']:
            raise Exception("Missing IO config mode {} but trying to prepare manager.".format(mode))
        else:
            start = time.time()
            io = larcv_threadio()
            io_cfg = {'filler_name' : self._config['IO'][mode]['FILLER'],
                      'verbosity'   : self._config['IO'][mode]['VERBOSITY'],
                      'filler_cfg'  : self._config['IO'][mode]['FILE']}
            io.configure(io_cfg)
            io.start_manager(self._config['MINIBATCH_SIZE'])
            self._dataloaders.update({ mode : io})
            self._dataloaders[mode].next(store_entries   = (not self._config['TRAINING']),
                                         store_event_ids = (not self._config['TRAINING']))

            end = time.time()

            sys.stdout.write("Time to start {0} IO: {1:.2}s\n".format(mode, end - start))

        return

    def fetch_minibatch_data(self, mode):
        # Return a dictionary object with keys 'image', 'label', and others as needed
        #            minibatch_data   = self._dataloaders['train'].fetch_data(
        #        self._config['TRAIN_CONFIG']['KEYWORD_DATA']).data()

        raise NotImplementedError("Must implement fetch_minibatch_data in trainer.")

    def fetch_minibatch_dims(self, mode):
        # Return a dictionary object with keys 'image', 'label', and others as needed
        # self._dataloaders['train'].fetch_data(keyword_label).dim() as an example
        raise NotImplementedError("Must implement fetch_minibatch_dims in trainer.")

    def set_network_object(self, network):
        self._net = network

    def initialize(self):

        # Verify the network object is set:
        if not hasattr(self, '_net'):
            raise Exception("Must set network object by calling set_network_object() before initialize")



        # Prepare data managers:
        for mode in self._config['IO']:

            if mode not in ['TRAIN', 'TEST', 'ANA']:
                raise Exception("Unknown mode {} requested, must be in ['TRAIN', 'TEST', 'ANA']".format(mode))

            self.prepare_manager(mode)

            if mode == 'ANA' and 'OUTPUT' in self._config['IO'][mode]:
                print "Initializing output file"
                self._output = larcv.IOManager(self._config['IO'][mode]['OUTPUT'])
                self._output.initialize()

            if 'PROFILE_IO' in self._config['IO'][mode] and self._config['IO'][mode]['PROFILE_IO']:
                start = time.time()
                N = 5
                for i in range(5):
                    self._dataloaders[mode].next()
                    _ = self.fetch_minibatch_data(mode)
                end = time.time()
                sys.stdout.write("Time to read {N} batches of data: ".format(N=N) + str(end - start) + "\n")

        # Net construction:
        start = time.time()
        sys.stdout.write("Begin constructing network\n")

        # Make sure all required dimensions are present:

        # Either use TRAIN or ANA in the dim fetching
        if 'TRAIN' in self._config['IO']:
            dims = self.fetch_minibatch_dims(mode='TRAIN')
        elif 'ANA' in self._config['IO']:
            dims = self.fetch_minibatch_dims(mode='ANA')
        else:
            raise Exception("Tried to fetch dims from TRAIN or ANA but could not find configs.")


        self._net.construct_network(dims=dims)


        end = time.time()
        sys.stdout.write("Done constructing network. ({0:.2}s)\n".format(end-start))


        # Configure global process (session, summary, etc.)
        self._sess = tf.Session()
        self._writer = tf.summary.FileWriter(self._config['LOGDIR'] + '/train/')
        self._saver = tf.train.Saver()

        if 'TEST' in self._config['IO']:
            self._writer_test = tf.summary.FileWriter(self._config['LOGDIR'] + '/test/')

        #
        # Network variable initialization
        #
        if not self._config['RESTORE']:
                self._sess.run(tf.global_variables_initializer())
                self._writer.add_graph(self._sess.graph)
        else:
            latest_checkpoint = tf.train.latest_checkpoint(self._config['LOGDIR']+"/train/checkpoints/")
            print "Restoring model from {}".format(latest_checkpoint)
            self._saver.restore(self._sess, latest_checkpoint)
            print "Successfully restored."

    def train_step(self):

        self._iteration = self._net.global_step(self._sess)
        report_step  = self._iteration % self._config['REPORT_ITERATION'] == 0
        summary_step = 'SUMMARY_ITERATION' in self._config and (self._iteration % self._config['SUMMARY_ITERATION']) == 0
        checkpt_step = 'SAVE_ITERATION' in self._config and (self._iteration % self._config['SAVE_ITERATION']) == 0 and self._iteration != 0

        # We keep track of time spent on data IO and GPU calculations
        time_io   = 0.0
        time_gpu  = 0.0

        # Nullify the gradients
        self._net.zero_gradients(self._sess)

        # Loop over minibatches
        for j in xrange(self._config['N_MINIBATCH']):
            io_start = time.time()

            minibatch_data = self.fetch_minibatch_data('TRAIN')
            minibatch_dims = self.fetch_minibatch_dims('TRAIN')

            # Reshape labels by dict entry, if needed, or all at once:
            if isinstance(minibatch_data['label'], dict):
                for key in minibatch_data['label'].keys():
                    minibatch_data['label'][key] = numpy.reshape(
                        minibatch_data['label'][key], minibatch_dims['label'][key])
            else:
                minibatch_data['label'] = numpy.reshape(minibatch_data['label'], minibatch_dims['label'])

            # Reshape any other needed objects:
            for key in minibatch_data.keys():
                if 'label' not in key:
                    minibatch_data[key] = numpy.reshape(minibatch_data[key], minibatch_dims[key])

            io_end = time.time()
            time_io += io_end - io_start
            # compute gradients
            gpu_start = time.time()


            res,doc = self._net.accum_gradients(sess   = self._sess,
                                                inputs = minibatch_data)


            gpu_end  = time.time()
            time_gpu += gpu_end - gpu_start

            io_start = time.time()
            self._dataloaders['TRAIN'].next(store_entries   = (not self._config['TRAINING']),
                                            store_event_ids = (not self._config['TRAINING']))
            io_end   = time.time()

            time_io += io_end - io_start

            if self._batch_metrics is None:
                self._batch_metrics = numpy.zeros((self._config['N_MINIBATCH'],len(res)-1),dtype=numpy.float32)
                self._descr_metrics = doc[1:]

            self._batch_metrics[j,:] = res[1:]

        # update
        gpu_start = time.time()
        self._net.apply_gradients(self._sess)
        gpu_end   = time.time()

        time_gpu += gpu_end - gpu_start

        # read-in test data set if needed (TEST = true, AND it's a report/summary step)
        test_data = None
        if (report_step or summary_step) and 'TEST' in self._config['IO']:

            # Read the next batch:
            self._dataloaders['TEST'].next()


            test_data = self.fetch_minibatch_data('TEST')
            test_dims = self.fetch_minibatch_dims('TEST')

            # Reshape labels by dict entry, if needed, or all at once:
            if isinstance(test_data['label'], dict):
                for key in test_data['label'].keys():
                    test_data['label'][key] = numpy.reshape(
                        test_data['label'][key], test_dims['label'][key])
            else:
                test_data['label'] = numpy.reshape(test_data['label'], test_dims['label'])
            # Reshape any other needed objects:
            for key in test_data.keys():
                if 'label' not in key:
                    test_data[key] = numpy.reshape(test_data[key], test_dims[key])



        # Report
        if report_step:
            sys.stdout.write('@ iteration {}\n'.format(self._iteration))
            sys.stdout.write('Train set: ')
            self._report(numpy.mean(self._batch_metrics,axis=0),self._descr_metrics)
            if 'TEST' in self._dataloaders:
                res,doc = self._net.run_test(self._sess, test_data)
                sys.stdout.write('Test set: ')
                self._report(res,doc)
            sys.stdout.write(" -- IO Time: {0:.2}s\t GPU Time: {1:.2}s\n".format(time_io, time_gpu))

        # Save log
        if summary_step:
            # Run summary
            self._writer.add_summary(self._net.make_summary(self._sess, minibatch_data),
                                     self._iteration)
            if 'TEST' in self._config['IO']:
                self._writer_test.add_summary(self._net.make_summary(self._sess, test_data),
                                              self._iteration)

        # Save snapshot
        if checkpt_step:
            # Save snapshot
            ssf_path = self._saver.save(self._sess,
                self._config['LOGDIR']+"/train/checkpoints/save",
                global_step=self._iteration)
            sys.stdout.write('saved @ ' + str(ssf_path) + '\n')
            sys.stdout.flush()

    def ana(self, inputs):

        return  self._net.inference(sess   = self._sess,
                                    inputs = inputs)



    def ana_step(self):
        raise NotImplementedError("Must implement ana_step uniquely.")

    def batch_process(self):

        # Run iterations
        for i in xrange(self._config['ITERATIONS']):
            if self._config['TRAINING'] and self._iteration >= self._config['ITERATIONS']:
                print('Finished training (iteration %d)' % self._iteration)
                break

            # Start IO thread for the next batch while we train the network
            if self._config['TRAINING']:
                self.train_step()
            else:
                self.ana_step()

        if 'ANA' in self._config['IO'] and 'OUTPUT' in self._config['IO']['ANA']:
            self._output.finalize()
