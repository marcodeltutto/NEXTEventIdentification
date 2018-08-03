import os
import sys
import time

import numpy

import tensorflow as tf

from ROOT import larcv

import uresnet, uresnet3d
# import uresnet, uresnet3d
import trainercore


class uresnet_trainer(trainercore.trainercore):

    def __init__(self, config):
        super(uresnet_trainer, self).__init__(config)

        if not self.check_params():
            raise Exception("Parameter check failed.")

        if '3d' in config['NAME']:
            net = uresnet3d.uresnet3d()
        else:
            net = uresnet.uresnet()

        net.set_params(config['NETWORK'])

        self.set_network_object(net)


    def fetch_minibatch_data(self, mode):
        # Return a dictionary object with keys 'image', 'label', and others as needed
        #            minibatch_data   = self._dataloaders['train'].fetch_data(
        #        self._config['TRAIN_CONFIG']['KEYWORD_DATA']).data()

        this_data = dict()
        this_data['image'] = self._dataloaders[mode].fetch_data(
            self._config['IO'][mode]['KEYWORD_DATA']).data()

        this_data['label'] = self._dataloaders[mode].fetch_data(
            self._config['IO'][mode]['KEYWORD_LABEL']).data()


        # If the weights for each pixel are to be normalized, compute the weights too:
        if self._config['NETWORK']['BALANCE_LOSS']:
            this_data['weight'] = self.compute_weights(this_data['label'])


        return this_data

    def fetch_minibatch_dims(self, mode):
        # Return a dictionary object with keys 'image', 'label', and others as needed
        # self._dataloaders['train'].fetch_data(keyword_label).dim() as an example
        this_dims = dict()
        this_dims['image'] = self._dataloaders[mode].fetch_data(
            self._config['IO'][mode]['KEYWORD_DATA']).dim()

        this_dims['label'] = self._dataloaders[mode].fetch_data(
            self._config['IO'][mode]['KEYWORD_LABEL']).dim()

        # If the weights for each pixel are to be normalized, compute the weights too:
        if self._config['NETWORK']['BALANCE_LOSS']:
            this_dims['weight'] = this_dims['image']

        return this_dims



    def compute_weights(self, labels):
        # Take the labels, and compute the per-label weight

        # Prepare output weights:

        weights = numpy.zeros(labels.shape)

        # print "entering compute weights, batch_size: " + str(len(labels))

        i = 0
        for batch in labels:
            # First, figure out what the labels are and how many of each:
            values, counts = numpy.unique(batch, return_counts=True)

            n_pixels = numpy.sum(counts)
            for value, count in zip(values, counts):
                weight = 1.0*(n_pixels - count) / n_pixels
                # print "  B{i}, L{l}, weight: ".format(i=i, l=value) + str(weight)
                mask = labels[i] == value
                weights[i, mask] += weight

            # Normalize the weights to sum to 1 for each event:
            s =  numpy.sum(weights[i])
            if s < 0.001:
                weights[i] *= 0.0
                weights[i] += 1.0
            else:
                weights[i] *= 1. / s
            i += 1


        return weights


    def ana_step(self):


        minibatch_data = self.fetch_minibatch_data('ANA')
        minibatch_dims = self.fetch_minibatch_dims('ANA')


        # Reshape any other needed objects:
        for key in minibatch_data.keys():
            minibatch_data[key] = numpy.reshape(minibatch_data[key], minibatch_dims[key])


        softmax = self.ana(minibatch_data)


        report_step  = self._iteration % self._config['REPORT_ITERATION'] == 0

        if self._output:

            # if report_step:
            #     print "Step {} - Acc all: {}, Acc non zero: {}".format(self._iteration,
            #         acc_all, acc_nonzero)

            # for entry in xrange(len(softmax)):
            #   self._output.read_entry(entry)
            #   data  = numpy.array(minibatch_data[entry]).reshape(softmax.shape[1:-1])
            entries   = self._dataloaders['ANA'].fetch_entries()
            event_ids = self._dataloaders['ANA'].fetch_event_ids()


            for entry in xrange(self._config['MINIBATCH_SIZE']):
                self._output.read_entry(entries[entry])

                larcv_data = self._output.get_data("image2d","sbndwire")
                larcv_lept = self._output.get_data("sparse2d","lepton")
                larcv_nlep = self._output.get_data("sparse2d","nonlepton")
                for projection_id in range(len(softmax)):

                    data = minibatch_data['image'][entry,:,:,projection_id]
                    nonzero_rows, nonzero_columns  = numpy.where(data > 0.1)
                    indexes = nonzero_columns * larcv_data.at(projection_id).meta().rows() + nonzero_rows
                    indexes = indexes.astype(dtype=numpy.uint64)

                    lepton_score = softmax[projection_id][entry,:,:,1]
                    nonlepton_score  = softmax[projection_id][entry,:,:,2]

                    mapped_lepton_score = lepton_score[nonzero_rows,nonzero_columns].astype(dtype=numpy.float32)
                    mapped_nonlepton_score = nonlepton_score[nonzero_rows, nonzero_columns].astype(dtype=numpy.float32)

                    # sum_score = lepton_score + nonlepton_score
                    # lepton_score = lepton_score / sum_score
                    # nonlepton_score  = nonlepton_score  / sum_score

                    nonlepton_vs = larcv.as_tensor2d(mapped_nonlepton_score, indexes)
                    nonlepton_vs.id(projection_id)
                    larcv_nlep.set(nonlepton_vs, larcv_data.at(projection_id).meta())
                    lepton_vs   = larcv.as_tensor2d(mapped_lepton_score, indexes)
                    lepton_vs.id(projection_id)
                    larcv_lept.set(lepton_vs, larcv_data.at(projection_id).meta())

                self._output.save_entry()
        else:
            print "Acc all: {}, Acc non zero: {}".format(acc_all, acc_nonzero)


        self._dataloaders['ANA'].next(store_entries   = (not self._config['TRAINING']),
                                      store_event_ids = (not self._config['TRAINING']))


