import os
import tarfile
import sys
import glob
import random
import math
import datetime
import time
import itertools
import json
import re
import logging
from collections import OrderedDict
import numpy as np
import scipy.misc
import tensorflow as tf

from PIL import Image

def print_num_of_total_parameters(output_detail=False, output_to_logging=False):
    total_parameters = 0
    parameters_string = ""

    for variable in tf.global_variables():

        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
        if len(shape) == 1:
            parameters_string += ("%s %d, \n" % (variable.name, variable_parameters))
        else:
            parameters_string += ("%s %s=%d, \n" % (variable.name, str(shape), variable_parameters))

    if output_to_logging:
        if output_detail:
            logging.info(parameters_string)
        logging.info("Total %d variables, %s params" % (len(tf.trainable_variables()), "{:,}".format(total_parameters)))
    else:
        if output_detail:
            print(parameters_string)
        print("Total %d variables, %s params" % (len(tf.trainable_variables()), "{:,}".format(total_parameters)))



class DeeplabLIPModel():
    TENSOR_NAME_LAST_CONV = "logits/semantic/weights:0"
    TENSOR_NAME_LAST_B = "logits/semantic/biases:0"

    TENSOR_NAME_DECODER_0_OUTPUT = \
            "decoder/decoder_conv0_pointwise/Relu:0" 

    TENSOR_NAME_DECODER_1_OUTPUT = \
            "decoder/decoder_conv1_pointwise/Relu:0" 

    TENSOR_NAME_CHECK = \
            "aspp0/weights:0" 

    TENSOR_NAME_DECODER_CONCAT = \
            "decoder/concat:0" 

    TENSOR_NAME_RESIZE = \
            "ResizeBilinear_3:0" 

    TENSOR_NAME_PREDICT = 'SemanticPredictions:0'
            

    #INPUT_TENSOR_NAME = "original_images" 
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self):
        pass

    def load_from_fpb_to_detect(self, 
            pb_path,
            visible_gpu_devices="0"):
        """
            pb_path: string path of *.pb to restore the model parameters.

            visible_gpu_devices: string index list like "0,1" to choose 
            the gpu to deploy. Attention, "0,1" is the physical index of
            os.environ['CUDA_VISIBLE_DEVICES'], for example, if 
            os.environ['CUDA_VISIVLE_DEVICES'] = "0,3", then 
            visible_gpu_devices="1", is the physical GPU:3.
        """

        graph_def = tf.GraphDef.FromString(open(pb_path,"rb").read())

        self.graph = tf.Graph()
        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')
        with self.graph.as_default():
            _ = tf.import_graph_def(graph_def, name='')

            self.input_images = \
            self.graph.get_tensor_by_name(self.INPUT_TENSOR_NAME)
            self.predicts = \
            self.graph.get_tensor_by_name(self.TENSOR_NAME_PREDICT)

            print("\n\n LOAD Model.PB OVER \n")
        self.gpucfg = tf.GPUOptions(allow_growth=True,\
                visible_device_list=visible_gpu_devices)
        self.config = tf.ConfigProto(gpu_options=self.gpucfg)
        self.sess = tf.Session(graph=self.graph, config=self.config) 

    def detect_img(self, dataset):
        sess = self.sess
        res = {
            'imgs': [],
            'segs': []
                }
        total_imgs = dataset.datasize
        for i in range(total_imgs):
            t1 = datetime.datetime.now()

            img = dataset.get_next_raw_image()
            seg_map = self.detect_sigle_img(img)
            t2 = datetime.datetime.now()

            used = float((t2-t1).microseconds) / 1000.0
            res['imgs'].append(img[0])
            res['segs'].append(seg_map)
            if (i+1) % 10 == 0:
                msg = "<<=====Finish Detection: {}, {}/{} time:{} (ms/pic)\
                =======>>".format(float(i+1)/float(total_imgs),
                        i+1, total_imgs, used)
                print(msg)
        return res


    def detect_sigle_img(self, img):
        """
            img: np.array [H,W,C], C must=3
        """
        sess = self.sess
        batch_seg_map = sess.run(
            self.predicts,
            feed_dict={ self.input_images: np.array([img]) }
            )
        seg_map = batch_seg_map[0]
        if len(seg_map.shape) >= 3:
            seg_map = np.array(seg_map).argmax(axis=2)

        return seg_map


