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

    def __init__(self, 
            mode="training", 
            config="", 
            model_dir="", 
            model_label=0):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.model_path = model_dir + "/model.ckpt-" + \
                            str(model_label) + ".data-00000-of-00001"
        self.model_meta_path = model_dir + "/model.ckpt-" + \
                        str(model_label) + ".meta"

    def get_image(self, img_path=""):
        """Inferences DeepLab model and visualizes result."""
        img_name = "5672157fa4062ff06c6019e4058d4667.jpg"
        img_path = os.path.join("../Mask_RCNN_BBox/Images/blouse/",img_name)
        image = Image.open(img_path)

        print('running deeplab on image %s...' % img_path)

        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)

        return resized_image

    def weight_variable(self, shape, name="weights"):
        initial = tf.truncated_normal(shape, \
                dtype=tf.float32,stddev=0.01) 
        return tf.Variable(initial, name=name)

    def bias_variable(self, shape, name="biases"):
        initial = tf.constant(0.01, \
                dtype=tf.float32, shape=shape)
        return tf.Variable(initial, name=name)


    def load_from_meta(self):
        tf.reset_default_graph()
        saver = tf.train.import_meta_graph(self.model_meta_path)



    def load_from_pb(self, tarball_path, 
            mode="training", 
            init_last=True,
            num_seg_class=20):
        """Creates and loads pretrained deeplab model."""

        graph_def = None
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            print(tar_info.name)
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.GraphDef.FromString(file_handle.read())
                break
        tar_file.close()

        self.graph = tf.Graph()
        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            _ = tf.import_graph_def(graph_def, name='')
            #tf.train.import_meta_graph(self.model_meta_path)
            
            #print(_)

            
            print("variables")
            for var in tf.trainable_variables():
                #print(var.name)
                print(var)
            #print("operation")
            #for op in self.graph.get_operations():
            #    print(op.name)
            #print_num_of_total_parameters(True)
            print("\n\n Model Param \n\n")

            self.input_images = \
            self.graph.get_tensor_by_name(self.INPUT_TENSOR_NAME)
            img_shape = tf.shape(self.input_images)

            image_batch = img_shape[0]
            image_height = img_shape[1]
            image_width = img_shape[2]

            self.gt_segments = tf.placeholder(
                    dtype=tf.int32,
                    shape=(None, None, None)
                    )

            self.last_kernel = \
            self.graph.get_tensor_by_name(self.TENSOR_NAME_LAST_CONV)

            self.last_bias = \
            self.graph.get_tensor_by_name(self.TENSOR_NAME_LAST_B)

            self.last_feature_tensor = \
            self.graph.get_tensor_by_name(self.TENSOR_NAME_DECODER_1_OUTPUT)


            self.decoder_concat = \
            self.graph.get_tensor_by_name(self.TENSOR_NAME_DECODER_CONCAT)

            self.resize_tensor = \
            self.graph.get_tensor_by_name(self.TENSOR_NAME_RESIZE)

            self.predicts = \
            self.graph.get_tensor_by_name(self.TENSOR_NAME_PREDICT)
            

            print(self.input_images)
            #print(self.decoder_concat)
            print(self.last_feature_tensor)

            #print(self.feature)
            print(self.last_kernel)
            print(self.last_bias)

            print(self.resize_tensor)

            print(self.predicts)

                
            
            if mode == "training" and init_last == True:
                with tf.name_scope("logits_finetune"):
                    with tf.name_scope("semantic_finetune"):
                        self.kernel = self.weight_variable([1,1,256,num_seg_class])
                        self.bias = self.bias_variable([num_seg_class])

                        self.predicts = tf.nn.conv2d(self.last_feature_tensor, \
                                self.kernel, [1,1,1,1], padding='SAME')
                        self.predicts = self.predicts + self.bias
                        print(self.predicts)
                        
                        #self.predicts = tf.nn.relu(self.predicts+bias)
                        self.predicts = tf.image.resize_bilinear(
                                images=self.predicts,
                                size=[129, 129])
                        
                        self.predicts = tf.image.resize_bilinear(
                                images=self.predicts,
                                size=[image_height, image_width],
                                align_corners=True)
           
            #self.gt_segments = tf.squeeze(self.gt_segments)
            mask = tf.less_equal(self.gt_segments, 19)
            print("mask:")
            print(mask)
            print(self.gt_segments)

            #self.logits_segments = tf.boolean_mask(self.predicts, mask)
            self.logits_segments = self.predicts
            #print()
            #self.gt_segments = tf.boolean_mask(self.gt_segments, mask)
            

            self.logits_segments = tf.check_numerics(
                    self.logits_segments, " predicts nan") 

            #self.gt_segments = tf.check_numerics(
            #        tf.cast(self.gt_segments,tf.float32), "\
            #        gt_segments nan") 

            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.logits_segments,
                    labels= tf.cast(self.gt_segments,tf.int32),
                    name="fine-tune-loss"
                    )
            self.loss = tf.check_numerics(self.loss, "loss nan ")
            self.loss = tf.reduce_mean(self.loss)
            




            print("\n predicts \n")
            #print(tf.trainable_variables())
            print(self.predicts)
            print(self.loss)
            
            self.check = \
            self.graph.get_tensor_by_name(self.TENSOR_NAME_CHECK)


            #self.optimize = \
            #tf.train.AdamOptimizer(
            #        learning_rate=0.01,
            #        name="fine-tune-op").minimize(self.loss,
            #                var_list=tf.trainable_variables())
            self.optimize = \
                tf.train.GradientDescentOptimizer(
                        learning_rate=0.01,
                        name="fine-tune-op-SGD"
                        ).minimize(self.loss)


            print("\n over \n")


    def load_from_fpb_to_detect(self, 
            pb_path, 
            mode="inference", 
            init_last=False,
            num_seg_class=20):
        """Creates and loads pretrained deeplab model."""

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
    
    def train_model(self, 
            dataset, 
            start_epoch = 0,
            nepochs=50000, 
            steps_per_epoch=30000,
            batch_size=1,
            save_model_dir='./output_model/',
            step_to_log=100):
        epoch = start_epoch
        total_loss = 0.0
        total_step_loss = 0.0
        with tf.Session(graph=self.graph) as sess:
            # init summary
            tf.summary.scalar("loss", self.loss)
            merged_summary  = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter('/tmp/lip_',
                    sess.graph)

            tf.global_variables_initializer().run()
            saver = tf.train.Saver()


            ckpt = tf.train.get_checkpoint_state(save_model_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("\nRestore Model From\
                        {}\n".format(ckpt.model_checkpoint_path))
            else:
                print("\nNot Model To Restore, Init From Pre-train\n")
                #saver.restore(sess,'./deeplabv3_pascal_train_aug/model.ckpt.data-00000-of-00001')

            for var in tf.global_variables():
                print(var)
            print(tf.trainable_variables())

            for epoch in range(start_epoch + nepochs):
                for i in range(steps_per_epoch):
                    batch_xs, batch_ys = \
                    dataset.get_next_batch_datas(batch_size)
                    feed_dict = {
                        self.input_images: batch_xs,
                        self.gt_segments: batch_ys
                        }
                    
                    _,loss,check,summary = sess.run(
                            [self.optimize, self.loss,
                                self.check,merged_summary], 
                            feed_dict=feed_dict
                            )
                    total_loss += loss
                    total_step_loss += loss
                    if (i+1) % step_to_log == 0:
                        msg = "Epoch:{} Step:{} mean_step_loss:{} mean_loss:{}".format(\
                                epoch, i+1, total_step_loss / float(step_to_log), \
                                total_loss / float(i+1))
                        total_step_loss = 0.0
                        print(msg)
                        summary_writer.add_summary(
                                summary, 
                                epoch*steps_per_epoch + i
                                )
                        summary_writer.flush()
                msg = "\n Epoch:{} epoch_mean_loss:{} \n".\
                        format(epoch, total_loss / float(steps_per_epoch))
                print(msg)
                total_loss = 0.0
                total_step_loss = 0.0
                saver.save(sess, save_model_dir + 'model.ckpt', global_step=epoch)
   

    def detect(self, 
            image,
            save_model_dir='./output_model_2/'):
        resized_image = image
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(self.model_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("\nRestore Model From\
                        {}\n".format(ckpt.model_checkpoint_path))
            else:
                print("\nNot Model To Restore\n")

            batch_seg_map = sess.run(
                self.predicts,
                feed_dict={ self.input_images: resized_image }
                )
            seg_map = batch_seg_map[0]
            print(seg_map.shape)
            if len(seg_map.shape) >= 3:
                seg_map = np.array(seg_map).argmax(axis=2)
            print(seg_map.shape)
            return resized_image, seg_map

    def detect_img(self, dataset):
        with tf.Session(graph=self.graph) as sess:
            res = {
                'imgs': [],
                'segs': []
                    }
            total_imgs = dataset.datasize
            for i in range(total_imgs):
                t1 = datetime.datetime.now()

                img = dataset.get_next_raw_image()
                batch_seg_map = sess.run(
                    self.predicts,
                    feed_dict={ self.input_images: img }
                    )
                seg_map = batch_seg_map[0]
                t2 = datetime.datetime.now()

                if len(seg_map.shape) >= 3:
                    seg_map = np.array(seg_map).argmax(axis=2)
                used = float((t2-t1).microseconds) / 1000.0
                res['imgs'].append(img[0])
                res['segs'].append(seg_map)
                if (i+1) % 10 == 0:
                    msg = "<<=====Finish Detection: {}, {}/{} time:{} (ms/pic)\
                    =======>>".format(float(i+1)/float(total_imgs),
                            i+1, total_imgs, used)
                    print(msg)
            return res




