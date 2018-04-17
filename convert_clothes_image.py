"""Converts PASCAL VOC 2012 data to TFRecord file format with Example protos.

PASCAL VOC 2012 dataset is expected to have the following directory structure:

  + pascal_voc_seg
    - build_data.py
    - build_voc2012_data.py (current working directory).
    + VOCdevkit
      + VOC2012
        + JPEGImages
        + SegmentationClass
        + ImageSets
          + Segmentation
    + tfrecord

Image folder:
  ./VOCdevkit/VOC2012/JPEGImages

Semantic segmentation annotations:
  ./VOCdevkit/VOC2012/SegmentationClass

list folder:
  ./VOCdevkit/VOC2012/ImageSets/Segmentation

This script converts data into sharded data files and save at tfrecord folder.

The Example proto contains the following fields:

  image/encoded: encoded image content.
  image/filename: image filename.
  image/format: image file format.
  image/height: image height.
  image/width: image width.
  image/channels: image channels.
  image/segmentation/class/encoded: encoded semantic segmentation content.
  image/segmentation/class/format: semantic segmentation file format.
"""
import math
import os.path
import sys
import build_data
import tensorflow as tf
import numpy as np
from PIL import Image

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('image_folder',
                           '/home/liyongbin/Mask_RCNN_BBox/',
                           'Folder containing images.')

tf.app.flags.DEFINE_string(
    'semantic_segmentation_folder',
    './train_img/train_parse_annotation/val_segmentations',
    'Folder containing semantic segmentation annotations.')

tf.app.flags.DEFINE_string(
    'list_folder',
    '/home/liyongbin/Mask_RCNN_BBox/',
    'Folder containing lists for training and validation')

tf.app.flags.DEFINE_string(
    'output_dir',
    '/home/liyongbin/train_clothes_img/tfcord',
    'Path to save converted SSTable of TensorFlow examples.')



_NUM_SHARDS = 4


def _convert_dataset(dataset_split):
  """Converts the specified dataset split to TFRecord format.

  Args:
    dataset_split: The dataset split (e.g., train, test).

  Raises:
    RuntimeError: If loaded image and label have different shape.
  """
  dataset = os.path.basename(dataset_split)[:-4]
  sys.stdout.write('Processing ' + dataset)
  filenames = [x.strip('\n').split(",")[0] for x in open(dataset_split, 'r')]

  segs = [x.strip('\n').split(",")[2:] for x in open(dataset_split, 'r')]
  filenames = filenames[1:]
  segs = segs[1:]

  num_images = len(filenames)
  num_per_shard = int(math.ceil(num_images / float(_NUM_SHARDS)))

  image_reader = build_data.ImageReader('jpeg', channels=3)
  label_reader = build_data.ImageReader('png', channels=1)

  for shard_id in range(_NUM_SHARDS):
    output_filename = os.path.join(
        FLAGS.output_dir,
        '%s-%05d-of-%05d.tfrecord' % (dataset, shard_id, _NUM_SHARDS))

    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
      start_idx = shard_id * num_per_shard
      end_idx = min((shard_id + 1) * num_per_shard, num_images)
      for i in range(start_idx, end_idx):
        sys.stdout.write('\r>> Converting image %d/%d shard %d ' % (
            i + 1, len(filenames), shard_id))
        sys.stdout.flush()
        # Read the image.
        image_filename = os.path.join(
            FLAGS.image_folder, filenames[i])
        image_data = tf.gfile.FastGFile(image_filename, 'rb').read()

        height,width = image_reader.read_image_dims(image_data)

        seg_label = segs[i]
        seg_data = np.zeros([height,width,1])
        print(height, width)
        for kp_index,kp_info in enumerate(seg_label):
            items = [int(x) for x in kp_info.split("_")]
            if items[2] >= 0:
                seg_data[min(items[0],height-1),min(items[1],width-1),[0]] = kp_index+1
        #seg_data = Image.fromarray(seg_data)
        seg_data = label_reader.encode_image(seg_data)

        example = build_data.image_seg_to_tfexample(
            image_data, filenames[i], height, width, seg_data)
        tfrecord_writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()


def main(unused_argv):
  dataset_splits = tf.gfile.Glob(os.path.join(FLAGS.list_folder, 'train.csv'))
  for dataset_split in dataset_splits:
    print(dataset_split)
    _convert_dataset(dataset_split)


if __name__ == '__main__':
  tf.app.run()
