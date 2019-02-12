from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import csv
import hashlib
import io
import os

import numpy as np
import PIL.Image as pil
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
from object_detection.utils.np_box_ops import iou

import pandas as pd 


tf.app.flags.DEFINE_string('image_dir', '', 'Location of images ')
tf.app.flags.DEFINE_string('output_path', '', 'Path to which TFRecord files wiil be written')
tf.app.flags.DEFINE_string('csv_file', '', 'Path of csv file')
tf.app.flags.DEFINE_integer('validation_set_size', '500', 'Number of images to be used as a validation set.')
tf.app.flags.DEFINE_string('resize', '','resize the image to aim size "width, height" like "200,3000"')

FLAGS = tf.app.flags.FLAGS


def convert_csv_to_tfrecords(image_dir, output_path, csv_file, validation_set_size):
  train_count = 0
  val_count = 0

  train_writer = tf.python_io.TFRecordWriter('%s_train.tfrecord'%
                                             output_path)
  val_writer = tf.python_io.TFRecordWriter('%s_val.tfrecord'%
                                           output_path)
  data = pd.read_csv(csv_file)
  data = data.fillna(0) 

  img_num = 0
  for i in range(len(data)):
    img_id = data['fileId'][i]
    result = data['labelString'][i]
    tmp = result.split('\n')
    img_path = os.path.join(image_dir+img_id)
    print("Image Name: " + img_id)
    is_validation_img = img_num < validation_set_size
    img_num += 1
    box = []
    label = []
    for j in range(len(tmp)): 
      tmp_j = tmp[j].split(' ')
      box.append([float(tmp_j[1]),float(tmp_j[2]),float(tmp_j[3]),float(tmp_j[4])])
      label.append(tmp_j[0])

    #example = prepare_example(img_path, label, box)
    example = prepare_example_1(img_path, tmp)
    if is_validation_img:
      val_writer.write(example.SerializeToString())
      val_count += 1
    else:
      train_writer.write(example.SerializeToString())
      train_count += 1

  train_writer.close()
  val_writer.close()

def prepare_example_1(image_path, anno):
  image_id = os.path.basename(image_path)
  with tf.gfile.GFile(image_path, 'rb') as fid:
    encoded_png = fid.read()
  encoded_png_io = io.BytesIO(encoded_png)
  image = pil.open(encoded_png_io)

  key = hashlib.sha256(encoded_png).hexdigest()
  width,height = image.size
  #import pdb; pdb.set_trace()
  xmin_norm = []
  ymin_norm = []
  xmax_norm = []
  ymax_norm = []
  label = []
  for j in range(len(anno)):
      anno_j = anno[j].split(' ')
      xmin_norm.append(float(anno_j[1]))
      ymin_norm.append(float(anno_j[2]))
      xmax_norm.append(float(anno_j[3]))
      ymax_norm.append(float(anno_j[4]))
      label.append(anno_j[0])


  # resize image
  #new_width,new_height = tuple(map(int,FLAGS.resize.split(',')))
  #image = image.resize([new_width,new_height],pil.LANCZOS)


  img_byte_arr = io.BytesIO()
  image.save(img_byte_arr,format='PNG',quality=100)
  encoded_png = img_byte_arr.getvalue()

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(image_id.encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(image_id.encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_png),
      'image/format': dataset_util.bytes_feature('png'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin_norm),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax_norm),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin_norm),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax_norm),
      'image/object/class/text': dataset_util.bytes_list_feature(x.encode('utf8') for x in label),
  }))

  return example



def prepare_example(image_path, label, rect):
  image_id = os.path.basename(image_path)
  with tf.gfile.GFile(image_path, 'rb') as fid:
    encoded_png = fid.read()
  encoded_png_io = io.BytesIO(encoded_png)
  image = pil.open(encoded_png_io)
  
  key = hashlib.sha256(encoded_png).hexdigest()
  width,height = image.size
  import pdb; pdb.set_trace()
  xmin_norm = rect[0] 
  xmax_norm = rect[2] 
  ymax_norm = rect[3]
  
  # resize image
  #new_width,new_height = tuple(map(int,FLAGS.resize.split(',')))
  #image = image.resize([new_width,new_height],pil.LANCZOS)


  img_byte_arr = io.BytesIO()
  image.save(img_byte_arr,format='PNG',quality=100)
  encoded_png = img_byte_arr.getvalue()

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(image_id.encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(image_id.encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_png),
      'image/format': dataset_util.bytes_feature('png'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin_norm),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax_norm),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin_norm),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax_norm),
      'image/object/class/text': dataset_util.bytes_list_feature(label),
  }))

  return example


def main(_):
  convert_csv_to_tfrecords(
      image_dir=FLAGS.image_dir,
      output_path=FLAGS.output_path,
      csv_file=FLAGS.csv_file,
      validation_set_size=FLAGS.validation_set_size)

if __name__ == '__main__':
  tf.app.run()
