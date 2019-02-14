'''
USAGE python output_csv_results.py \
      threshold=0.5 \
      data_dir=path_to_data \
      model_path=path_to_model \
      output_path=path_to_output  \
      label_map_path=path_to_label_map
 

python output_csv_results.py threshold=0.8 data_dir=~/Desktop/models/research/object_detection/test_images model_path=~/Desktop/models/research/object_detection/car_detection_pb_graph/frozen_inference_graph.pb output_path=~/Desktop/output 
'''
import time
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
import zipfile
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import os.path


NUM_IMAGES_PER_ITERATION = 100
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

error = False

if len(sys.argv) != 6:
  print("Invalid arguments.")
  sys.exit(2)

if(sys.argv[1].find('threshold=') != -1):
  THRESHOLD = float(sys.argv[1].replace('threshold=', ''))
else:
  error = True

if(sys.argv[2].find('data_dir=') != -1):
  data_dir = sys.argv[2].replace('data_dir=', '')
  if data_dir[-1:] == "\\":
    print('asdfjkashdfksdafkl')
else:
  error = True

if(sys.argv[3].find('model_path=') != -1):
  model_path = sys.argv[3].replace('model_path=', '')
else:
  error = True

if(sys.argv[4].find('output_path=') != -1):
  output_path = sys.argv[4].replace('output_path=', '')
else:
  error = True

if(sys.argv[5].find('label_map=') != -1):
  label_map_path = sys.argv[5].replace('label_map=', '')
else:
  error = True

if error == True:
  print("Invalid arguments.")
  sys.exit(2)

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = model_path

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = label_map_path

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

PATH_TO_TEST_IMAGES_DIR = data_dir
TEST_IMAGE_PATHS = []
for image_path in os.listdir(data_dir):
  if image_path.find(".png") != -1:
    TEST_IMAGE_PATHS.append(os.path.join(PATH_TO_TEST_IMAGES_DIR , image_path))


def run_inference_for_multiple_images(images, graph):
  with graph.as_default():
    with tf.Session() as sess:
      output_dicts = []
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)

      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      for index, image in enumerate(images):
        image = np.expand_dims(image, axis=0)
        output_array = sess.run(tensor_dict, feed_dict={image_tensor: image})
        # all outputs are float32 numpy arrays, so convert types as appropriate
        formatted_output_array = []
        for i in range(len(output_array['num_detections'])):
          temp_output_dict = {}
          temp_output_dict['num_detections'] = int(output_array['num_detections'][i])
          temp_output_dict['detection_classes'] = output_array['detection_classes'][i].astype(np.uint8)
          temp_output_dict['detection_boxes'] = output_array['detection_boxes'][i]
          temp_output_dict['detection_scores'] = output_array['detection_scores'][i]
          formatted_output_array.append(temp_output_dict)

        output_dicts.append(formatted_output_array)
       

  return output_dicts


image_count = 0
f = open(output_path,"w+")
f.write('fileId,labelString\n')
TEST_IMAGE_PATHS.sort()
np_images = []
id_images = []
start = time.time()
reload_images = True
if os.path.isfile('image_np.npy') and os.path.isfile('image_id.npy'):
    reload_images = False

if reload_images:
  for image_path in TEST_IMAGE_PATHS: 
    image = Image.open(image_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    np_images.append(image_np)
    id_images.append(os.path.basename(image_path))
    image_count += 1
    print(image_count)

  np.save('image_np',np.array(np_images))
  np.save('image_id',id_images)
  print('Saved image_np and image_id')

else:
  np_images = np.load('./image_np.npy')
  np_images = list(np_images)
  id_images = np.load('./image_id.npy') 
  print('Loaded image_np and image_id.')

print("Total Test Images: ", len(TEST_IMAGE_PATHS))

num_loops = int(len(TEST_IMAGE_PATHS) / NUM_IMAGES_PER_ITERATION) + 1

output_array = []
output_array_section = []
np_images_section = []
# Actual detection.
for x in range(num_loops):
  beginning_index = x * NUM_IMAGES_PER_ITERATION
  end_index = beginning_index + NUM_IMAGES_PER_ITERATION

  if end_index > len(TEST_IMAGE_PATHS):
      end_index = len(TEST_IMAGE_PATHS) 
  

  print("Processing Images: ", beginning_index, end_index)
  output_array_section = run_inference_for_multiple_images(np_images[beginning_index:end_index], detection_graph)
  for tmp_output_dict,image_np,image_id in zip(output_array_section,np_images[beginning_index:end_index],id_images[beginning_index:end_index]):
    output_dict = tmp_output_dict[0]
    f.write(image_id + ',')  
    index = 0
    counter = 0
    for score in output_dict['detection_scores']:
      if score > THRESHOLD:
        counter += 1
    counter2 = counter
    if counter > 1:
      f.write("\"")
    for score in output_dict['detection_scores']:
      if score > THRESHOLD:
        if str(output_dict['detection_classes'][index]) == '1':
          f.write("car ")
        else:
          f.write("pedestrian ")
        boundingBoxValues = []
        f.write(str(score) + " ")
        boundingBox = str(output_dict['detection_boxes'][index])
        boundingBox = boundingBox.replace("[", "")
        boundingBox = boundingBox.replace("]", "")
        boundingBoxValues = boundingBox.split()
        newBoundingBox = ""
        newBoundingBox = boundingBoxValues[1] + " " + boundingBoxValues[0] + " " + boundingBoxValues[3] + " " + boundingBoxValues[2]

        f.write(newBoundingBox)
        if counter2 == 1 and counter > 1:
          f.write("\"")
        f.write("\n")
        counter2 -= 1
        index += 1
  
    if counter == 0:
      f.write("\n")

    image_count += 1

end = time.time()
print('Total time:')
print(end - start)

f.close()


