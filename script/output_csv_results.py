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


NUM_IMAGES_PER_ITERATION = 5
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

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = data_dir
TEST_IMAGE_PATHS = []
for image_path in os.listdir(data_dir):
  if image_path.find(".png") != -1:
    TEST_IMAGE_PATHS.append(os.path.join(PATH_TO_TEST_IMAGES_DIR , image_path))

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image[0].shape[0], image[0].shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_array = sess.run(tensor_dict,
                             feed_dict={image_tensor: image})
      # all outputs are float32 numpy arrays, so convert types as appropriate
      formatted_output_array = []
      for i in range(len(output_array['num_detections'])):
        temp_output_dict = {}
        temp_output_dict['num_detections'] = int(output_array['num_detections'][i])
        temp_output_dict['detection_classes'] = output_array['detection_classes'][i].astype(np.uint8)
        temp_output_dict['detection_boxes'] = output_array['detection_boxes'][i]
        temp_output_dict['detection_scores'] = output_array['detection_scores'][i]
        if 'detection_masks' in output_array:
          temp_output_dict['detection_masks'] = output_array['detection_masks'][i]
        formatted_output_array.append(temp_output_dict)
      
  return formatted_output_array


image_count = 0
f = open(output_path,"w+")
f.write('file Id,label - confidence - bounding box\n')
TEST_IMAGE_PATHS.sort()
np_images = []
start = time.time()
save_dir = './np_save'
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
    image_count += 1
    print(image_count)

  np.save(save_dir,np.array(np_images))
  print('Saved image_np.')

else:
  np_images = np.load('./np_save.npy')
  np_images = list(np_images)
  print('Loaded image_np.')

print("Total Test Images: ", len(TEST_IMAGE_PATHS))

num_loops = int(len(TEST_IMAGE_PATHS) / NUM_IMAGES_PER_ITERATION)

output_array = []
output_array_section = []
np_images_section = []
# Actual detection.

for x in range(0, num_loops):
  beginning_index = x * NUM_IMAGES_PER_ITERATION
  end_index = beginning_index + NUM_IMAGES_PER_ITERATION
  print("Processing Images: ", beginning_index, end_index)
  output_array_section = run_inference_for_single_image(np_images[beginning_index:end_index], detection_graph)
  for output_dict,image_np in zip(output_array_section,np_images[beginning_index:end_index]):
    '''
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        min_score_thresh=THRESHOLD,
        line_thickness=3)
    '''
    f.write(image_path.replace(data_dir, "").replace("/", "") + ',')  
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
    #plt.figure(figsize=IMAGE_SIZE)
    #plt.imshow(image_np)
    #img = Image.fromarray(image_np, 'RGB')
    #img.show()
    #image_name = image_path.replace(data_dir, "")
    #img.save(os.path.join(output_path, image_name))
    image_count += 1

end = time.time()
print('Total time:')
print(end - start)

f.close()


