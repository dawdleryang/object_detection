import sys
sys.path.append('..')
import os
import json
import time
import tensorflow as tf
import numpy as np
from PIL import Image
from utils import label_map_util
if len(sys.argv) < 5:
    print('Usage: python {} output_json_path checkpoint_path test_ids_path image_dir'.format(sys.argv[0]))
    exit()
PATH_OUTPUT = sys.argv[1]
PATH_TO_CKPT = sys.argv[2]
PATH_TEST_IDS = sys.argv[3]
DIR_IMAGE = sys.argv[4]
PATH_TO_LABELS = '/home/xuleiyang/kitti/data/label_map.pbtxt'
NUM_CLASSES = 21
def get_results(boxes, classes, scores, category_index, im_width, im_height,
    min_score_thresh=.5):
    bboxes = list()
    for i, box in enumerate(boxes):
        if scores[i] > min_score_thresh:
            ymin, xmin, ymax, xmax = box
            bbox = {
                'bbox': {
                    'xmax': xmax * im_width,
                    'xmin': xmin * im_width,
                    'ymax': ymax * im_height,
                    'ymin': ymin * im_height
                },
                'category': category_index[classes[i]]['name'],
                'score': float(scores[i])
            }
            bboxes.append(bbox)
    #import pdb; pdb.set_trace()
    return bboxes
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
test_ids = [line.split()[0] for line in open(PATH_TEST_IDS)]
total_time = 0
test_annos = dict()
flag = False

classes = ['car','pedestrian','unknown']

with detection_graph.as_default():
    with tf.Session(graph=detection_graph, config=config) as sess:

        test_ids.sort()
        submit_dict = {'id': [], 'result': []}         

        for image_id in test_ids:
            submit_line = ""
            image_path = os.path.join(DIR_IMAGE, image_id) # + '.jpg')
            image = Image.open(image_path)
            image_np = np.array(image).astype(np.uint8)
            im_width, im_height, _ = image_np.shape
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            start_time = time.time()
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            end_time = time.time()
            print('{} {} {:.3f}s'.format(time.ctime(), image_id, end_time - start_time))
            
            #import pdb; pdb.set_trace()
            '''
            for i in range(num_detections):
                box = boxes[i]
                cls = classes[i]
                scr = scores[i]
                
                xmin_norm = float(box['xmin']) #/ width)
                ymin_norm = float(box['ymin']) #/ height)
                xmax_norm = float(box['xmax']) #/ width)
                ymax_norm = float(box['ymax']) #/ height)
           
                #class_name = strings[class_id]
                #submit_line +="{} {} {} {} {}".format(class_name, xmin_norm, ymin_norm, xmax_norm, ymax_norm)
                #if count < len(data['anno_result']):
                #    submit_line += '\n'


            submit_dict['id'].append(image_id)
            submit_dict['result'].append(submit_line)
           
            '''


 
            if flag:
                total_time += end_time - start_time
            else:
                flag = True
            test_annos[image_id] = {'objects': get_results(
                np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores), category_index,
                im_width, im_height)}

            
            #for anno in test_annos:
            #    cls = anno['category']
            #    box = anno['boxes'] 
                


print('total time: {}, total images: {}, average time: {}'.format(
    total_time, len(test_ids), total_time / len(test_ids)))
test_annos = {'imgs': test_annos}
fd = open(PATH_OUTPUT, 'w')
json.dump(test_annos, fd)
fd.close()
