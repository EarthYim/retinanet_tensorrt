import tensorflow as tf
from keras_retinanet.utils.image import preprocess_image
from keras_retinanet.utils.image import resize_image
from keras_retinanet.utils.image import read_image_bgr
from keras_retinanet.utils.visualization import draw_box, draw_caption
import cv2 as cv
import numpy as np

output_names = ['filtered_detections/map/TensorArrayStack/TensorArrayGatherV3', 'filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3', 'filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3']
input_names = ['input_1']

def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk."""
    with tf.gfile.FastGFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def

def load_trt(path):
    path = './model/trt_graph.pb' # remove this after testing
    trt_graph = get_frozen_graph(path) 

    # Create session and load graph
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_sess = tf.Session(config=tf_config)
    tf.import_graph_def(trt_graph, name='')


#    Get graph input size
    for node in trt_graph.node:
        if 'input_' in node.name:
            size = node.attr['shape'].shape
            image_size = [size.dim[i].size for i in range(1, 4)]
            break
    print("image_size: {}".format(image_size))


    # input and output tensor names.
    input_tensor_name = input_names[0] + ":0"
    output_tensor_name_0 = output_names[0] + ":0"
    output_tensor_name_1 = output_names[1] + ":0"
    output_tensor_name_2 = output_names[2] + ":0"
    print("input_tensor_name: {}\noutput_tensor_name: {}".format(
        input_tensor_name, output_tensor_name_0))

    output_tensor_0 = tf_sess.graph.get_tensor_by_name(output_tensor_name_0)
    output_tensor_1 = tf_sess.graph.get_tensor_by_name(output_tensor_name_1)
    output_tensor_2 = tf_sess.graph.get_tensor_by_name(output_tensor_name_2)

    return (input_tensor_name, output_tensor_0, output_tensor_1, output_tensor_2)

def make_inference(img):
    
    image = read_image_bgr(img)
    #draw = image.copy()

    image = preprocess_image(image)
    (image, scale) = resize_image(image)
    image = np.expand_dims(image, axis=0)

    graph = load_trt(trt_path)

    feed_dict = {graph[0]: image}
    preds = tf_sess.run([graph[1], graph[2], graph[3]], feed_dict)
    boxes, scores, labels = preds[0], preds[1], preds[2]

    return boxes, scores, labels


from imutils import paths
from time import time
dura = []

img_path = '../retinanet/src'
trt_path = None
imagePaths = list(paths.list_files(img_path))

id = {0:'gate', 1:"flare"}

for i in imagePaths:
    start_time = time()

    boxes, scores, labels = make_inference(i)
    elasp = time() - start_time
    print(elasp)
    dura.append(elasp)

dura = np.array(dura)
print("\nw/ tensorrt mean: {}".format(np.mean(dura)))
