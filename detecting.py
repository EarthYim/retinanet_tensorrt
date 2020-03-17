from rt_lib import make_inference
from retinanet_image import read_image_bgr, preprocess_image, resize_image, draw_box, draw_caption, draw_boxes
import numpy as np
import cv2 as cv
import argparse
from imutils import paths
import os
from time import time

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to image")
ap.add_argument("-o","--output", required=True, help="output images path")
args = vars(ap.parse_args())

imagePaths = list(paths.list_files(args["input"]))

id = {0:'gate', 1:'flare'}

dura = []

for i in imagePaths:

    fname = i.split(os.path.sep)[-1]
    
    start_time = time()

    image = read_image_bgr(i)
    draw = image.copy()
    image = preprocess_image(image)
    (image, scale) = resize_image(image)
    image = np.expand_dims(image, axis=0)

    graph = load_trt()

    (boxes, scores, labels) = make_inference(image, graph)
    
    #filtering and drawing
    for (box, score, label) in zip(boxes[0], scores[0], labels[0]):
        
        if score < 0.3:
            continue
        
        box /= scale
        box = box.astype("int")
        x, y, x_max, y_max = box[0], box[1], box[2], box[3] #output x, y, xmax, ymax
        draw_box(draw, box, color=(0,0,255))
        caption = "{} {:.3f}".format(id[label], score) #output label
        draw_caption(draw, box, caption)
    
    per_frame = time() - start_time
    print(per_frame)
    dura.append(per_frame)
    
    #cv.imwrite(args["output"]+'/'+fname, draw)

dura = np.array(dura)
print("\n mean: {}".format(np.mean(dura)))


        
