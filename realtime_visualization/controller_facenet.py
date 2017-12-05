import cv2
import numpy as np
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from os.path import join as pjoin
import sys
import copy
import detect_face
import nn4 as network
import random

from numpy import *
from scipy.misc import imread
from scipy.misc import imresize
from numpy import random

from caffe_classes import class_names

from sklearn.externals import joblib

#face detection parameters
minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor
model_dir='./model_check_point/model.ckpt-500000'#"Directory containing the graph definition and checkpoint files.")
model_def= 'models.nn4'  # "Points to a module containing the definition of the inference graph.")
image_size=96 #"Image size (height, width) in pixels."
pool_type='MAX' #"The type of pooling to use for some of the inception layers {'MAX', 'L2'}.
use_lrn=False #"Enables Local Response Normalization after the first layers of the inception network."
seed=42,# "Random seed."
batch_size= None # "Number of images to process in a batch."
frame_interval=3 # frame intervals
graph_face = tf.Graph()  
with graph_face.as_default():
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, './model_check_point/')

class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        self.counter = 0
        self.current_image = 0
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
        print('Creating networks and loading parameters')
        #gpu_memory_fraction=1.0

    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        image_copy = image.copy()
        image = cv2.resize(image, (0,0), fx=0.25, fy=0.25) 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        w, h = gray.shape
        ret = np.empty((w, h, 3), dtype=np.uint8)
        ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = gray
        img = ret
        global graph_face
        with graph_face.as_default():
            bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        for face_position in bounding_boxes:
            face_position=face_position.astype(int)
            face_position = face_position * 4
            image_copy = cv2.rectangle(image_copy, (face_position[0], face_position[1]), (face_position[2], face_position[3]), (0, 255, 0), 2)
        ret, jpeg = cv2.imencode('.jpg', image_copy)
        return jpeg.tobytes()
