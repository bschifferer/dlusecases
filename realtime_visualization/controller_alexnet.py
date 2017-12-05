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
#facenet embedding parameters
model_dir='./model_check_point/model.ckpt-500000'#"Directory containing the graph definition and checkpoint files.")
model_def= 'models.nn4'  # "Points to a module containing the definition of the inference graph.")
image_size=96 #"Image size (height, width) in pixels."
pool_type='MAX' #"The type of pooling to use for some of the inception layers {'MAX', 'L2'}.
use_lrn=False #"Enables Local Response Normalization after the first layers of the inception network."
seed=42,# "Random seed."
batch_size= None # "Number of images to process in a batch."
frame_interval=3 # frame intervals
print('Init face detection')
graph_face = tf.Graph()  
with graph_face.as_default():
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, './model_check_point/')

###Alexnet
def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups =  tf.split(input, group, 3)   #tf.split(3, group, input)
        kernel_groups = tf.split(kernel, group, 3)  #tf.split(3, group, kernel) 
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)          #tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])

graph_alex = tf.Graph()  
with graph_alex.as_default():
    net_data = load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()
    train_x = zeros((1, 227,227,3)).astype(float32)
    train_y = zeros((1, 1000))
    xdim = train_x.shape[1:]
    ydim = train_y.shape[1]
    
    
    
    x = tf.placeholder(tf.float32, (None,) + xdim)
    #conv1
    #conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
    k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
    conv1W = tf.Variable(net_data["conv1"][0])
    conv1b = tf.Variable(net_data["conv1"][1])
    conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
    conv1 = tf.nn.relu(conv1_in)
    
    #lrn1
    #lrn(2, 2e-05, 0.75, name='norm1')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn1 = tf.nn.local_response_normalization(conv1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)
    
    #maxpool1
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
    
    
    #conv2
    #conv(5, 5, 256, 1, 1, group=2, name='conv2')
    k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv2W = tf.Variable(net_data["conv2"][0])
    conv2b = tf.Variable(net_data["conv2"][1])
    conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv2 = tf.nn.relu(conv2_in)
    
    
    #lrn2
    #lrn(2, 2e-05, 0.75, name='norm2')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn2 = tf.nn.local_response_normalization(conv2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)
    
    #maxpool2
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
    
    #conv3
    #conv(3, 3, 384, 1, 1, name='conv3')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
    conv3W = tf.Variable(net_data["conv3"][0])
    conv3b = tf.Variable(net_data["conv3"][1])
    conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv3 = tf.nn.relu(conv3_in)
    
    #conv4
    #conv(3, 3, 384, 1, 1, group=2, name='conv4')
    #k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
    #conv4W = tf.Variable(net_data["conv4"][0])
    #conv4b = tf.Variable(net_data["conv4"][1])
    #conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    #conv4 = tf.nn.relu(conv4_in)
    
    
    #conv5
    #conv(3, 3, 256, 1, 1, group=2, name='conv5')
    #k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
    #conv5W = tf.Variable(net_data["conv5"][0])
    #conv5b = tf.Variable(net_data["conv5"][1])
    #conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    #conv5 = tf.nn.relu(conv5_in)
    
    #maxpool5
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
    #k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    #maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
    
    #fc6
    #fc(4096, name='fc6')
    #fc6W = tf.Variable(net_data["fc6"][0])
    #fc6b = tf.Variable(net_data["fc6"][1])
    #fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)
    
    #fc7
    #fc(4096, name='fc7')
    #fc7W = tf.Variable(net_data["fc7"][0])
    #fc7b = tf.Variable(net_data["fc7"][1])
    #fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)
    
    #fc8
    #fc(1000, relu=False, name='fc8')
    #fc8W = tf.Variable(net_data["fc8"][0])
    #fc8b = tf.Variable(net_data["fc8"][1])
    #fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
    
    #prob
    #softmax(name='prob'))
    #prob = tf.nn.softmax(fc8)
    net_data=0
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)



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
        image2 = image.copy()
        image3 = image.copy()
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
            image2 = cv2.rectangle(image2, (face_position[0], face_position[1]), (face_position[2], face_position[3]), (0, 255, 0), 2)
        ret, jpeg = cv2.imencode('.jpg', image2)
        
        output_rows = 480
        output_cols = 1000
        max_cols = 0
        nn_size = 227
        
        img_alex = cv2.resize(image3, (int(nn_size*(image3.shape[1]/image3.shape[0])),nn_size))
        img_crop = img_alex[:,(int((img_alex.shape[1]-nn_size)/2)):(int((img_alex.shape[1]-nn_size)/2)+nn_size)]
        		
        output_alexnet = np.zeros((output_rows, output_cols, 3), dtype=np.uint8)
        output_alexnet.fill(255)
        output_alexnet[(int((output_rows-img_crop.shape[0])/2)):(int((output_rows-img_crop.shape[0])/2)+img_crop.shape[0]),0:img_crop.shape[1],:] = np.asarray(img_crop)
        max_cols = img_crop.shape[1] + 20
        #print(image_crop.shape)
        global graph_alex
        with graph_alex.as_default():
            output = sess.run([conv1_in, conv2_in, conv3_in], feed_dict = {x:[img_crop]})
        
        #output_object = np.mean(np.asarray(output_object)[0,:,:,], axis=2) #conv_in1,1 mit 20; conv_in1_10, conv2_in_20; 10;15  ;;9;35
        output_puffer_total = np.zeros((nn_size, nn_size, 3), dtype=np.uint8)
        output_puffer_total.fill(0)
        for i,j in enumerate([10,15,20,55]):
            output_object = output[0][0,:,:,j]
            output_puffer = np.zeros((output_object.shape[0], output_object.shape[1], 3), dtype=np.uint8)
            output_puffer.fill(0)
            output_puffer[:,:,0] = output_puffer[:,:,1] = output_puffer[:,:,2] = output_object
            output_puffer = cv2.resize(output_puffer, (int(nn_size/2),int(nn_size/2)))
            if i == 0:
                output_puffer_total[0:113,0:113,:] = output_puffer
            if i == 1:
                output_puffer_total[113:226,0:113,:] = output_puffer
            if i == 2:
                output_puffer_total[0:113,113:226,:] = output_puffer
            if i == 3:
                output_puffer_total[113:226,113:226,:] = output_puffer
        
        
        output_alexnet[(int((output_rows-output_puffer_total.shape[0])/2)):(int((output_rows-output_puffer_total.shape[0])/2)+output_puffer_total.shape[0]),max_cols:(max_cols+output_puffer_total.shape[1]),:] = np.asarray(output_puffer_total)
        max_cols = 20 + (max_cols+output_puffer_total.shape[1])
        
        output_puffer_total = np.zeros((nn_size, nn_size, 3), dtype=np.uint8)
        output_puffer_total.fill(0)
        for i,j in enumerate([20,1,5,40]):
            output_object = output[1][0,:,:,j]
            output_puffer = np.zeros((output_object.shape[0], output_object.shape[1], 3), dtype=np.uint8)
            output_puffer.fill(0)
            output_puffer[:,:,0] = output_puffer[:,:,1] = output_puffer[:,:,2] = output_object
            output_puffer = cv2.resize(output_puffer, (int(nn_size/2),int(nn_size/2)))
            if i == 0:
                output_puffer_total[0:113,0:113,:] = output_puffer
            if i == 1:
                output_puffer_total[113:226,0:113,:] = output_puffer
            if i == 2:
                output_puffer_total[0:113,113:226,:] = output_puffer
            if i == 3:
                output_puffer_total[113:226,113:226,:] = output_puffer
        
        
        output_alexnet[(int((output_rows-output_puffer_total.shape[0])/2)):(int((output_rows-output_puffer_total.shape[0])/2)+output_puffer_total.shape[0]),max_cols:(max_cols+output_puffer_total.shape[1]),:] = np.asarray(output_puffer_total)
        max_cols = 20 + (max_cols+output_puffer_total.shape[1])
		
        output_puffer_total = np.zeros((nn_size, nn_size, 3), dtype=np.uint8)
        output_puffer_total.fill(0)
        for i,j in enumerate([20,1,5,40]):
            output_object = output[2][0,:,:,j]
            output_puffer = np.zeros((output_object.shape[0], output_object.shape[1], 3), dtype=np.uint8)
            output_puffer.fill(0)
            output_puffer[:,:,0] = output_puffer[:,:,1] = output_puffer[:,:,2] = output_object
            output_puffer = cv2.resize(output_puffer, (int(nn_size/2),int(nn_size/2)))
            if i == 0:
                output_puffer_total[0:113,0:113,:] = output_puffer
            if i == 1:
                output_puffer_total[113:226,0:113,:] = output_puffer
            if i == 2:
                output_puffer_total[0:113,113:226,:] = output_puffer
            if i == 3:
                output_puffer_total[113:226,113:226,:] = output_puffer
        
        output_alexnet[(int((output_rows-output_puffer_total.shape[0])/2)):(int((output_rows-output_puffer_total.shape[0])/2)+output_puffer_total.shape[0]),max_cols:(max_cols+output_puffer_total.shape[1]),:] = np.asarray(output_puffer_total)
        max_cols = 20 + (max_cols+output_puffer_total.shape[1])
		
        #print(output.shape)
        #for input_im_ind in range(output.shape[0]):
            #inds = argsort(output)[input_im_ind,:]
            #for i in range(5):
                #print(class_names[inds[-1-i]], output[input_im_ind, inds[-1-i]])
        ret2, jpeg2 = cv2.imencode('.jpg', output_alexnet)
        return [jpeg.tobytes(), jpeg2.tobytes()]
    
    def to_rgb(img):
        w, h = img.shape
        ret = np.empty((w, h, 3), dtype=np.uint8)
        ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
        return ret