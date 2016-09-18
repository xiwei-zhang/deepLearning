import tensorflow as tf
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

import basicOperations

import ipdb
import time
import re

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# ipdb.set_trace()

## define constant
input_path = "/home/seawave/work/you/src/ma/script/launch_script/imgProcResult3"
TP_path = "/home/seawave/work/you/src/ma/script/launch_script/crops/TP"
FP_path = "/home/seawave/work/you/src/ma/script/launch_script/crops/FP"
output_path = "/home/seawave/work/you/src/ma/script/launch_script/CNNResult"
CNN_model_path = "/home/seawave/work/you/src/ma/script/launch_script/CNNModel/model.ckpt"

pattern1 = ".+/(?P<folder>[0-9a-zA-Z]+)_(?P<image>[0-9a-zA-Z]+).txt"
re1 = re.compile(pattern1)
R = 10 # 51*51
size = R*2
R_FP_TP  = 10 # ratio of FP/TP
drop_out = 0.5 #0.5 

FLIP = 0
ROTATION = 0
BALANCE_SETS = 1
# train_image_set = np.linspace(0,10,11).astype(np.uint16)
# train_image_set = np.linspace(0,146,147).astype(np.uint16)
# train_image_set = np.linspace(0,99,100).astype(np.uint16)
# test_image_set = np.linspace(147,148,1).astype(np.uint16)

imResizeTable = {1440:10, 1504:10, 2048:12, 2544:15}

def getTrainingData():
   print "testImage", testIdx
   tempSet = np.linspace(0,147,148).astype(np.uint16)
   train_image_set = np.delete(tempSet, testIdx)
   test_image_set = np.array([testIdx])

   
   trainning_set = []
   trainning_class = []
   ### get train image ###
   for imageIdx in train_image_set:
       txt_file = txt_files[imageIdx]
       m1 = re1.match(txt_file)
       folder = m1.groupdict()['folder']
       image = m1.groupdict()['image']
       
       for ASF_image in ASF_files:
           m2 = re.search(folder, ASF_image)
           m3 = re.search(image, ASF_image)
           if m2 != None and m3 != None:
               break
        
       imASF = misc.imread(ASF_image)
       axis0 = imASF.shape[0]
       axis1 = imASF.shape[1]
       R0 = imResizeTable[axis1]
       imASFPadd = np.ndarray(shape=(axis0+R0*2,axis1+R0*2), dtype='uint8')
       imASFPadd[R0:axis0+R0, R0:axis1+R0] = imASF
       # imASFPadd = imASFPadd / 255.0
   
       ## Filter candidates
       ##  Keep all TP candidates
       ##  Random select 2 times of FP candidates
       candi_file = open(txt_file, 'r')
       lines = candi_file.readlines()
   
       TPCandiIdxList = []
       FPCandiIdxList = []
       for lineIdx in range(len(lines)):
           words = lines[lineIdx].split()
           if words[-1] == '1':
               TPCandiIdxList.append(lineIdx)
           else:
               FPCandiIdxList.append(lineIdx)
   
       ##  inject FP idx to TP list
       FPCandiIdxRand = np.random.permutation(FPCandiIdxList)
       NFPcandi = R_FP_TP * len(TPCandiIdxList)
       NFPcandi = min(NFPcandi, len(FPCandiIdxRand))
       for candiIdx in range(NFPcandi):
           TPCandiIdxList.append( FPCandiIdxRand[candiIdx] ) 
       
       ##  Load candidates into training set
       for candiIdx in TPCandiIdxList:
           line = lines[candiIdx]
           words = line.split()
           x_center = int(words[0]) + R0
           y_center = int(words[1]) + R0
           
           imCrop = imASFPadd[y_center-R0 : y_center+R0, x_center-R0:x_center+R0]
           if (axis1 != 1440 and axis1 != 1504):
               imCrop = misc.imresize(imCrop,(size,size))
           if (imCrop.shape != (size,size)):
               print "ERROR !!!!"
           trainning_set.append(imCrop)

           if words[-1] == '1':
               trainning_class.append([0,1])
           else:
               trainning_class.append([1,0])

           ##  Start FLIPPING
           if words[-1] == '1':
               if FLIP:
                    ## Horizental flipping
                    imFLIP = imCrop[:,::-1]
                    trainning_set.append(imFLIP)
                    ## Vertical flipping
                    imFLIP = imCrop[::-1,:]
                    trainning_set.append(imFLIP)

                    ## add training class
                    trainning_class.append([0,1])
                    trainning_class.append([0,1])
               if ROTATION:
                    ## Rotate 90 counter clock wise
                    imROT90 = np.rot90(imCrop)
                    trainning_set.append(imROT90)
                    imROT90 = np.rot90(imROT90)
                    trainning_set.append(imROT90)
                    imROT90 = np.rot90(imROT90)
                    trainning_set.append(imROT90)

                    trainning_class.append([0,1])
                    trainning_class.append([0,1])
                    trainning_class.append([0,1])
               if BALANCE_SETS:
                    for ii in range(R_FP_TP/2):
                        trainning_set.append(imCrop)
                        trainning_class.append([0,1])

                    
   array_trainning_set = np.reshape(trainning_set, (len(trainning_set), size*size))
   array_trainning_set = array_trainning_set /255.0
   array_trainning_class = np.reshape(trainning_class, (len(trainning_class), 2))

   return (array_trainning_set, array_trainning_class, test_image_set)


## CNN functions
def weight_variable(shape, name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name)

def bias_variable(shape, name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name)
  
#Convolution and Pooling
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
                        



if __name__ == "__main__":

    txt_files = basicOperations.getFiles(input_path, "txt")
    ASF_files = basicOperations.getFiles(input_path, "png", "ASF")
    MACandi_files = basicOperations.getFiles(input_path, "png", "MaCandi")

    for testIdx in range(148):
        size_pool = size
        #### Get training data
        array_trainning_set, array_trainning_class, test_image_set = getTrainingData()

        ###################################################################3
        #### SETUP CNN
        # tf.set_random_seed(1234) 

        sess = tf.InteractiveSession()
        x = tf.placeholder(tf.float32, [None, size*size])
        y_ = tf.placeholder(tf.float32, [None, 2])
        
        #First Convolutional Layer
        W_conv1 = weight_variable([5, 5, 1, 32], "W_conv1")
        b_conv1 = bias_variable([32], "b_conv1")
        x_image = tf.reshape(x, [-1,size,size,1])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)
        N_norm1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
        size_pool /= 2
        
        #Second Convolutional Layer
        W_conv2 = weight_variable([3, 3, 32, 64], "W_conv2")
        b_conv2 = bias_variable([64], "b_conv2")
        
        h_conv2 = tf.nn.relu(conv2d(N_norm1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)
        N_norm2 = tf.nn.lrn(h_pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
        size_pool /= 2
        
        #Third Convolutional Layer
        W_conv3 = weight_variable([3, 3, 64, 64], "W_conv3")
        b_conv3 = bias_variable([64], "b_conv3")
        
        h_conv3 = tf.nn.relu(conv2d(N_norm2, W_conv3) + b_conv3)
        # h_pool3 = max_pool_2x2(h_conv3)
        N_norm3 = tf.nn.lrn(h_conv3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
        # size_pool /= 2

        ##Fourth Convolutional Layer
        ## W_conv4 = weight_variable([3, 3, 128, 128])
        ## b_conv4 = bias_variable([128])
        ## 
        ## h_conv4 = tf.nn.relu(conv2d(N_norm3, W_conv4) + b_conv4)
        ## N_norm4 = tf.nn.lrn(h_conv4, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

        #Densely Connected Layer 1
        W_fc1 = weight_variable([size_pool * size_pool * 64, 128], "W_fc1")
        b_fc1 = bias_variable([128], "b_fc1")
        
        N_norm4_flat = tf.reshape(N_norm3, [-1, size_pool * size_pool * 64])
        h_fc1 = tf.nn.relu(tf.matmul(N_norm4_flat, W_fc1) + b_fc1)
        
        #Densely Connected Layer 2
        # W_fc2 = weight_variable([1024, 512])
        # b_fc2 = bias_variable([512])
        # 
        # h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

        #Dropout
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        
        ## Readout layer
        W_fc2 = weight_variable([128, 2], "W_fc2")
        b_fc2 = bias_variable([2], "b_fc2")
        
        y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)




        ## Train and Evaluate the Model
        # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
        cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0)))
    
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        sess.run(tf.initialize_all_variables())
    
        saver = tf.train.Saver()
        prediction = tf.argmax(y_conv, 1)
        probabilities = y_conv
        feat_h_fc1 = h_fc1
    
        ### START TRAINING
        N_sample = array_trainning_set.shape[0]
        print "  Total N_sample:", N_sample
        count = 0
        for i in range(N_sample / 50):  ##### 150 DEFAULT
            count = count % N_sample
            if (count + 50 > N_sample):
                count_end = N_sample 
            else:
                count_end = count + 50
            train_set = array_trainning_set[count: count_end]
            train_class = array_trainning_class[count: count_end]
            count += 50
    
            if i%10 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: train_set, y_: train_class, keep_prob: 1.0})
                # print("step %d, training accuracy %g"%(i, train_accuracy))
                # print prediction.eval(feed_dict={x: train_set, keep_prob: 1.0})
                # print probabilities.eval(feed_dict={x: train_set, keep_prob: 1.0})
            train_step.run(feed_dict={x: train_set, y_: train_class, keep_prob: drop_out})
    
        ## save the variables
        save_path = saver.save(sess, CNN_model_path)
        print("Model saved in file: %s" % save_path)
    
        ###########################################################################################
        ### Test set 
        test_set = []
        test_class = []
        for imageIdx in test_image_set:
            txt_file = txt_files[imageIdx]
            m1 = re1.match(txt_file)
            folder = m1.groupdict()['folder']
            image = m1.groupdict()['image']
            
            for ASF_image in ASF_files:
                m2 = re.search(folder, ASF_image)
                m3 = re.search(image, ASF_image)
                if m2 != None and m3 != None:
                    break
             
            imASF = misc.imread(ASF_image)
            axis0 = imASF.shape[0]
            axis1 = imASF.shape[1]
            R0 = imResizeTable[axis1]
            imASFPadd = np.ndarray(shape=(axis0+R0*2,axis1+R0*2), dtype='uint8')
            imASFPadd[R0:axis0+R0, R0:axis1+R0] = imASF
    
            ## Keep all candidates
            candi_file = open(txt_file, 'r')
            lines = candi_file.readlines()
    
            ## Load candidates into training set
            centers = []
            for line in lines:
                words = line.split()
                x_center = int(words[0]) + R0
                y_center = int(words[1]) + R0
                centers.append([x_center - R0, y_center - R0])
                
                imCrop = imASFPadd[y_center-R0 : y_center+R0, x_center-R0:x_center+R0]
                if (axis1 != 1440 or axis1 != 1504):
                    imCrop = misc.imresize(imCrop,(size,size))

                test_set.append(imCrop)
    
                if words[-1] == '1':
                    test_class.append([0,1])
                    # out_file = TP_out_path + "/" + out_filename
                else:
                    test_class.append([1,0])
                    # out_file = FP_out_path + "/" + out_filename
                # misc.imsave(out_file, imCrop)
    
            array_test_set = np.reshape(test_set, (len(test_set), size*size))
            array_test_class = np.reshape(test_class, (len(test_class), 2))
            array_test_set = array_test_set / 255.0
            
            out_filename =output_path +"/"+folder+"_"+image+".txt"
            print "  ",out_filename
            f = open(out_filename, 'w')
            flag = True
            idxBegin = 0
            idxEnd = 0
            while flag:
                # print idxBegin,"/",len(array_test_set)
                if (idxBegin+3000) > len(array_test_set):
                    flag = False
                    idxEnd = len(array_test_set - 1) 
                else:
                    idxEnd = idxBegin + 3000
        
                FP_test_predict =  probabilities.eval(feed_dict={x: array_test_set[idxBegin:idxEnd], keep_prob: 1.0})
                toto = feat_h_fc1.eval(feed_dict={x: array_test_set[idxBegin:idxEnd], keep_prob: 1.0})

                for i in range(len(FP_test_predict)):
                    # imname = FP_files[IdxFPRand[NFP_train + i + idxBegin]]
                    reIdx = idxBegin + i
                    f.write(str(centers[reIdx][0]) + " " + str(centers[reIdx][1]) + " " + str(array_test_class[reIdx][1]) + " " + str(FP_test_predict[i][1]) + "\n")  

                    if array_test_class[reIdx][1] == 1:
                        print i
        
                idxBegin = idxEnd
    
            f.close()
            ipdb.set_trace()
            
        sess.close()
    

    ipdb.set_trace()
    
