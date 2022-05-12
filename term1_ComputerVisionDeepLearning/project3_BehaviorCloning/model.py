import csv
import cv2
import numpy as np
import os
import pandas
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import math
import random
import tensorflow as tf
from keras.regularizers import l2
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.layers.advanced_activations import ELU
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, ELU

flags = tf.app.flags
FLAGS = flags.FLAGS

#DEFINE FLAGS VARIABLES#
using_generator=True

flags.DEFINE_float('steering_adjustment', 0.27, "Adjustment angle.")
flags.DEFINE_integer('epochs', 4, "The number of epochs.")

cwd = os.getcwd()
## IMPORT COLUMNS FROM driving_log.csv INTO LISTS ##
colnames = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']


data_folder_info=[('./data','/'),('./MyData',"\\"),('./Data_reverse',"\\")]

images_files_list=[]

measurements=[]

def dir_convert(abdir,subdir,split_info):
    try:
        dir_final=subdir+'/IMG/'+abdir.split(split_info)[-1]
    except:
        print("subdir=",subdir)
        print("abdir=",abdir)
        print("split_info=",split_info)
    return dir_final
def dir_conv_all(center_orig,subdir,split_info):
    dir_final_all=[]
    for i,center_orig_i in enumerate(center_orig):
        try:
            center_conv=dir_convert(center_orig_i,subdir,split_info)
        except:
            print(i)
            print(center_orig_i)
        dir_final_all+=[center_conv]
    return dir_final_all

center=[]
left=[]
right=[]
steering=[]
speed=[]
for data_foler in data_folder_info:
    subdir=data_foler[0]
    split_info=data_foler[1]
    print("split_info=",split_info)
    data = pandas.read_csv(subdir+'/driving_log.csv', skiprows=[0], names=colnames)

    center_orig= data.center.tolist()
    dir_final_all=dir_conv_all(center_orig,subdir,split_info)
    center+=dir_final_all
    # center_recover = data.center.tolist()
    left_orig = data.left.tolist()
    dir_final_all=dir_conv_all(left_orig,subdir,split_info)
    left+=dir_final_all

    right_orig = data.right.tolist()
    dir_final_all=dir_conv_all(right_orig,subdir,split_info)
    right+=dir_final_all

    steering+= data.steering.tolist()
    # steering_recover = data.steering.tolist()
    speed+=data.speed.tolist()

center_recover = center
steering_recover=steering


for i,steering_i in enumerate(steering):
    # skip it if ~0 speed - not representative of driving behavior
    if float(speed[i]) < 0.1:
        continue
    # get center image path and angle
    images_files_list.append(center[i])
    measurements.append(float(steering_i))
    # get left image path and angle
    images_files_list.append(left[i])
    measurements.append(float(steering_i) + 0.2)
    # get left image path and angle
    images_files_list.append(right[i])
    measurements.append(float(steering_i) - 0.2)


## FILTER STRAIGHT, LEFT, RIGHT TURNS ##
#  (d_### is list of images name, a_### is list of angles going with list)
d_straight, d_left, d_right = [], [], []
a_straight, a_left, a_right = [], [], []
for i in steering:
  #Positive angle is turning from Left -> Right. Negative is turning from Right -> Left#
  index = steering.index(i)
  if i > 0.15:
    d_right.append(center[index])
    a_right.append(i)
  if i < -0.15:
    d_left.append(center[index])
    a_left.append(i)
  else:
    d_straight.append(center[index])
    a_straight.append(i)

## ADD RECOVERY ##
#  Find the amount of sample differences between driving straight & driving left, driving straight & driving right #
ds_size, dl_size, dr_size = len(d_straight), len(d_left), len(d_right)
main_size = math.ceil(len(center_recover))
l_xtra = ds_size - dl_size
r_xtra = ds_size - dr_size
# Generate random list of indices for left and right recovery images
indice_L = random.sample(range(main_size), l_xtra)
indice_R = random.sample(range(main_size), r_xtra)

for i in indice_L:
  if steering_recover[i] < -0.15:
    d_left.append(right[i])
    a_left.append(steering_recover[i] - FLAGS.steering_adjustment)

for i in indice_R:
  if steering_recover[i] > 0.15:
    d_right.append(left[i])
    a_right.append(steering_recover[i] + FLAGS.steering_adjustment)

## COMBINE TRAINING IMAGE NAMES AND ANGLES INTO X_train and y_train ##

images_files_list=d_straight+ d_left + d_right

measurements=a_straight + a_left + a_right

# print("number of straight driving=",len(a_straight))
# print("number of left turn=",len(a_left))
# print("number of right turn=",len(a_right))

def histogram(angles,num_bins):
    avg_samples_per_bin = len(angles)/num_bins
    hist, bins = np.histogram(angles, num_bins)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    # plt.bar(center, hist, align='center', width=width)
    # plt.plot((np.min(angles), np.max(angles)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
    # plt.show()
    return avg_samples_per_bin,hist,bins

num_bins = 23
avg_samples_per_bin,hist,bins=histogram(measurements,num_bins)

def balancing(angles,avg_samples_per_bin,hist,bins):
    keep_probs = []
    target = avg_samples_per_bin * .5
    for i in range(num_bins):
        if hist[i] < target:
            keep_probs.append(1.)
        else:
            keep_probs.append(1./(hist[i]/target))
    remove_list = []
    for i in range(len(angles)):
        for j in range(num_bins):
            if angles[i] > bins[j] and angles[i] <= bins[j+1]:
                # delete from X and y with probability 1 - keep_probs[j]
                if np.random.rand() > keep_probs[j]:
                    remove_list.append(i)
    # image_paths = np.delete(image_paths, remove_list, axis=0)
    # angles = np.delete(angles, remove_list)
    return remove_list

def del_list(mylist,indexes):
    indexes = sorted(indexes, reverse=True)
    for i in indexes:
        mylist.pop(i)
    return mylist

remove_list=balancing(measurements,avg_samples_per_bin,hist,bins)
measurements=del_list(measurements,remove_list)
images_files_list=del_list(images_files_list,remove_list)

def plotting_hist(angles,num_bins):
    hist, bins = np.histogram(angles, num_bins)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.plot((np.min(angles), np.max(angles)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
    plt.show()

# plotting_hist(measurements,num_bins)



## SPLIT TRAIN AND VALID ##
#  Shuffle center and steering. Use 10% of central images and steering angles for validation.
images_files_list, measurements = shuffle(images_files_list, measurements)
images_files_list, X_valid, measurements, y_valid = train_test_split(images_files_list, measurements, test_size = 0.10, random_state = 100)


import cv2
import numpy as np
import sklearn

def random_brightness(image):
    #Convert 2 HSV colorspace from RGB colorspace
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    #Generate new random brightness
    rand = random.uniform(0.3,1.0)
    hsv[:,:,2] = rand*hsv[:,:,2]
    #Convert back to RGB colorspace
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return new_img

def generator(samples,samples_angles,Training=True,batch_size=128):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples,samples_angles=shuffle(samples,samples_angles)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            batch_samples_angles=samples_angles[offset:offset+batch_size]
            images = []
            angles = []
            for i in range(batch_size):
            # for i,batch_sample in enumerate(batch_samples):
                choice = int(np.random.choice(len(samples), 1))
                # name = './IMG/'+batch_sample[0].split('/')[-1]
                # name="./MyData/IMG/" + samples[choice].split("\\")[-1]
                name=samples[choice]
                # center_image = cv2.imread(name)
                if os.path.isfile(name):
                    center_image = random_brightness(cv2.imread(name))
                    center_image = cv2.resize(center_image[60:140, :], (64, 64))
                else:
                    print("no file for "+name)
                # center_angle = float(batch_sample[3])
                # center_angle =float(batch_samples_angles[i])
                center_angle =float(samples_angles[choice])
                if Training:
                    flip_coin=random.randint(0,1)
                    if flip_coin==1:
                        center_image=cv2.flip(center_image,1)
                        center_angle=center_angle*(-1.0)
                else:
                    pass
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(images_files_list,measurements,Training=True,batch_size=128)
validation_generator = generator(X_valid,y_valid,Training=False,batch_size=128)

if not using_generator:

    # lines=[]
    # with open(cwd+'/data/driving_log.csv') as csvfile:
    #     reader=csv.reader(csvfile)
    #     for line in reader:
    #         lines.append(line)
    #
    # images=[]
    # measurements=[]
    #
    # for i,line in enumerate(lines):
    #     if i>0:
    #         for j in range(3):
    #             source_path=line[j]
    #             filename=source_path.split('/')[-1]
    #             current_path=cwd+"/data/IMG/"+filename
    #             image=cv2.imread(current_path)
    #             #image = cv2.resize(image, (120, 120))
    #             images.append(image)
    #             measurement=float(line[3])
    #             measurements.append(measurement)

    images=[]
    for images_file in images_files_list:
        image=cv2.imread(images_file)
        # image = cv2.resize(image, (64, 64))
        images.append(image)

    augmented_images,augmented_measurements=[],[]

    for image, measurement in zip(images,measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        augmented_images.append(cv2.flip(image,1))
        augmented_measurements.append(measurement*-1.0)

    X_train=np.array(augmented_images)
    y_train=np.array(augmented_measurements)


    # from sklearn.model_selection import train_test_split
    # train_samples, validation_samples = train_test_split(samples, test_size=0.2)

from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Cropping2D,Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model=Sequential()
if using_generator:
    model.add(Lambda(lambda x:x/255.0-0.5,input_shape=(64,64,3)))
else:
    model.add(Lambda(lambda x:x/255.0-0.5,input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))

input_shape = (64, 64, 3)
model = Sequential()
model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=input_shape))
model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample=(2, 2), W_regularizer=l2(0.001)))
model.add(Activation('relu'))
model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample=(2, 2), W_regularizer=l2(0.001)))
model.add(Activation('relu'))
model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample=(2, 2), W_regularizer=l2(0.001)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(2, 2), W_regularizer=l2(0.001)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(2, 2), W_regularizer=l2(0.001)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(80, W_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(40, W_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(16, W_regularizer=l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(10, W_regularizer=l2(0.001)))
model.add(Dense(1, W_regularizer=l2(0.001)))
adam = Adam(lr=0.0001)
model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
model.summary()

model.fit_generator(train_generator, samples_per_epoch=len(images_files_list), validation_data=validation_generator,nb_val_samples=len(X_valid), nb_epoch=FLAGS.epochs)

model.save('model_NVIDA_DataBalanced_MoreData_ep4_0501.h5')
exit()