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

flags = tf.app.flags
FLAGS = flags.FLAGS

#DEFINE FLAGS VARIABLES#
flags.DEFINE_float('steering_adjustment', 0.27, "Adjustment angle.")

cwd = os.getcwd()
## IMPORT COLUMNS FROM driving_log.csv INTO LISTS ##
colnames = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
data = pandas.read_csv(cwd+'/data/driving_log.csv', skiprows=[0], names=colnames)
center = data.center.tolist()
center_recover = data.center.tolist()
left = data.left.tolist()
right = data.right.tolist()
steering = data.steering.tolist()
steering_recover = data.steering.tolist()

## SPLIT TRAIN AND VALID ##
#  Shuffle center and steering. Use 10% of central images and steering angles for validation.
center, steering = shuffle(center, steering)
center, X_valid, steering, y_valid = train_test_split(center, steering, test_size = 0.10, random_state = 100)

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

# Filter angle less than -0.15 and add right camera images into driving left list, minus an adjustment angle #
for i in indice_L:
  if steering_recover[i] < -0.15:
    d_left.append(right[i])
    a_left.append(steering_recover[i] - FLAGS.steering_adjustment)

# Filter angle more than 0.15 and add left camera images into driving right list, add an adjustment angle #
for i in indice_R:
  if steering_recover[i] > 0.15:
    d_right.append(left[i])
    a_right.append(steering_recover[i] + FLAGS.steering_adjustment)

## COMBINE TRAINING IMAGE NAMES AND ANGLES INTO X_train and y_train ##
# X_train = d_straight + d_left + d_right
# y_train = np.float32(a_straight + a_left + a_right)
images_files_list=d_straight + d_left + d_right

images=[]
for filename in images_files_list:
        current_path=cwd+"/data/IMG/"+filename.split('/')[-1]
        if os.path.isfile(current_path):
            image=cv2.imread(current_path)
        else:
            print("no file for "+current_path)
        #image = cv2.resize(image, (120, 120))
        images.append(image)

measurements=a_straight + a_left + a_right

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

augmented_images,augmented_measurements=[],[]

for image, measurement in zip(images,measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)

X_train=np.array(augmented_images)
y_train=np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Cropping2D,Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model=Sequential()
model.add(Lambda(lambda x:x/255.0-0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))


model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Dropout(0.5))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Dropout(0.5))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Dropout(0.5))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Dropout(0.5))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
model.summary()
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=5)

model.save('model_NVIDA_DataRecovering.h5')
exit()