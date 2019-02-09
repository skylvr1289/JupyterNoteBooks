import sys
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense, Activation
from keras import callbacks
from keras import optimizers
import time

start_time = time.time()

DEVEL = False
argvs = sys.argv
argc = len(argvs)

if argc > 1 and (argvs[1] == "--development" or argvs[1] == "-d"):
    DEVEL = True

if DEVEL:
    epochs = 2
else:
    epochs = 20

train_data_path = 'data/train'
validation_data_path = 'data/test'

"""
Parameters Intialization
"""
# Changed from 150, 150
# img_width, img_height = 4000, 4000
img_width, img_height = 300, 300
batch_size = 32
samples_per_epoch = 1000
validation_steps = 300
nb_filters1 = 32
nb_filters2 = 64
conv1_size = 3
conv2_size = 2
pool_size = 2
# classes_num = 3
classes_num = 2
lr = 0.0004

"""
Model Construction
"""

model = Sequential()
model.add(Convolution2D(nb_filters1, conv1_size, conv1_size, border_mode="same", input_shape=(img_width, img_height, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Convolution2D(nb_filters2, conv2_size, conv2_size, border_mode="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size), dim_ordering='th'))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(classes_num, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=lr),
              metrics=['accuracy'])

"""
Image Preprocessing
"""
img_train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

img_test_datagen = ImageDataGenerator(rescale=1. / 255)

"""
Model Training
"""
train_generator = img_train_datagen.flow_from_directory(
    train_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')
"""
Model Validation
"""

validation_generator = img_test_datagen.flow_from_directory(
    validation_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

"""
Tensorboard log
"""
tb_log_dir = './tf-log/'
tb_cb = callbacks.TensorBoard(log_dir=tb_log_dir, histogram_freq=0)
cbks = [tb_cb]

model.fit_generator(
    train_generator,
    samples_per_epoch=samples_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=cbks,
    validation_steps=validation_steps)

"""
Save Model and Weights
"""
target_model_dir = './models/'
if not os.path.exists(target_model_dir):
    os.mkdir(target_model_dir)
model.save('./models/model.h5')
model.save_weights('./models/weights.h5')

# Calculate execution time
end_time = time.time()
duration = end_time-start_time

if duration < 60:
    print("Execution Time:", duration, "seconds")
elif duration > 60 and duration < 3600:
    duration = duration / 60
    print("Execution Time:", duration, "minutes")
else:
    duration = duration / (60*60)
    print("Execution Time:", duration, "hours")
