import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
import time

start_time = time.time()

# Path Defintion
model_path = './models/model.h5'
model_weights_path = './models/weights.h5'
test_path = 'data/alien_test'

# pre-trained models Loading Block
model = load_model(model_path)
model.load_weights(model_weights_path)

# image parameters defintion
# img_width, img_height = 4000, 4000
img_width, img_height = 300, 300
# Prediction Function


def predict(file):
    x = load_img(file, target_size=(img_width, img_height))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    array = model.predict(x)
    result = array[0]
    # print(result)
    answer = np.argmax(result)
    if answer == 1:
        print("Predicted: True")
    elif answer == 0:
        print("Predicted: False")
    return answer


# Walk the directory for every image
for i, ret in enumerate(os.walk(test_path)):
    for i, filename in enumerate(ret[2]):
        if filename.startswith("."):
            continue

        print(ret[0] + '/' + filename)
        result = predict(ret[0] + '/' + filename)
        print(" ")

# Calculate execution time
end_time = time.time()
duration = end_time-start_time

if duration < 60:
    print("Execution Time:", duration, "seconds")
elif duration > 60 and duration < 3600:
    duration = duration/60
    print("Execution Time:", duration, "minutes")
else:
    duration = duration/(60*60)
    print("Execution Time:", duration, "hours")
