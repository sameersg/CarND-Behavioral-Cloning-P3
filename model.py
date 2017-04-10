import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D



lines = []

with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
correction = 0.2

for line in lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        #print(filename)
        current_path = '../data/IMG/' + filename
        #print(current_path)
        #image = cv2.imread(current_path)
        image = cv2.imread(current_path)[:,:,::-1]  # Read image and convert from BGR to RGB
        images.append(image)

    measurement = float(line[3])
    measurements.append(measurement)
    measurements.append(measurement + correction)
    measurements.append(measurement - correction)
    #print (measurements)



augumented_images, augumented_measurements = [], []

for image, measuerement in zip(images, measurements):
    augumented_images.append(image)
    augumented_measurements.append(measuerement)
    augumented_images.append(cv2.flip(image,1))
    augumented_measurements.append(measuerement*-1.0)


X_train = np.array(augumented_images)
y_train = np.array(augumented_measurements)


model = Sequential()

model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70, 25), (1, 1))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3, verbose=1)


model.save('model.h5')
'''
# Output visualization
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model MSE loss')
plt.ylabel('MSE loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper right')
plt.show()
'''
