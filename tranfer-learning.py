from __future__ import print_function
import os
from scipy import misc
import random
import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.constraints import maxnorm
from keras import applications

file_dirs = []
for root, dirs, files in os.walk("/data/vietdv/PKLot/PKLotSegmented/UFPR05", topdown=False):
    for name in files:
        file_dirs.append(os.path.join(root, name))


random.shuffle(file_dirs)
num_samples = len(file_dirs)
X = []
Y = []

for i in range(0, num_samples):
    f = misc.imread(file_dirs[i])
    f = misc.imresize(f, (48, 64))
    X.append(f)
    folders = file_dirs[i].split('/')
    label = 0 if folders[len(folders) - 2] == 'Occupied' else 1
    Y.append(label)

X = np.array(X)
Y = np.array(Y)
num_validate = 10000
num_test = 0
num_val_test = num_validate + num_test
(x_train, y_train) = (X[:num_samples - num_val_test], Y[:num_samples - num_val_test])
(x_test, y_test) = (X[num_samples - num_val_test:num_samples - num_test],
                    Y[num_samples - num_val_test:num_samples - num_test])

batch_size = 256
num_classes = 2
epochs = 10
img_rows, img_cols = 48, 64

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float64')
x_test = x_test.astype('float64')

if keras.backend.image_data_format() == 'channels_first':
    input_shape = (3, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 3)


model = applications.VGG19(weights="imagenet", include_top=False, input_shape=input_shape)
model.summary()
for layer in model.layers:
    layer.trainable = False

# Adding custom layer
x = model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(2, activation="softmax")(x)

final_model = Model(inputs=model.input, outputs=predictions)
final_model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=180,
    channel_shift_range=0.3
)
test_datagen = ImageDataGenerator(
    rescale=1. / 255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=180,
    channel_shift_range=0.3
)
print(x_test.shape)
final_model.fit_generator(train_datagen.flow(x_train, y_train,
                                             batch_size=batch_size),
                          steps_per_epoch=x_train.shape[0] // batch_size,
                          epochs=epochs,
                          verbose=2,
                          validation_data=test_datagen.flow(x_test, y_test),
                          validation_steps=num_validate)

final_model.save('PKLot.h5')
