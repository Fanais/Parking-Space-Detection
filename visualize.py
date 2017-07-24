import keras
from keras.models import load_model
import numpy as np
import cv2

model = load_model('PKLot_V.h5')
a = model.get_weights()

for i in range(10):
    b = a[0][:, :, :, i]
    b = 255 * (b - np.min(b)) / (np.max(b) - np.min(b))
    b = np.rint(b)
    im = cv2.resize(b, (100, 100))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = np.array(im, dtype=np.uint8)
    cv2.imshow(str(i), im)
    cv2.waitKey(-1)
