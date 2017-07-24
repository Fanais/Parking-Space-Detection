import numpy as np
import cv2
from perpective_transform import four_point_transform
import keras
from keras.models import load_model

li = []
lis = []
cap = cv2.VideoCapture('vietdv.mp4')
model = load_model('PKLot_V.h5')
# mean = np.array([166.95719972, 166.4651756, 161.39754563])
# std = np.array([51.85200214, 50.85509469, 49.63896689])
mean = np.array([166.843922, 166.31826174, 161.24557455])
std = np.array([51.83049202, 50.84292846, 49.62470782])
# mean = np.array([142.7435437, 141.75742675, 134.94970735])
# std = np.array([57.89781862, 56.52802884, 57.24394328])


def draw(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        li.append([x, y])


cv2.namedWindow('image')
cv2.setMouseCallback('image', draw)


def predict(img):
    x_val = np.array([img])
    x_val = x_val.astype('float32')
    x_val = (x_val - mean) / (std + 1e-9)
    x_val /= 255
    y_val = model.predict(x_val)
    prediction = np.round(y_val)
    return 1 if prediction[0][1] else 0


if (cap.isOpened()):
    ret, frame = cap.read()
    while (1):
        cv2.imshow('image', frame)
        k = cv2.waitKey(1)
        if k == 32:
            lis.append(li)
            pts = np.array(li, np.int32)
            cv2.polylines(frame, [pts], True, (0, 255, 255))
            li = []
            print('press space')
        if k == ord('q'):
            break


u = []
while (cap.isOpened()):
    ret, frame = cap.read()
    for li in lis:
        pts = np.array(li, np.int32)
        warped = four_point_transform(frame, pts)
        cv2.imshow("Warped", warped)
        warped = cv2.resize(warped, (64, 48))
        col = predict(warped)
        if col == 1:
            cv2.polylines(frame, [pts], True, (0, 255, 0))
            # print warped
            # print u
            # break
        else:
            cv2.polylines(frame, [pts], True, (0, 0, 255))
    cv2.imshow('image', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.release()
cv2.destroyAllWindows()
