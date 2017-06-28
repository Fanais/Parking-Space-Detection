import numpy as np
import cv2
from perpective_transform import four_point_transform
import keras
from keras.models import load_model

li = []
lis = []
cap = cv2.VideoCapture('vietdv.mp4')
model = load_model('my_model_1m.h5')

def draw(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        li.append([x, y])

cv2.namedWindow('image')
cv2.setMouseCallback('image',draw)

if (cap.isOpened()): 
    ret, frame = cap.read()
    while (1):
        cv2.imshow('image', frame)
        k = cv2.waitKey(1)
        if k == 32:
            lis.append(li)
            pts = np.array(li, np.int32)
            cv2.polylines(frame, [pts], True, (0,255,255))
            li = []
            print 'press space'
        if k == ord('q'):
            break

def predict(img):
    x_val = np.array([img])
    x_val = x_val.astype('float32')
    x_val /= 255
    y_val = model.predict(x_val)
    prediction = np.round(y_val)
    return 1 if prediction[0][1] else 0
    
while (cap.isOpened()):
    ret, frame = cap.read()
    for li in lis:
        pts = np.array(li, np.int32)
        warped = four_point_transform(frame, pts)
        cv2.imshow("Warped", warped)
        warped = cv2.resize(warped, (64, 48))
        col = predict(warped)
        if col == 0:
            cv2.polylines(frame, [pts], True, (0, 255, 0))
        else:
            cv2.polylines(frame, [pts], True, (0, 0, 255))
    cv2.imshow('image', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.release()
cv2.destroyAllWindows()