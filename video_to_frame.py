import cv2
import numpy as np
import os
vid = cv2.VideoCapture('http://192.168.43.192:8080/video')
if not os.path.exists('images'):
    os.makedirs('images')
index = 0
c=1
while(True):
    ret, frame = vid.read()
    if not ret: 
        break
    if c%10==0:
        name = './images/frame' + str(index) + '.jpg'
        print ('Creating...' + name)
        cv2.imwrite(name, frame)
        cv2.waitKey(1)
        index += 1
    c+=1
