# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 09:18:29 2019

@author: user
"""

import numpy as np
import cv2
import sys
'''
if len(sys.argv) != 2:
    print('Input video name is missing')
    exit()
'''

print('Select 3 tracking targets') 

cv2.namedWindow("tracking")
camera = cv2.VideoCapture(0)
tracker = cv2.MultiTracker_create()
init_once = False

ok, image=camera.read()
if not ok:
    print('Failed to read video')
    exit()

bbox1 = cv2.selectROI('tracking', image)
p1 = (int(bbox1[0]), int(bbox1[1]))
p2 = (int(bbox1[0] + bbox1[2]), int(bbox1[1] + bbox1[3]))
cv2.rectangle(image, p1, p2, (255,0,0), 2, 1)
bbox2 = cv2.selectROI('tracking', image)


p1 = (int(bbox2[0]), int(bbox2[1]))
p2 = (int(bbox2[0] + bbox2[2]), int(bbox2[1] + bbox2[3]))
cv2.rectangle(image, p1, p2, (255,0,0), 2, 1)
bbox3 = cv2.selectROI('tracking', image)

p1 = (int(bbox3[0]), int(bbox3[1]))
p2 = (int(bbox3[0] + bbox3[2]), int(bbox3[1] + bbox3[3]))
cv2.rectangle(image, p1, p2, (255,0,0), 2, 1)

while camera.isOpened():
    ok, image=camera.read()
    if not ok:
        print('no image to read')
        break

    if not init_once:
        ok = tracker.add(cv2.TrackerMIL_create(), image, bbox1)
        ok = tracker.add(cv2.TrackerMIL_create(), image, bbox2)
        ok = tracker.add(cv2.TrackerMIL_create(), image, bbox3)
        init_once = True

    ok, boxes = tracker.update(image)
    print(ok, boxes)

    for newbox in boxes:
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv2.rectangle(image, p1, p2, (200,0,0))

    cv2.imshow('tracking', image)
    k = cv2.waitKey(1)
    if k == 27 : break # esc pressed
