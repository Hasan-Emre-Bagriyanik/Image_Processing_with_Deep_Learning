# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 19:58:28 2023

@author: Hasan Emre
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

OPENCV_OBJECT_TRACKERS = {"csrt" : cv2.TrackerCSRT_create,
                          "kcf" : cv2.TrackerKCF_create,
                          "mil" : cv2.TrackerMIL_create }

tracker_name = "csrt"

trackers = OPENCV_OBJECT_TRACKERS[tracker_name]


video_path = "MOT17-04-SDP.mp4"
cap = cv2.VideoCapture(video_path)

fps = 30
f = 0

while True:
    
    ret, frame  =cap.read()
    (H,W) = frame.shape[:2]
    frame = cv2.resize(frame, dsize=(960, 540))
    
    (success, boxes) = trackers.update(frame)
    
    info = [("Tracker", tracker_name),
            ("Tracker","Yes" if success else "No")]

    string_text = ""
    
    for (i, (k,v)) in enumerate(info):
        text = "{}: {}".format(k,v)
        string_text = string_text + text + " "
        
    cv2.putText(frame, string_text, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
   
    for box in boxes:
        (x,y,w,h) = [int(v) for v in box]
        cv2.rectangle(frame, (x,y), (x+w, y+h),(0,0,255),2)
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    
    if key == ord("t"): 
        
        box = cv2.selectROI("Frame", frame, fromCenter=False)
        tracker = OPENCV_OBJECT_TRACKERS[tracker_name]()
        trackers.add(tracker, frame, box)
        
    
    elif key == ord("q"): break

    f += 1
    
cap.release()
cv2.destroyAllWindows()







