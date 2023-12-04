
import cv2
import numpy as np

def non_maxi_suppression(boxes, probs = None, overlabThreshold=0.3):
    
    if len(boxes) == 0: 
        return []
    
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
        
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    
    # alanı bulalım 
    area = (x2 - x1 + 1)*(y2 - y1 + 1)
    
    idxs = y2
    
    # olasılık değerleri
    if probs is not None:
        idxs = probs
        
    # indeksi sırala
    idxs = np.argsort(idxs)
    
    pick = []  # seçilen kutular
    
    while len(idxs) > 0:
        
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        
        # en büyük ve ne küçük x ve y 
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # w,h bul
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        # overlap
        overlap = (w*h)/area[idxs[:last]]  
        
        idxs = np.delete(idxs,np.concatenate(([last], np.where(overlap > overlabThreshold))))
        
    return boxes[idxs].astype("int")


































