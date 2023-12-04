# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 10:50:48 2023
@author: Hasan Emre
"""

#%% import library

# Gerekli kütüphaneleri içe aktar
import cv2
import numpy as np

# Resmi "img3.jpg" dosyasından yükle ve gri tonlamada göster
img = cv2.imread("img3.jpg", 0)
cv2.imshow("Resim", img)

# Kenar Tespiti (Canny Edge Detection)
# Eşik değerleri (threshold) 200 ve 255 kullanarak kenarları bul
edges = cv2.Canny(image=img, threshold1=200, threshold2=255)
cv2.imshow("Kenar Tespiti", edges)

# Yüz Tespiti
# Haar Cascade sınıflandırıcıyı kullanarak yüz tespiti yap
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face_rect = face_cascade.detectMultiScale(img)

# Tespit edilen yüzleri çerçeve içine al ve mavi renkle işaretle
for (x, y, w, h) in face_rect:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
cv2.imshow("Yüz Tespiti", img)

# İnsan Tespiti (HOG)
# HOG tanımlayıcısı kullanarak insan tespiti yap
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
(rects, weights) = hog.detectMultiScale(img, padding=(8, 8), scale=1.05)

# Tespit edilen insanları çerçeve içine al ve mavi renkle işaretle
for (x, y, w, h) in rects:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
cv2.imshow("İnsan Tespiti", img)

# Resmi göster
cv2.waitKey(0)
cv2.destroyAllWindows()
