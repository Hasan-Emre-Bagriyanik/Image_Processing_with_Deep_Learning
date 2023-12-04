# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 11:33:06 2023

@author: Hasan Emre
"""

#%% import library
import cv2
import matplotlib.pyplot as plt
import numpy as np

#%% resmi içe aktar

img = cv2.imread("sudoku.jpg", 0)
img = np.float32(img)
print(img.shape)
plt.figure(), plt.imshow(img, cmap = "gray"), plt.axis("off")


#%% harris corner detection
# blocksize = komşuluk boyutu, ksize = kutucuğun boyutu
dst = cv2.cornerHarris(img, blockSize = 2, ksize = 3, k = 0.04)
plt.figure(), plt.imshow(dst, cmap = "gray"), plt.axis("off")

# genişlettik ve rengini 1 e eşitleyerek beyaz yaptık
dst = cv2.dilate(dst, None)
img[dst>0.2*dst.max()] = 1
plt.figure(), plt.imshow(dst, cmap = "gray"), plt.axis("off")


#%%  shi tomasi detection

img = cv2.imread("sudoku.jpg", 0)
img = np.float32(img)

corners = cv2.goodFeaturesToTrack(img, 120, 0.01, 10)
# "goodFeaturesToTrack" işlevi ile resimde köşeleri tespit ediyoruz.
# İlk parametre resim, ikinci parametre maksimum köşe sayısı,
# üçüncü parametre ise köşelerin kalitesi için eşik değeri,
# dördüncü parametre ise minimum mesafe olarak belirtilir.

# Tespit edilen köşeleri tam sayı (integer) veri tipine dönüştürüyoruz.
corners = np.int64(corners)


for i in corners:
    x,y = i.ravel() # düzleştiriyoruz
    cv2.circle(img, (x,y), 3, (125,125,125), cv2.FILLED)
    # Her köşenin etrafına bir daire çiziyoruz. Dairenin merkezi (x, y),
    # yarıçapı 3 ve rengi (125, 125, 125) olarak belirtilmiştir.
    
plt.imshow(img)
plt.axis("off")
