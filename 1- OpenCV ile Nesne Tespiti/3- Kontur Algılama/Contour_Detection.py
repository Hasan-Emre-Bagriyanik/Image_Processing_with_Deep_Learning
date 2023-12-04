# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 11:33:06 2023
@author: Hasan Emre
"""

#%% Gerekli kütüphaneleri içe aktarıyoruz
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Resmi siyah-beyaz olarak içe aktarıyoruz
img = cv2.imread("contour.jpg", 0)

# Resmi görselleştiriyoruz ve eksenleri kapatıyoruz
plt.figure(), plt.imshow(img, cmap="gray"), plt.axis("off")

# Konturları ve hiyerarşiyi buluyoruz
contours, hierarch = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# Dış ve iç konturlar için boş görüntüler oluşturuyoruz
external_contour = np.zeros(img.shape)
internal_contour = np.zeros(img.shape)

# Konturları işleyerek dış ve iç konturları ayırıyoruz
for i in range(len(contours)):
    if hierarch[0][i][3] == -1:  # Dış kontur
        cv2.drawContours(external_contour, contours, i, 255, -1)
    else:  # İç kontur
        cv2.drawContours(internal_contour, contours, i, 255, -1)

# Dış konturları görselleştiriyoruz
plt.figure(), plt.imshow(external_contour, cmap="gray"), plt.axis("off")

# İç konturları görselleştiriyoruz
plt.figure(), plt.imshow(internal_contour, cmap="gray"), plt.axis("off")
