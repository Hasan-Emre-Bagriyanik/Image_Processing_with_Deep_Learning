# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 11:31:33 2023

@author: Hasan Emre
"""


#%%  import library
import cv2
import matplotlib.pyplot as plt
import numpy as np

#%% resmi içe aktarma 

img = cv2.imread("london.jpg", 0)
plt.figure(), plt.imshow(img, cmap = "gray"), plt.axis("off")


#%% Canny kenar algılama

edges = cv2.Canny(image = img, threshold1 = 0, threshold2 = 255)
plt.figure(), plt.imshow(edges, cmap = "gray"), plt.axis("off")


#%% Median değeri hesaplama
med_val = np.median(img)
print(med_val)

#%% Alt ve üst eşik değerlerini hesaplama
low = int(max(0,(1 - 0.33)*med_val))
high = int(min(255, (1 + 0.33)*med_val))

print(low)
print(high)

edges = cv2.Canny(image = img, threshold1 = low, threshold2 = high)
plt.figure(), plt.imshow(edges, cmap = "gray"), plt.axis("off")


#%% Blur işlemi

blurred_img = cv2.blur(img, ksize = (4,4))
plt.figure(), plt.imshow(blurred_img, cmap = "gray"), plt.axis("off")

# Blur sonrası median değeri hesaplama
med_val = np.median(blurred_img)
print(med_val)

# Blur sonrası alt ve üst eşik değerlerini hesaplama
low = int(max(0,(1 - 0.33)*med_val))
high = int(min(255, (1 + 0.33)*med_val))
print(low)
print(high)

# Blur sonrası Canny kenar algılama
edges = cv2.Canny(image = blurred_img, threshold1 = low, threshold2 = high)
plt.figure(), plt.imshow(edges, cmap = "gray"), plt.axis("off")

