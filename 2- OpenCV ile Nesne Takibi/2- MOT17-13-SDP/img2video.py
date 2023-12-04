# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 11:06:56 2023

@author: Hasan Emre
"""

#%% 

"""
- veri setini inceleme
- veri seti indirme 
- resimi videoya çevirme 
- eda -> gt
"""
# burada 750 tane resimi video haline getirdik
# Gerekli kütüphaneleri içe aktarın
import cv2
import os
from os.path import isfile, join
import matplotlib.pyplot as plt

# Resim dosyalarının bulunduğu dizin yolu
pathIn = r"img1"

# Oluşturulan video dosyasının yolu ve adı
pathOut = "MOT17-13-SDP.mp4"

# Dizindeki tüm dosyaları listeye alın
files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

# Video çerçevelerinin hızı (fps) ve boyutu (size) ayarları
fps = 25
size = (1920, 1080)

# Video çıkışını oluşturun
out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*"MP4V"), fps, size, isColor=True)

# Dizi içindeki her resmi işleyin ve videoya yazın
for i in files:
    print(i)

    # Resim dosyasının tam yolu
    fileName = pathIn + "\\" + i

    # Resmi okuyun
    img = cv2.imread(fileName)

    # Videoya çerçeve ekleyin
    out.write(img)

# Video çıkışını kapatın
out.release()

