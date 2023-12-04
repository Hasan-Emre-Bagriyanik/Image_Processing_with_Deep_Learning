# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 10:33:08 2023
@author: Hasan Emre
"""

#%% import library

# Gerekli kütüphaneleri içe aktar
import os
import cv2

# Mevcut dizindeki dosyaları listele
files = os.listdir()
img_path_list = []

# .jpg uzantılı dosyaları bul ve listeye ekle
for f in files:
    if f.endswith(".jpg"):
        img_path_list.append(f)
        
# Bulunan .jpg dosyalarının listesini görüntüle
print(img_path_list)

# HOG (Histogram of Oriented Gradients) tanımlayıcısı oluştur
hog = cv2.HOGDescriptor()

# HOG tanımlayıcısına bir SVM (Support Vector Machine) modeli ekleyin (insan tespiti için önceden eğitilmiş bir model)
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Dosya listesindeki her bir resim üzerinde dön
for imgPath in img_path_list:
    print(imgPath)

    # Resmi oku
    image = cv2.imread(imgPath)
    
    # HOG ile insan tespiti yap
    (rects, weights) = hog.detectMultiScale(image, padding=(8, 8), scale=1.05)
    
    # Tespit edilen insanları çerçeve içine al ve yeşil renkle işaretle
    for (x, y, w, h) in rects:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Sonucu göster
    cv2.imshow("Yaya: ", image)
    
    # Kullanıcı "q" tuşuna basarsa bir sonraki resme geç
    if cv2.waitKey(0) & 0xFF == ord("q"):
        continue
