# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 15:00:44 2023
@author: Hasan Emre
"""

#%% import library

# Gerekli kütüphaneleri içe aktar
import cv2
import os

# Mevcut dizindeki dosyaları listele
files = os.listdir()
print(files)

# Resim dosyalarının bulunduğu yolu saklayacak bir liste oluştur
img_path_list = []

# Dosyalar arasında dön ve .jpeg uzantılı olanları bulup listeye ekle
for f in files:
    if f.endswith(".jpeg"):
        img_path_list.append(f)

# Bulunan .jpeg dosyalarının listesini görüntüle
print(img_path_list)


for j in img_path_list:
    print(j)
    # Resmi oku
    image = cv2.imread(j)
    
    # Resmi gri tonlamaya dönüştür
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Kedilerin yüzlerini tespit etmek için bir sınıflandırıcı (Cascade Classifier) oluştur
    detector = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")
    
    # Yüz tespiti yap
    rects = detector.detectMultiScale(gray, scaleFactor=1.045, minNeighbors=1)
    
    # Tespit edilen yüzleri çevrele ve numaralandır
    for (i, (x, y, w, h)) in enumerate(rects):
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 255), 2)
        cv2.putText(image, "Kedi {}".format(i+1), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        
    # Sonucu göster
    cv2.imshow(j, image)
    
    # "q" tuşuna basıldığında sonraki resme geç
    if cv2.waitKey(0) & 0xFF == ord("q"):
        continue
