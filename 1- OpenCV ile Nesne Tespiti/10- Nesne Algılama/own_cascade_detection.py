# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 16:13:22 2023
@author: Hasan Emre
"""

#%% import library

# Gerekli kütüphaneleri içe aktar
import cv2

# Tanımlamalar
objectName = "Fare"  # Nesnenin adı
frameWidth = 200  # Görüntü genişliği
frameHeight = 360  # Görüntü yüksekliği
color = (255, 0, 0)  # Çizgi rengi (RGB formatında)

# Kamera yakalama
cap = cv2.VideoCapture(0)  # Kamerayı aç

# Kamera ayarları (genişlik ve yükseklik)
cap.set(3, frameWidth)  # Görüntü genişliği
cap.set(4, frameHeight)  # Görüntü yüksekliği

# Boş bir işlev (trackbar'ın kullanılabilmesi için gerekli)
def empty(a):
    pass

# Trackbar oluşturma
cv2.namedWindow("Sonuc")  # Sonuç penceresi oluştur
cv2.resizeWindow("Sonuc", frameWidth, frameHeight + 100)  # Pencere boyutunu ayarla
cv2.createTrackbar("Scale", "Sonuc", 400, 1000, empty)   # Ölçeklendirme trackbar'ı oluştur
cv2.createTrackbar("Neighbor", "Sonuc", 4, 50, empty)  # Komşuluk trackbar'ı oluştur

# Cascade Sınıflandırıcı
cascade = cv2.CascadeClassifier("cascade.xml")  # Nesne tespiti için bir sınıflandırıcı yükle

while True:
    # Görüntüyü oku
    success, img = cap.read()
    
    if success:
        # Görüntüyü BGR'den Griye dönüştür
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Tespit parametreleri
        scaleVal = 1 + (cv2.getTrackbarPos("Scale", "Sonuc") / 1000)  # Ölçekleme faktörü
        neighbor = cv2.getTrackbarPos("Neighbor", "Sonuc")  # Komşuluk parametresi
        
        # Nesne tespiti
        rects = cascade.detectMultiScale(gray, scaleVal, neighbor)
    
        for (x, y, w, h) in rects:
            # Tespit edilen nesnenin etrafına bir dikdörtgen çiz
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)
            # Nesnenin adını ve koordinatlarını ekle
            cv2.putText(img, objectName, (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
        
        # Sonucu göster
        cv2.imshow("Sonuc", img)
        
    # Kullanıcı "q" tuşuna basarsa döngüyü kır
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
