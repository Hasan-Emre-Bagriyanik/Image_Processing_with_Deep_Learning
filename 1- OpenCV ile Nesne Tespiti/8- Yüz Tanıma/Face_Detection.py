# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 12:14:12 2023
@author: Hasan Emre
"""

#%% import library

# Gerekli kütüphaneleri içe aktar
import cv2
import matplotlib.pyplot as plt

# Resmi içe aktar (Einstein'ın fotoğrafı, siyah beyaz olarak)
einstein = cv2.imread("einstein.jpg", 0)
plt.figure(), plt.imshow(einstein, cmap="gray"), plt.axis("off")

# Yüz tespiti için bir sınıflandırıcı (Cascade Classifier) oluştur
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Einstein'ın yüzünü tespit et
face_rect = face_cascade.detectMultiScale(einstein)

# Tespit edilen yüzleri dikdörtgen ile çevrele
for (x, y, w, h) in face_rect:
    cv2.rectangle(einstein, (x, y), (x+w, y+h), (255, 255, 255), 10)

# Sonucu görselleştir
plt.figure(), plt.imshow(einstein, cmap="gray"), plt.axis("off")

#%% Barcelona

# Barcelona şehrinin fotoğrafını içe aktar (siyah beyaz olarak)
barcelona = cv2.imread("barcelona.jpg", 0)
plt.figure(), plt.imshow(barcelona, cmap="gray"), plt.axis("off")

# Barcelona fotoğrafındaki yüzleri tespit et (minNeighbors parametresi ile ayarlandı)
face_rect = face_cascade.detectMultiScale(barcelona, minNeighbors=7)

# Tespit edilen yüzleri dikdörtgen ile çevrele
for (x, y, w, h) in face_rect:
    cv2.rectangle(barcelona, (x, y), (x+w, y+h), (255, 255, 255), 10)

# Sonucu görselleştir
plt.figure(), plt.imshow(barcelona, cmap="gray"), plt.axis("off")

#%% Video

# Kamera erişimini başlat
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if ret:
        # Kameradan gelen görüntüde yüzleri tespit et
        face_rect = face_cascade.detectMultiScale(frame, minNeighbors=10)
        
        # Tespit edilen yüzleri dikdörtgen ile çevrele
        for (x, y, w, h) in face_rect:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 10)
        
        # Sonucu görüntüle
        cv2.imshow("Yüz Tespiti", frame)
        
    # Kullanıcı "q" tuşuna basana kadar devam et
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Kamerayı serbest bırak ve tüm pencereleri kapat
cap.release()
cv2.destroyAllWindows()


#%% 

"""
İlgili kütüphaneler içe aktarılır: OpenCV ve Matplotlib.

İlk olarak, "einstein.jpg" adlı bir resmi siyah beyaz olarak içe aktarır ve ekranda gösterir. Ardından yüz tespiti yapmak için bir yüz sınıflandırıcısı (Cascade Classifier) oluştururuz.

Yüz tespiti, face_cascade.detectMultiScale(einstein) ile gerçekleştirilir. Tespit edilen yüzlerin koordinatlarına ve boyutlarına (x, y, w, h) erişilir.

Her tespit edilen yüzün etrafına bir dikdörtgen çizilir ve sonuç görselleştirilir.

Benzer şekilde, "barcelona.jpg" adlı başka bir resim için yüz tespiti yapılır ve tespit edilen yüzler çizilen dikdörtgenlerle işaretlenir.

Video yakalama işlemi başlatılır. Kameradan gelen görüntüleri sürekli olarak okur ve yüz tespiti yapar. Tespit edilen yüzler etrafına dikdörtgenler çizilir ve sonuç görüntülenir. Kullanıcı "q" tuşuna basana kadar bu işlem devam eder.
"""
