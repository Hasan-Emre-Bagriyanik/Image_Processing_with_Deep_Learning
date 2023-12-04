# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 09:21:11 2023
@author: Hasan Emre
"""

#%% import library
import cv2
import matplotlib.pyplot as plt

# Gerekli kütüphaneler içe aktarılıyor

# template matching: şablon eşleme 

# Ana görüntüyü "cat.jpg" dosyasından oku ve siyah-beyaz (grayscale) yap
img = cv2.imread("cat.jpg", 0)
print(img.shape)  # Görüntünün boyutunu yazdır

# Şablon resmini "cat_face.jpg" dosyasından oku ve siyah-beyaz yap
template = cv2.imread("cat_face.jpg", 0)
print(template.shape)  # Şablon resminin boyutunu yazdır

# Şablonun genişliği ve yüksekliği (w, h) alınır
w, h = template.shape

# Kullanılacak şablon eşleme yöntemlerinin listesi
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

# Şablon eşleme yöntemlerinin her biri için işlemleri gerçekleştir
for meth in methods:
    
    # Şu anki yöntemi seç
    method = eval(meth)
    
    # Şablon eşlemeyi gerçekleştir ve sonucu "res" değişkenine at
    res = cv2.matchTemplate(img, template, method)
    print(res.shape)  # Eşleme sonucunun boyutunu yazdır

    # Eşleme sonucunda en düşük ve en yüksek benzerlik değerleri ile konumlar alınır
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # Eğer şu anki yöntem SQDIFF ise, en düşük benzerlik konumu alınır, aksi takdirde en yüksek benzerlik konumu alınır
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    # Dikdörtgenin sağ alt köşesinin koordinatları hesaplanır
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # Bulunan eşleme bölgesi üzerine bir dikdörtgen çizilir
    cv2.rectangle(img, top_left, bottom_right, 255, 2)
    
    # Sonuçları görselleştir
    plt.figure()
    plt.subplot(121), plt.imshow(res, cmap="gray")
    plt.title("Eşleşen sonuç"), plt.axis("off")
    plt.subplot(122), plt.imshow(img, cmap="gray")
    plt.title("Test edilen sonuç"), plt.axis("off")
    
    # Şu anki yönteme göre başlık eklenir
    plt.suptitle(meth)
    
    # Sonuçlar ekranda gösterilir
    plt.show()
