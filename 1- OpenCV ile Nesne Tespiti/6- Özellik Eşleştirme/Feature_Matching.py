# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 10:21:04 2023
@author: Hasan Emre
"""

#%% import library

# Gerekli kütüphaneler içe aktarılır
import cv2
import matplotlib.pyplot as plt

# Ana görüntüyü içe aktar
chos = cv2.imread("chocolates.jpg", 0)
plt.figure(), plt.imshow(chos, cmap="gray"), plt.axis("off")

# Aranacak olan görüntüyü içe aktar
cho = cv2.imread("nestle.jpg", 0)
plt.figure(), plt.imshow(cho, cmap="gray"), plt.axis("off")

# ORB tanımlayıcısı oluştur
orb = cv2.ORB_create()

# Anahtar noktaların tespiti
kp1, des1 = orb.detectAndCompute(cho, None)
kp2, des2 = orb.detectAndCompute(chos, None)

# BF Matcher (Brute-Force Matcher) oluştur
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

# Noktaları eşleştir
matches = bf.match(des1, des2)

# Mesafeye göre sırala
matches = sorted(matches, key=lambda x: x.distance)

# Eşleşen resimleri görselleştirme
plt.figure()
img_match = cv2.drawMatches(cho, kp1, chos, kp2, matches[:20], None, flags=2)
plt.imshow(img_match), plt.axis("off"), plt.title("ORB")

# SIFT (Scale-Invariant Feature Transform) tanımlayıcısı oluştur
sift = cv2.xfeatures2d.SIFT_create()

# BF Matcher
bf = cv2.BFMatcher()

# Anahtar noktaları SIFT ile tespit et
kp1, des1 = sift.detectAndCompute(cho, None)
kp2, des2 = sift.detectAndCompute(chos, None)

# Eşleşmeleri bul
matches = bf.knnMatch(des1, des2, k=2)

güzel_eslesme = []

for match1, match2 in matches:
    if match1.distance < 0.75 * match2.distance:
        güzel_eslesme.append([match1])

plt.figure()
sift_matches = cv2.drawMatchesKnn(cho, kp1, chos, kp2, güzel_eslesme, None, flags=2)
plt.imshow(sift_matches), plt.axis("off"), plt.title("SIFT")

#%% 
"""
Gerekli kütüphaneler içe aktarılır: OpenCV ve Matplotlib.

Ana görüntü ("chocolates.jpg") ve aranacak olan görüntü ("nestle.jpg") siyah beyaz olarak içe aktarılır ve ekranda gösterilir.

ORB (Oriented FAST and Rotated BRIEF) tanımlayıcısı oluşturulur.

İlk görüntü için anahtar noktalar (keypoints) ve tanımlayıcılar (descriptors) tespit edilir.

İkinci görüntü için anahtar noktalar ve tanımlayıcılar tespit edilir.

Brute-Force Matcher (BF Matcher) oluşturulur ve NORM_HAMMING mesafe ölçüsü kullanılır.

İki görüntü arasındaki eşleşmeler bulunur ve benzerlik mesafesine göre sıralanır.

İlk 20 eşleşme görselleştirilir ve "ORB" başlığı ile ekranda gösterilir.

Scale-Invariant Feature Transform (SIFT) tanımlayıcısı oluşturulur.

Brute-Force Matcher tekrar oluşturulur.

İlk görüntü için anahtar noktalar ve tanımlayıcılar tespit edilir.

İkinci görüntü için anahtar noktalar ve tanımlayıcılar tespit edilir.

Eşleşmeler bulunur, ancak bu kez bir eşik değeri kullanılarak "güzel" eşleşmeleri ayırır.

"güzel" eşleşmeler görselleştirilir ve "SIFT" başlığı ile ekranda gösterilir.
"""
