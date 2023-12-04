import cv2 
import random 

# Görüntüyü oku ve boyutunu (600, 600) olarak ayarla
image = cv2.imread("pyramid.jpg")
image = cv2.resize(image, dsize=(600, 600))

# Görüntüyü ekranda göster
cv2.imshow("image", image)

# Seçici arama algoritmasını başlat
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

# Algoritmanın çalışacağı temel görüntüyü belirle
ss.setBaseImage(image)

# Seçici arama algoritmasını kaliteye göre ayarla
ss.switchToSelectiveSearchQuality()

# Başlangıç mesajını yazdır
print("Başlangıç")

# Algoritma ile bölgeleri işle ve dikdörtgenleri döndür
rects = ss.process()

# Orijinal görüntüyü kopyala
output = image.copy()

# İlk 50 dikdörtgen için rastgele renklerde dikdörtgenler oluştur ve çiz
for (x, y, w, h) in rects[:50]:
    # Rastgele bir renk oluştur (0-255 arası üç farklı bileşen ile)
    color = [random.randint(0, 255) for j in range(0, 3)]
    # Oluşturulan dikdörtgeni orijinal görüntü üzerine çiz
    cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

# Dikdörtgenler çizildikten sonra görüntüyü ekranda göster
cv2.imshow("output", output)







