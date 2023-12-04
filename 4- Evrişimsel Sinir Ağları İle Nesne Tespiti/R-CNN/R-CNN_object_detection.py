import cv2
import pickle
import numpy as np
import random
from tensorflow.keras.preprocessing.image import img_to_array

# Resmi oku ve göster
image = cv2.imread("mnist.png")
cv2.imshow("Image", image)

# Seçici arama algoritmasını başlat
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
# Algoritmanın çalışacağı temel görüntüyü belirle
ss.setBaseImage(image)
# Seçici arama algoritmasını kaliteye göre ayarla
ss.switchToSelectiveSearchQuality()
# Başlangıç mesajını yazdır
print("Seçici Arama Başlangıcı (SS)")
# Algoritmayla bölgeleri işle ve dikdörtgenleri döndür
rects = ss.process()

proposals = []
boxes = []
output = image.copy()

# Seçici arama sonucu dönen bölgelerde döngü
for (x, y, w, h) in rects[:100]:
    # Rastgele renk oluştur ve bu renkle dikdörtgen çiz
    color = [random.randint(0, 255) for j in range(0, 3)]
    cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
    
    # Bölgeyi gri tonlamaya dönüştür ve boyutunu yeniden ayarla
    roi = image[y:y+h, x:x+w]
    roi = cv2.resize(roi, dsize=(32, 32), interpolation=cv2.INTER_LANCZOS4)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = img_to_array(roi)
    
    proposals.append(roi)
    boxes.append((x, y, w + x, h + y))  # Kutu koordinatlarını kaydet
    
proposals = np.array(proposals, dtype="float64")
boxes = np.array(boxes, dtype="int32")

print("Sınıflandırma")
# Eğitilmiş modeli yükle
pickle_in = open("model_trained.p", "rb")
model = pickle.load(pickle_in)
proba = model.predict(proposals)

number_list = []
idx = []
# Yüksek olasılıklı sayıları ve indeksleri kaydet
for i in range(len(proba)):
    max_prob = np.max(proba[i, :])
    if max_prob > 0.95:
        idx.append(i)
        number_list.append(np.argmax(proba[i]))

# Yüksek olasılıklı sayıları ve kutu koordinatlarıyla eşleştirerek görselde işaretle
for i in range(len(number_list)):
    j = idx[i]
    cv2.rectangle(image, (boxes[j, 0], boxes[j, 1]), (boxes[j, 2], boxes[j, 3]), (0, 255, 0), 2)
    cv2.putText(image, str(np.argmax(proba[j])), (boxes[j, 0] + 5, boxes[j, 1] + 5),
                cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 1)
    cv2.imshow("Image", image)
