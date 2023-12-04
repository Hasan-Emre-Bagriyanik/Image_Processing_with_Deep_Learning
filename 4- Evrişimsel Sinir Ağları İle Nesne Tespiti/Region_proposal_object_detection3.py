from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import numpy as np
import cv2

from non_max_suppression import non_max_suppression  # non_max_suppression modülünü içe aktar

# Seçici arama fonksiyonunu tanımla
def selective_search(image):
    # Seçici arama algoritmasını oluştur
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchQuality()
    
    # İşlenmiş bölgeleri elde et
    rects = ss.process()
    
    return rects[:1000]  # İlk 1000 bölgeyi döndür

# Modeli ve görüntüyü yükle
model = ResNet50(weights="imagenet")
image = cv2.imread("pyramid.jpg")
image = cv2.resize(image, dsize=(600, 600))
(H, W) = image.shape[:2]

# Seçici arama ile dikdörtgen bölgeleri al
rects = selective_search(image)

boxes = []
proposals = []

# Her bir bölge için işlem yap
for (x, y, w, h) in rects:
    # Boyut kontrolü yap
    if W / float(W) < 0.1 or h / float(H) < 0.1:
        continue
    
    # ROI oluştur
    roi = image[y:y + h, x:x + w]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, (224, 224))
    
    # ROI'yi diziye dönüştür ve işle
    roi = img_to_array(roi)
    roi = preprocess_input(roi)
    
    # Önerileri ve kutuları listelere ekle
    proposals.append(roi)
    boxes.append((x, y, w, h))

# Önerileri numpy dizisine dönüştür
proposals = np.array(proposals)

# Tahminler yap
preds = model.predict(proposals)
preds = imagenet_utils.decode_predictions(preds, top=1)

min_conf = 0.5  # Minimum güven eşiği
labels = {}

# Her bir tahmin için işlem yap
for (i, p) in enumerate(preds):
    (_, label, prob) = p[0]
    
    # Güven eşiği kontrolü
    if prob >= min_conf:
        (x, y, x, y) = boxes[i]
        box = (x, y, x + w, y + h)
        L = labels.get(label, [])
        L.append((box, prob))
        labels[label] = L

# Kopya bir görüntü oluştur
clone = image.copy()

# Etiketlerin üzerinde dön
for label in labels.keys():
    for (box, prob) in labels[label]:
        boxes = np.array([p[0] for p in labels[label]])
        proba = np.array([p[1] for p in labels[label]])
        boxes = non_max_suppression(boxes, proba)
        
        # Her bir kutu için işlem yap
        for (startX, startY, endX, endY) in boxes:
            # Kutuyu çiz
            cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 0, 255), 2)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            
            # Etiketi ekle
            cv2.putText(clone, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Görüntüyü göster
        cv2.imshow("After", clone)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
