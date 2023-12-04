# Gerekli kütüphaneleri ve modülleri içeri aktar
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import numpy as np
import cv2

# Kendi yazdığınız modüllerden gerekli fonksiyonları içeri aktar
from Sliding_windows import sliding_windows
from Image_Pyramid import image_pyramid
from non_max_suppression import non_max_suppression

# Sabit parametreleri tanımla
WIDTH = 600
HEIGHT = 600
PYR_SCALE = 1.5
WIN_STEP = 16
ROI_SIZE = (200, 150)
INPUT_SIZE = (224, 224)

# ResNet50 modelini 'imagenet' veri kümesi ile yükle
print("Resnet yükleniyor")
model = ResNet50(weights="imagenet", include_top=True)

# Resmi yükle ve boyutunu ayarla
orig = cv2.imread("husky.jpg")
orig = cv2.resize(orig, dsize=(WIDTH, HEIGHT))
cv2.imshow("Husky", orig)

# Görüntünün yükseklik ve genişliğini al
(H, W) = orig.shape[:2]

# Görüntü piramidi oluştur
pyramid = image_pyramid(orig, PYR_SCALE, ROI_SIZE)

# ROI'lar ve konumlar için boş listeler oluştur
rois = []
locs = []

# Görüntü piramidi üzerinde döngü
for image in pyramid:
    scale = W / float(image.shape[1])
    
    # Kayan pencere işlemi gerçekleştir
    for (x, y, roiOrig) in sliding_windows(image, WIN_STEP, ROI_SIZE):
        x = int(x * scale)
        y = int(y * scale)
        w = int(ROI_SIZE[0] * scale)
        h = int(ROI_SIZE[1] * scale)
        
        # ROI boyutunu ayarla ve ön işleme geçir
        roi = cv2.resize(roiOrig, INPUT_SIZE)
        roi = img_to_array(roi)
        roi = preprocess_input(roi)
        
        # ROI'ları ve konumları ilgili listelere ekle
        rois.append(roi)
        locs.append((x, y, x + w, y + h))

# ROI'ları NumPy dizisine dönüştür
rois = np.array(rois, dtype="float32")

# Sınıflandırma işlemi gerçekleştir
print("Sınıflandırma işlemi")
preds = model.predict(rois)
preds = imagenet_utils.decode_predictions(preds, top=1)

# Etiketleri ve en düşük güven eşiği değerini tanımla
labels = {}
min_conf = 0.9

# Her bir tahmin için işlem yap
for (i, p) in enumerate(preds):
    (_, label, prob) = p[0]
    
    # Belirlenen güven eşiğini karşılayan etiketleri kaydet
    if prob >= min_conf:
        box = locs[i]
        L = labels.get(label, [])
        L.append(box)  # Etiket ve kutuyu etiket sözlüğüne ekle
        labels[label] = L

# Her bir etiket için işlem yap
for label in labels.keys():
    clone = orig.copy()
    
    # Etiketlere göre kutuları ve etiketleri görüntüye ekle
    for box in labels[label]:
        (startX, startY, endX, endY) = box
        cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 255, 0), 2)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(clone, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    
    # Kutuların bulunduğu görüntüyü göster
    cv2.imshow("ilk", clone)
    
    clone = orig.copy()
    
    # Non-maxima suppression işlemi uygula
    boxes = np.array(labels[label])
    boxes = non_max_suppression(boxes)
    
    # Kutucukları çiz ve etiketleri ekle
    for (startX, startY, endX, endY) in boxes:
        cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 255, 0), 2)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(clone,(startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    
    # Kutulu görüntüyü göster
    cv2.imshow("Maxima", clone)
    
    # 'q' tuşuna basıldığında döngüden çık
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Görüntüleme işlemini kapat
cv2.destroyAllWindows()

