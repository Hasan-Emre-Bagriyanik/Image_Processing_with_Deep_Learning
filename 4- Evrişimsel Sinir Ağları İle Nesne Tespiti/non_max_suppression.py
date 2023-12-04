
import cv2
import numpy as np

def non_max_suppression(boxes, probs=None, overlapThreshold=0.3):
    """
    Non-maximum suppression (NMS) algoritması.
    
    Parametreler:
    - boxes: Tespit edilen dikdörtgen kutuların koordinatları (x1, y1, x2, y2)
    - probs: Opsiyonel olarak tespit edilen kutuların olasılık değerleri
    - overlapThreshold: Örtüşme eşik değeri (varsayılan değer: 0.3)
    
    Fonksiyon, tespit edilen kutular üzerinde non-maximum suppression (NMS) işlemi gerçekleştirir.
    Bu işlem, yüksek olasılığa sahip kutuları korurken, örtüşen (overlap) ve zayıf kutuları eler.
    """

    # Kutu dizisi boşsa, boş bir liste döndür
    if len(boxes) == 0:
        return []

    # Eğer kutuların veri tipi integer ise, float'a dönüştür
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # Kutuların koordinatlarını ayrıştır
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Kutu alanlarını hesapla
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Olasılık değerleri varsa, indeksleri bu değerlere göre atayın
    idxs = y2
    if probs is not None:
        idxs = probs

    # İndeksleri olasılıklara göre sırala
    idxs = np.argsort(idxs)

    pick = []  # Seçilen kutuların indeksleri

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # Tüm kutuların koordinatlarını tek bir boyutta al
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # Genişlik ve yükseklikleri hesapla
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Örtüşmeyi (overlap) hesapla
        overlap = (w * h) / area[idxs[:last]]

        # Belirlenen eşik değerinden büyük örtüşmeleri içeren kutuları kaldır
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThreshold))))

    # Seçilen kutuların koordinatlarını integer'a dönüştürerek döndür
    return boxes[pick].astype("int")


































