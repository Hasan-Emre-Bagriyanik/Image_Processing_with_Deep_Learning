import cv2
import matplotlib.pyplot as plt

def image_pyramid(image, scale=1.5, minSize=(224, 224)):
    """
    Görüntü piramidi oluşturan fonksiyon.
    
    Parametreler:
    - image: Görüntü
    - scale: Ölçek faktörü (varsayılan değer: 1.5)
    - minSize: Minimum boyut (varsayılan değer: (224, 224))
    
    Fonksiyon, orijinal görüntüyü ve bir ölçek faktörünü alır. 
    Daha sonra, belirtilen ölçekte bir piramit oluşturur ve minimum boyuta ulaşana kadar bu işlemi tekrarlar.
    
    Yield ifadesi, her bir ölçek boyutundaki görüntüyü üretmek için kullanılır.
    """

    # Orijinal görüntüyü yield et
    yield image
    
    while True:
        # Yeni boyutları hesapla ve görüntüyü yeniden boyutlandır
        w = int(image.shape[1] / scale)
        image = cv2.resize(image, dsize=(w, w))
        
        # Yeni boyutların minimum boyutu aşmadığını kontrol et
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        
        # Yeni boyuttaki görüntüyü yield et
        yield image

        
# img = cv2.imread("husky.jpg")
# im = image_pyramid(img, 1.5, (10,10))
# for i, image in enumerate(im):
#     print(i)
#     if i == 0:
#         plt.imshow(image)
    
        