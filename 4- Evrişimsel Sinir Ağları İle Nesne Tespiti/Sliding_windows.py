import cv2 
import matplotlib.pyplot as plt

# Kaydırma penceresi oluşturan fonksiyon
def sliding_windows(image, step, ws):
    # Görüntü üzerinde yatay (height) boyunca kaydırma işlemi
    for y in range(0, image.shape[0] - ws[1], step):
        # Görüntü üzerinde dikey (width) boyunca kaydırma işlemi
        for x in range(0, image.shape[1] - ws[0], step):
            # Belirtilen boyutta bir kaydırma penceresi oluştur
            # (x, y) koordinatları ile pencerenin sol üst köşesini belirle
            # image[y:y+ws[1], x:x+ws[0]] ifadesiyle pencere boyutunu al
            yield (x, y, image[y:y + ws[1], x:x + ws[0]])
            
# img = cv2.imread("husky.jpg")
# im = sliding_windows(img, 5, (200,150))
# for i , image in enumerate(im):
#     print(i)

#     if i == 10190:
#         print(image[0], image[1])
#         plt.imshow(image[2])
        
