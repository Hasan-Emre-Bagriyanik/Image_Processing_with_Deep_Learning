# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 12:00:55 2023

@author: Hasan Emre
"""

#%% import library

# Gerekli kütüphaneleri içe aktarın
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Kolon adlarını belirleyin ve "gt.txt" dosyasını bir DataFrame'e yükleyin
col_list = ["frame_number","identity_number","left","top","width","height","score","class","visibility"]
data = pd.read_csv("gt.txt", names=col_list)

# Sınıfları saydırarak bir çubuk grafik çizin
plt.figure()
sns.countplot(data["class"])

# "class" sütunu 3 olan verileri "car" adlı bir DataFrame'e filtreleyin
car = data[data["class"] == 3]

# Video dosyasının yolu
video_path = "MOT17-13-SDP.mp4"

# Video yakalama nesnesi oluşturun
cap = cv2.VideoCapture(video_path)

# İzlenecek nesnenin kimliği (identity_number)
id1 = 29

# Video dosyasındaki toplam kare sayısını alın
numberOfImage = np.max(data["frame_number"])

# Video çerçevesi hızı (frames per second)
fps = 25

# Nesnenin sınırlayıcı kutularını depolamak için bir liste oluşturun
bound_box_list = []

# Her bir video karesini işlemek için döngü
for i in range(numberOfImage):
    ret, frame = cap.read()
    
    if ret:
        # Kare boyutunu yeniden boyutlandırın
        frame = cv2.resize(frame, dsize=(960, 540))
        
        # Belirli bir karedeki istenen nesneyi filtrelemek için koşullu indeksleme kullanın
        filter_id1 = np.logical_and(car["frame_number"] == i + 1, car["identity_number"] == id1)
        
        # Filtre sonucunda en az bir eşleşme varsa
        if len(car[filter_id1]) != 0:
            # Nesne sınırlayıcı kutusunun koordinatlarını hesaplayın
            x = int(car[filter_id1]['left'].values[0] / 2)
            y = int(car[filter_id1]['top'].values[0] / 2)
            w = int(car[filter_id1]['width'].values[0] / 2)
            h = int(car[filter_id1]['height'].values[0] / 2)
            
            # Sınırlayıcı kutuyu ve kutunun merkezini çerçeveye çizin
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (int(x + w / 2), int(y + h / 2)), 2, (0, 0, 255), -1)
            
            # Kare, x, y, genişlik, yükseklik, merkez_x, merkez_y bilgilerini listeye ekleyin
            bound_box_list.append([i, x, y, w, h, int(x + w / 2), int(y + h / 2)])

        # Kare üzerine bilgi yazısı ekleyin
        cv2.putText(frame, "Frame Number: " + str(i + 1), (10, 30), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Çerçeveyi görüntüleyin
        cv2.imshow("Frame", frame)
        
        # "q" tuşuna basıldığında döngüden çıkın
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Video yakalama nesnesini serbest bırakın ve tüm pencereleri kapatın
cap.release()
cv2.destroyAllWindows()

# kullandığımız listeyi kaydettik daha sonra kulanabilmek için
df = pd.DataFrame(bound_box_list, columns=["frame_no","x","y","w","h","center_x","center_y"])
df.to_csv("gt_new.txt", index=False)
