

# import library
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

# Bu kısımda gerekli kütüphaneler (OpenCV, NumPy, Matplotlib, Zaman, Pandas) import edilir.

# Nesne takibi için kullanılacak OpenCV algoritmaları ve isimleri bir sözlükte tanımlanır.
OPENCV_OBJECT_TRACKERS = {"csrt" : cv2.TrackerCSRT_create,
                          "kcf" : cv2.TrackerKCF_create}

# Bu kısımda kullanılacak nesne takip algoritmaları ve isimleri bir sözlükte tanımlanır.

tracker_name = "csrt"
tracker = OPENCV_OBJECT_TRACKERS[tracker_name]()
print("Tracker: ", tracker_name)

# Bu kısımda kullanılacak nesne takip algoritması seçilir ve bir takipçi nesnesi oluşturulur.

# Nesne konumunu belirlemek için gerçek (ground truth) verileri içeren bir dosya okunur.
gt = pd.read_csv("gt_new.txt")

# Bu kısımda bir 'gt_new.txt' dosyasından ground truth (gerçek konum) verileri okunur ve bir pandas DataFrame'e yüklenir.

video_path = "MOT17-13-SDP.mp4"
cap = cv2.VideoCapture(video_path)

# Bu kısımda bir video dosyası açılır.

# Genel parametreler
initBB = None
fps = 25
frame_number = []
f = 0
success_track_frame_success = 0
track_list = []
start_time = time.time()

# Bu kısımda genel parametreler ve takip sonuçlarını depolamak için değişkenler tanımlanır.

while True:
    
    ret, frame = cap.read()
    
    if ret:
        
        frame = cv2.resize(frame, (960,540))
        (H,W) = frame.shape[:2]
        
        # ground truth (gerçek konum) verilerini çerçeve üzerine çizdirme
        car_gt = gt[gt.frame_no == f]
        if len(car_gt) != 0:
            x = car_gt.x.values[0]
            y = car_gt.y.values[0]
            w = car_gt.w.values[0]
            h = car_gt.h.values[0]
 
            center_x = car_gt.center_x.values[0]
            center_y = car_gt.center_y.values[0]
        
            cv2.rectangle(frame, (x,y), (x+w, y+h),(0,255,0),2)
            cv2.circle(frame, (center_x, center_y), 2, (0,0,255),-1 )
            
        # Bu kısımda her bir frame üzerinde ground truth (gerçek konum) bilgilerini çizdirir.

        # Nesneyi takip etme
        if initBB is not None:
            (success, box) = tracker.update(frame)
            
            if f < np.max(gt.frame_no):
            
                (x,y,w,h) = [int(i) for i in box]
                
                cv2.rectangle(frame, (x,y), (x+w, y+h),(0,0,255),2)
                success_track_frame_success += 1
                track_center_x = int(x+w/2)       
                track_center_y = int(y+h/2)
                track_list.append([f, track_center_x, track_center_y])
                
            info = [("Tracker", tracker_name),
                    ("Tracker","Yes" if success else "No")]
            
            for (i, (o,p)) in enumerate(info):
                text = "{}: {}".format(o,p)
                cv2.putText(frame, text, (10, H - (i*20) -10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                
        # Bu kısımda takip sonuçlarını çizdirir ve takip başarısını ekrana yazar.

        cv2.putText(frame, "Frame Number: " + str(f), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        
        # Çerçeveyi gösterme
        cv2.imshow("frame",frame)
        
        # Tuş kontrolü
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord("t"): 
            initBB = cv2.selectROI("Frame", frame, fromCenter=False)
            
            tracker.init(frame)
            
        # Bu kısımda takip başlatma tuşu 't' ile takip başlatma işlemi gerçekleşir.
        
        if key == ord("q"): break
    
        # Çerçeve numarasını güncelleme
        frame_number.append(f)
        f += 1
    
    else: break

# Bu kısımda video tamamlanana kadar her bir frame üzerinde işlemler gerçekleşir.

cap.release()
cv2.destroyAllWindows()

stop_time = time.time()
time_diff = stop_time - start_time
# Değerlendirme

track_df = pd.DataFrame(track_list, columns=["frame_no","center_x","center_y"])

# Bu kısımda takip sonuçları bir pandas DataFrame'e yüklenir.

if len(track_df) != 0:
    print("Tracking method: ", tracker)
    print("Time: ", time_diff)
    print("Number of Frame to track (gt): ", len(gt))
    print("Number of frame track (track success): ", success_track_frame_success)
    
    track_df_frame = track_df.frame_no
    
    gt_center_x = gt.center_x[track_df_frame].values
    gt_center_y = gt.center_y[track_df_frame].values
    
    track_df_center_x = track_df.center_x.values
    track_df_center_y = track_df.center_y.values
    
    plt.plot(np.sqrt(((gt_center_x - track_df_center_x) ** 2 )+(gt_center_y - track_df_center_y)))
