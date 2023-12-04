import cv2
import pickle
import numpy as np
import time

def preProcess(img):
    # Görüntüyü gri tonlamaya çevirme
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Histogram eşitleme işlemi uygulama
    img = cv2.equalizeHist(img)
    
    # Görüntüyü normalize etme (0 ile 1 arasına getirme)
    img = img / 255.0
    
    return img

# Kamerayı başlatma
cap = cv2.VideoCapture(0)
cap.set(3, 480)
cap.set(4, 480)
time.sleep(2)

# Eğitilmiş modeli yükleme
pickle_in = open("model_trained.p", "rb")
model = pickle.load(pickle_in)

while True:
    
    # Kameradan bir çerçeve okuma
    success, frame = cap.read()
    
    # Çerçeveyi NumPy dizisine dönüştürme
    img = np.asarray(frame)
    
    # Görüntüyü 32x32 boyutlarına yeniden boyutlandırma
    img = cv2.resize(img, (32, 32))
    
    # Görüntüyü önişleme işlemlerine tabi tutma
    img = preProcess(img)

    # Görüntüyü CNN modeli için uygun formata dönüştürme
    img = img.reshape(1, 32, 32, 1)
    
    
    
    predictions = model.predict(img)
    classIndex = np.argmax(predictions)
    probVal = np.amax(predictions)
    
    print(classIndex, probVal)
    
    # Eğer tahmin olasılığı belirlenen bir eşik değerinden büyükse
    if probVal > 0.5:
        # Çerçeve üzerine tahmin sınıfını ve olasılığı yazma
        cv2.putText(frame, str(classIndex) + "  " + str(probVal), (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0, 1))
        
    # Çerçeve üzerinde sonuçları gösterme
    cv2.imshow("Rakam Sınıflandırma", frame)
    
    # Eğer 'q' tuşuna basılırsa döngüden çıkma
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Kamerayı serbest bırakma ve pencereyi kapatma
cap.release()
cv2.destroyAllWindows()
