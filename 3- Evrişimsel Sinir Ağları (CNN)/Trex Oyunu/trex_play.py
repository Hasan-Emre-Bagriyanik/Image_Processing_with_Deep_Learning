# Gerekli kütüphaneleri import etme
from keras.models import model_from_json
import numpy as np
from PIL import Image
import keyboard
import time
from mss import mss

# Kayıt ekranını belirleme
mon = {"top": 515, "left": 700, "width": 300, "height": 120}
sct = mss()

# Görüntü boyutları
width = 125
height = 50

# Modeli yükleme
model = model_from_json(open("model.json", "r").read())
model.load_weights("trex_weight.h5")

# Down = 0, Right = 1, Up = 2
labels = ["Down", "Right", "Up"]

# FPS (Frame Per Second) hesaplamak için kullanılan değişkenler
framerate_time = time.time()
counter = 0

# Diğer değişkenler
i = 0
delay = 0.4
key_down_pressed = False

while True:
    # Ekran görüntüsü almak
    img = sct.grab(mon)
    im = Image.frombytes("RGB", img.size, img.rgb)
    
    # Görüntüyü işlemek
    im2 = np.array(im.convert("L").resize((width, height)))
    im2 = im2 / 255
    
    # Model için girişi hazırlamak
    x = np.array([im2])
    x = x.reshape(x.shape[0], width, height, 1)
    
    # Modeli kullanarak tahmin yapmak
    r = model.predict(x)
    
    # Tahmin sonucunu almak
    result = np.argmax(r)
    
    # Down (Aşağı) durumu
    if result == 0:
        keyboard.press(keyboard.KEY_DOWN)
        key_down_pressed = True
    
    # Up (Yukarı) durumu
    elif result == 2:
        if key_down_pressed:
            keyboard.release(keyboard.KEY_DOWN)
        time.sleep(delay)
        keyboard.press(keyboard.KEY_UP)
        
        # Bekleme süresini ayarlamak
        if i < 1500:
            time.sleep(0.3)
        elif 1500 < i < 5000:
            time.sleep(0.2)
        else:
            time.sleep(0.17)
        
        # Down tuşuna basma
        keyboard.press(keyboard.KEY_DOWN)
        keyboard.release(keyboard.KEY_DOWN)
    
    # FPS hesaplama ve kontrol
    counter += 1
    if (time.time() - framerate_time) > 1:
        counter = 0
        framerate_time = time.time()
        
        # Gecikme süresini ayarlama
        if i <= 1500:
            delay -= 0.003
        else:
            delay -= 0.005
        
        if delay < 0:
            delay = 0
            
        # Bilgileri ekrana yazdırma
        print("-----------------------------------")
        print("Down: {}    Right: {}   Up: {}".format(r[0][0], r[0][1], r[0][2]))
        i += 1
