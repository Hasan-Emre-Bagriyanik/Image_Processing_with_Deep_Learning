# Gerekli kütüphaneleri import etme
import keyboard
import uuid  # Ekrandan kayıt alacağız
import time
from PIL import Image
from mss import mss

# Kayıt yapılacak ekranın konumunu belirleme
mon = {"top": 515, "left": 700, "width": 300, "height": 120}
sct = mss()

# Kayıt sayacını başlatma
i = 0

# Ekran kaydını gerçekleştiren fonksiyon
def record_screen(record_id, key):
    global i
    
    # Kayıt sayacını artırma
    i += 1
    print("{}: {}".format(key, i))
    
    # Ekran görüntüsü alma
    img = sct.grab(mon)
    
    # Pillow kütüphanesi ile görüntüyü oluşturma
    im = Image.frombytes("RGB", img.size, img.rgb)
    
    # Görüntüyü kaydetme
    im.save("./img/{}_{}_{}.png".format(key, record_id, i))
    
# Programdan çıkışı kontrol eden değişken
is_exit = False

# Çıkış fonksiyonu
def exit():
    global is_exit
    is_exit = True
    
# "esc" tuşuna basıldığında çıkış fonksiyonunu çağırma
keyboard.add_hotkey("esc", exit)

# Kayıt kimliği oluşturma
record_id = uuid.uuid4()

# Ana döngü
while True:
    
    # Programdan çıkış yapılmışsa döngüyü sonlandırma
    if is_exit:
        break

    try: 
        # Yukarı ok tuşuna basıldığında kayıt yapma
        if keyboard.is_pressed(keyboard.KEY_UP):
            record_screen(record_id, "up")
            time.sleep(0.1)
        # Aşağı ok tuşuna basıldığında kayıt yapma
        elif keyboard.is_pressed(keyboard.KEY_DOWN):
            record_screen(record_id, "down")
            time.sleep(0.1)
        # Sağ ok tuşuna basıldığında kayıt yapma
        elif keyboard.is_pressed("right"):
            time.sleep(0.1)
            record_screen(record_id, "right")
            
    except RuntimeError:
        continue
