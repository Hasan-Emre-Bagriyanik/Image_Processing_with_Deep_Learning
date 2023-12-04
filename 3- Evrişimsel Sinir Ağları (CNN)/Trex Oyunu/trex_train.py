# Gerekli kütüphaneleri import etme
import glob
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from PIL import Image
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns

# Uyarıları görmezden gelme
import warnings
warnings.filterwarnings("ignore")

# ./img dizinindeki tüm PNG dosyalarını listeleme
imgs = glob.glob("./img/*.png")

# Görüntülerin boyutları
width = 125
height = 50

# Giriş (X) ve çıkış (Y) verilerini depolamak için boş listeler
X = []
Y = []

# Tüm görüntülerin üzerinde dönme
for img in imgs:
    
    # Dosya adını ve etiketini çıkarma
    filename = os.path.basename(img)
    label = filename.split("_")[0]
    
    # Görüntüyü açma, boyutlandırma ve normalleştirme
    im = np.array(Image.open(img).convert("L").resize((width, height)))
    im = im / 255
    
    # X ve Y listelerine ekleme
    X.append(im)
    Y.append(label)
    
# X listesini NumPy dizisine dönüştürme
X = np.array(X)
# Görüntü boyutlarını yeniden şekillendirme
X = X.reshape(X.shape[0], width, height, 1)

# Sınıf etiketlerini one-hot encoding'e dönüştürme
def onehot_labels(values):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(sparse = False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded),1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded

Y = onehot_labels(Y)

# Eğitim ve test veri setlerini oluşturma
train_X, test_X, train_y, test_y = train_test_split(X, Y, test_size = 0.25, random_state = 2)    

# CNN modelini oluşturma
model = Sequential()   
model.add(Conv2D(32, kernel_size = (3,3), activation = "relu", input_shape = (width, height, 1)))
model.add(Conv2D(64, kernel_size = (3,3), activation = "relu"))
model.add(Conv2D(64, kernel_size = (3,3), activation = "relu"))
model.add(Conv2D(128, kernel_size = (3,3), activation = "relu"))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation = "relu"))
model.add(Dropout(0.4))
model.add(Dense(3, activation = "softmax"))

# Modeli derleme
model.compile(loss = "categorical_crossentropy", optimizer = "Adam", metrics = ["accuracy"])

# Modeli eğitme
model.fit(train_X, train_y, epochs = 50, batch_size = 32)

# Eğitim veri seti üzerinde modelin doğruluğunu değerlendirme
score_train = model.evaluate(train_X, train_y)
print("Eğitim doğruluğu: %", score_train[1] * 100)    

# Test veri seti üzerinde modelin doğruluğunu değerlendirme
score_test = model.evaluate(test_X, test_y)
print("Test doğruluğu: %", score_test[1] * 100)  

# Modeli JSON formatında kaydetme
open("model_new.json", "w").write(model.to_json())
# Model ağırlıklarını kaydetme
model.save_weights("trex_weight_new.h5")   
