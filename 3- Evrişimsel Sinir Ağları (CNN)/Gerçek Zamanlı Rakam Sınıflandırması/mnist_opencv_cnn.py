# Gerekli kütüphaneleri import etme
import numpy as np 
import cv2
import os 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns 
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import pickle

# Veri setinin bulunduğu dizin
path = "myData"

# Dizin içindeki sınıf (label) sayısını belirleme
myList = os.listdir(path)
noOfClasses = len(myList)
print("Label (sınıf) sayısı: ", noOfClasses)

# Görüntüleri ve sınıf etiketlerini depolamak için boş listeler
images = []
classNo = []

# Her bir sınıftaki görüntüleri yükleme
for i in range(noOfClasses):
    myImageList = os.listdir(path + "//" + str(i))
    for j in myImageList:
        # Görüntüyü okuma ve boyutlandırma
        img = cv2.imread(path + "//" + str(i) + "//" + j)
        img = cv2.resize(img, (32,32))
        images.append(img)
        classNo.append(i)

# Görüntüleri NumPy dizisine dönüştürme
images = np.array(images)
classNo = np.array(classNo)

# Veri seti boyutlarını yazdırma
print(images.shape)
print(classNo.shape)

# Veriyi eğitim, test ve doğrulama setlerine ayırma
x_train, x_test, y_train, y_test = train_test_split(images, classNo, test_size=0.5, random_state=42)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.5, random_state=42)

# Veri seti boyutlarını tekrar yazdırma
print(images.shape)
print(x_train.shape)
print(x_test.shape)
print(x_validation.shape)

# Veri setindeki sınıf (label) sayılarını görselleştirme
# Bu kısım, yorumlamanızda devre dışı bırakılmış durumda, eğer kullanacaksanız tekrar aktif edilebilir.

# Görüntü ön işleme fonksiyonu
def preProcess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

# Ön işleme sonrası bir örnek görüntüyü görselleştirme
idx = 2000
img = preProcess(x_train[idx])
img = cv2.resize(img, (300, 300))
cv2.imshow("Preprocess ", img)

# Ön işleme fonksiyonunu veri setine uygulama
x_train = np.array(list(map(preProcess, x_train)))
x_test = np.array(list(map(preProcess, x_test)))
x_validation= np.array(list(map(preProcess, x_validation)))

# Veri setini CNN modeli için uygun formata dönüştürme
x_train = x_train.reshape(-1,32,32,1)
print(x_train.shape)
x_test = x_test.reshape(-1,32,32,1)
x_validation = x_validation.reshape(-1,32,32,1)

# Veri artırma (data augmentation) işlemi
dataGen = ImageDataGenerator(width_shift_range=0.1, 
                             height_shift_range= 0.1,
                             zoom_range=0.1,
                             rotation_range= 10)

dataGen.fit(x_train)

# Sınıf etiketlerini kategorik formata dönüştürme
y_train = to_categorical(y_train, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)

# CNN modelini oluşturma
model = Sequential()
model.add(Conv2D(filters = 8, kernel_size = (5,5),input_shape = (32,32,1), activation="relu", padding = "same"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters = 16, kernel_size = (3,3), activation="relu", padding = "same"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(units = 256, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units = noOfClasses,activation="softmax"))

# Modeli derleme
model.compile(loss = "categorical_crossentropy", optimizer = "Adam", metrics = ["accuracy"])

# Eğitim için batch boyutu
batch_size = 250

# Modeli eğitme
hist = model.fit_generator(dataGen.flow(x_train, y_train, batch_size = batch_size),
                                        validation_data = (x_validation ,y_validation),
                                        epochs=15, steps_per_epoch = x_train.shape[0]//batch_size, shuffle=1)

# Eğitilmiş modeli dosyaya kaydetme
pickle_out = open("model_trained.p","wb")
pickle.dump(model,pickle_out)
pickle_out.close()

# Eğitim sonuçlarını görselleştirme
plt.figure()
plt.plot(hist.history["loss"], label = "Eğitim loss")
plt.plot(hist.history["val_loss"], label = "Val loss")
plt.legend()
plt.show()

plt.figure()
plt.plot(hist.history["accuracy"], label = "Eğitim accuracy")
plt.plot(hist.history["val_accuracy"], label = "Val accuracy")
plt.legend()
plt.show()

# Test seti üzerinde modelin performansını değerlendirme
score = model.evaluate(x_test, y_test, verbose=1)
print("Test loss: ", score[0])
print("Test accuracy: ", score[1])

# Doğrulama seti üzerinde confusion matrix oluşturma ve görselleştirme
y_pred = model.predict(x_validation)
y_pred_class = np.argmax(y_pred, axis = 1)
y_true = np.argmax(y_validation, axis=1)
cm = confusion_matrix(y_true, y_pred_class)

f,ax = plt.subplots(figsize=(8,8))
sns.heatmap(cm, annot=True, linewidths=0.01, cmap = "Greens", linecolor="gray",fmt = ".1f", ax=ax)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
