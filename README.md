## Vehicle-Type-Classification

### Vehicle Type Classification loyihasi uchun quyidagi bosqichlarni ko'rib chiqish kerak:
### 1. Dataset yig'ish.
* Dataset yig'ish hududga bog'liq holatda shakllantirilishi kerak. Ya'ni, dataset tarkibida hududga tegishli bo'lmagan mashina tasvirlari bo'lmasligi kerak.
* Tasvirlarni kamera ko'ra oladigan holat uchun moslab yig'ish modelni xususiy hol uchun a'lo darajada ishlashini ta'minlaydi.
* Dataset to'plashda balansni saqlash kerak. Har bir turga mos bo'lgan datalardan teng miqdorda bo'lib olish kerak.

### 2. Dataset tozalash.

### 3. Datasetni classlarga ajratish.

### 4. Model train qilish uchun kerakli texnologiya yoki arxitekturani qurish.
* Tensorflow tarkibidagi Keras kutubxonasidan foydalanib mashhur arxitekturalar yordamida model train qilish mumkin.

```python
import glob
import numpy as np
import tensorflow as tf
import cv2
from sklearn.model_selection import train_test_split

images = []
labels = []

for path in glob.glob("dataset/*"):
    label = path[8:]

    for image_path in glob.glob(path+"/*"):
        image = cv2.imread(image_path)
        images.append(image)
        labels.append(label)
    
def preprocessing(images,labels):
    label_class = ["car","van","minitruck","truck","bus"]
    
    images_ = []
    labels_ = []
    
    for i in range(len(images)):
        image = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(224,224),interpolation = cv2.INTER_AREA)
        image = image/255.
        image = image.astype('float16')

        label = label_class.index(labels[i])
        
        images_.append(image)
        labels_.append(label)
        
    labels = np.array(labels_).astype('uint16')
    images = np.array(images_)
    
    return images,labels

images, labels = preprocessing(images,labels)
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.15, random_state=42)

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

densenet121 = tf.keras.applications.DenseNet121(input_shape=(224,224,3), include_top=False)
densenet121.trainable=False

model = tf.keras.Sequential([
    densenet121,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(2048,activation = tf.keras.activations.relu),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1024,activation = tf.keras.activations.relu),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(512,activation = tf.keras.activations.relu),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(256,activation = tf.keras.activations.relu),
    tf.keras.layers.Dense(128,activation = tf.keras.activations.relu),
    tf.keras.layers.Dense(10,activation = tf.keras.activations.relu),
    tf.keras.layers.Dense(5,activation = tf.keras.activations.softmax),
])

model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics = tf.keras.metrics.SparseCategoricalAccuracy())

history = model.fit(train_ds, epochs = 5, validation_data = test_ds)

model.save("VehicleType2.h5")
```
* Yolov8.1 texnologiyasining classification uchun moslashgan tayyor modellaridan foydalanib Fine Tuning qilish.

### 5. Modelni sinovdan o'tkazish.
Modelni mavjud datalardan emas balki kameradan olingan yangi tasvirlar orqali test qilish kerak.

