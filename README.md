## Vehicle-Type-Classification

### Vehicle Type Classification loyihasi uchun quyidagi bosqichlarni ko'rib chiqish kerak:
### 1. Dataset yig'ish.
* Dataset yig'ish hududga bog'liq holatda shakllantirilishi kerak. Ya'ni, dataset tarkibida hududga tegishli bo'lmagan mashina tasvirlari bo'lmasligi kerak.
* Tasvirlarni kamera ko'ra oladigan holat uchun moslab yig'ish modelni xususiy hol uchun a'lo darajada ishlashini ta'minlaydi.
* Dataset to'plashda balansni saqlash kerak. Har bir turga mos bo'lgan datalardan teng miqdorda bo'lib olish kerak.

### 2. Dataset tozalash.

### 3. Datasetni classlarga ajratish.
Classlarga ajratishda transport turlarining bir biriga o'xshashlari iloji boricha bitta classga to'planishi kerak. Masalan, labo bilan damas yoki GAZel bilan shatakka oluvchilarning boshqa classlarda bo'lishi modelning xato ishlashiga olib keladi.

Tensorflow uchun:
dataset
|_type1
|_type2
|_type3
|_type4
|_type5

Yolov8 da Fine Tuning uchun:
dataset
|_train
  |_type1
  |_type2
  ...
|_val
  |_type1
  |_type2
  ...

### 4. Model train qilish uchun kerakli texnologiya yoki arxitekturani qurish.
* Tensorflow tarkibidagi Keras kutubxonasidan foydalanib mashhur arxitekturalar yordamida model train qilish mumkin.

```python
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import glob
import cv2

images = []
labels = []

for path in glob.glob("dataset/*"):
    label = path[8:]
    for image_path in glob.glob(path+"/*"):
        image = cv2.imread(image_path)
        images.append(image)
        labels.append(label)
    
def preprocessing(images,labels):
    label_class = ["type1","type2","type3","type4","type5"]
    
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

```python
from ultralytics import YOLO
model = YOLO('yolov8x-cls.pt')
results = model.train(data='dataset/', epochs=100, imgsz=224)
#yolo task=classify mode=train model=yolov8x-cls.pt data='dataset/' epochs=100 imgsz=224
```

### 5. Modelni sinovdan o'tkazish.
Modelni mavjud datalardan emas, balki kameradan olingan yangi tasvirlar orqali test qilish kerak.
```python
from ultralytics import YOLO
import cv2
import torch
import numpy as np

model1 = YOLO("yolov8x.pt")
model2 = YOLO("best.pt")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
label_class = [type1, type2, type3, type4, type5]

video = cv2.VideoCapture('8.mp4')
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame = video.read()
    if not ret:
        break
    frame = cv2.resize(frame,(width,height),interpolation = cv2.INTER_AREA)
        
    results = model1.predict(frame, conf=0.5, stream = True,device = device, classes = [1,2,3,5,7])
    
    for result in results:
        boxes = result.boxes.cpu().numpy()
        
        for box in boxes:
            r = box.xyxy[0].astype(int)
            xmin,ymin,xmax,ymax = r
            vehicle = frame[ymin:ymax, xmin:xmax]
            
            pred = model2.predict(vehicle, stream = False)
            cv2.rectangle(frame, r[:2], r[2:], (255, 255, 255), 2)
            
            frame = cv2.putText(frame, label_class[pred[0].probs.top1], r[:2], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow("input", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
video.release()
cv2.destroyAllWindows()   
```
