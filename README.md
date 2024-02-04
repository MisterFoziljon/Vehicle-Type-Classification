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
* Yolov8.1 texnologiyasining classification uchun moslashgan tayyor modellaridan foydalanib Fine Tuning qilish.

### 5. Modelni sinovdan o'tkazish.
Modelni mavjud datalardan emas balki kameradan olingan yangi tasvirlar orqali test qilish kerak.

