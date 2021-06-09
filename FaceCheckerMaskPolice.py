import os
import tensorflow as tf
import numpy as np
import pytesseract
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

train_dir = '/home/erik/PycharmProjects/proyectoFinalSI/observations-master/experiements/data'
train_mask_dir = os.path.join(train_dir, 'with_mask')
train_no_mask_dir = os.path.join(train_dir, 'without_mask')

print('total imagenes con mascarilla:', len(os.listdir(train_mask_dir)))
print('total imagenes sin mascarilla:', len(os.listdir(train_no_mask_dir)))

train_mask_fnames = os.listdir(train_mask_dir)
train_no_mask_fnames = os.listdir(train_no_mask_dir)
print(train_mask_fnames[:10])
print(train_no_mask_fnames[:10])

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(lr=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

img_height = 150
img_width = 150
batch_size = 10
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='training')

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation')

nb_epochs = 10
history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=nb_epochs)


def comprobardni(dni):
    fo = open("personasautorizadas.txt", "r")
    file_contents = fo.read()
    Flag = 0
    for i in file_contents.split('\n'):
        if dni == i:
            Flag = 1
    if Flag == 1:
        print('Persona autorizada')
    else:
        print('Persona no autorizada')


def comprobarlongitud(dni):
    dni = ''.join(i for i in dni if i.isdigit())
    longitud = len(dni)
    if dni.isdecimal() == True:
        if longitud != 8:
            dni = dni[0:8]

    return dni


def grabardni():
    comprobar = 'N'

    cap = cv2.VideoCapture("/dev/video2")
    cap.set(3, 640)
    cap.set(4, 480)

    while comprobar != 's':
        timer = cv2.getTickCount()
        _, img = cap.read()

        hImg, wImg, _ = img.shape
        conf = r'--oem 3 --psm 6 outputbase digits'
        boxes = pytesseract.image_to_boxes(img, config=conf)
        for b in boxes.splitlines():
            # print(b)
            b = b.split(' ')
            # print(b)
            x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
            cv2.rectangle(img, (x, hImg - y), (w, hImg - h), (50, 50, 255), 2)
            cv2.putText(img, b[0], (x, hImg - y + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
        cv2.putText(img, str(int(fps)), (75, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 230, 20), 2);
        cv2.imshow("Result", img)
        comprobar = input("Pulsa s/n para aceptar o repetir la imagen: ")
        cv2.waitKey(1)

    dni = pytesseract.image_to_string(img)
    print(dni)
    dniLimpio = comprobarlongitud(dni)
    print(dniLimpio)
    comprobardni(dniLimpio)


def comprobarmascarilla():
    asegurarmascarilla = 0
    vc = cv2.VideoCapture(0)
    plt.ion()
    if vc.isOpened():
        is_capturing, frame = vc.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        webcam_preview = plt.imshow(frame)
    else:
        is_capturing = False

    while is_capturing:
        try:
            is_capturing, frame = vc.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_res = cv2.resize(frame, dsize=(150, 150), interpolation=cv2.INTER_CUBIC)
            x = image.img_to_array(frame_res)
            x = np.expand_dims(x, axis=0)
            images = np.vstack([x])
            classes = model.predict(images, batch_size=10)
            if classes[0] > 0:
                print("No Lleva mascarilla")
                asegurarmascarilla = 0
            else:
                print("Si lleva mascarilla")
                asegurarmascarilla = asegurarmascarilla + 1
                if asegurarmascarilla == 10:
                    asegurarmascarilla = 0
                    break
            webcam_preview = plt.imshow(frame)
            try:
                plt.pause(1)
            except Exception:
                pass
        except KeyboardInterrupt:
            vc.release()
    vc.release()
    grabardni()


while True:
    input("Pulse cualquier tecla para iniciar el proceso ")
    comprobarmascarilla()
    print("\n\n\n\n\n")
