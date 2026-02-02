import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

#Todas as imagens serão redimensionadas para (244, 244)
img_size = (224, 224)

#Treinamento ocorre em pacotes de 32 imagens por vez
batch_size = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

#Carrega imagens organizadas em pastas
train_data = datagen.flow_from_directory(
    "dataset/train",
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

#Não utiliza data augmentation, pois as imagens precisam ser reais, sem modificação
val_data = ImageDataGenerator(rescale=1./255).flow_from_directory(
    "dataset/valid",
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

#Construção da CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')
])

#Compila o modelo
#optimizer -> algoritmo que ajusta os pesos
#loss='categorical_crossentropy' -> função de erro para classificação múltipla
#acuracy -> métrica de acerto
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Treinamento
history = model.fit(train_data, validation_data=val_data, epochs=30)

#Salvando o modelo
model.save("modelo_cnn_bflp.keras") #bflp: bugatti, ferrari, lamborghini e porsche

#Mapeamento de classes
classe_indices = train_data.class_indices

#salvando indices das classes
with open("classes_carros.json", "w") as f:
    json.dump(classe_indices, f)

#Curva de LOSS
plt.figure()
plt.plot(history.history['loss'], label='Loss Treino')
plt.plot(history.history['val_loss'], label='Loss Validação')
plt.title('Curva de Loss')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()
plt.savefig("loss.png")
plt.close()

#Curva de Accuracy
plt.figure()
plt.plot(history.history['accuracy'], label='Accuracy Treino')
plt.plot(history.history['val_accuracy'], label='Accuracy Validação')
plt.title('Curva de Accuracy')
plt.xlabel('Épocas')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("accuracy.png")
plt.close()