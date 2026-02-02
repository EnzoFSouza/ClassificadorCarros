import tensorflow as tf
import numpy as np
import cv2
import json
from tensorflow.keras.preprocessing import image

#Carregando o modelo
model = tf.keras.models.load_model("modelo_cnn_bflp.keras")

#Carregando as classes
with open("classes_carros.json") as f:
    classe_indices = json.load(f)

classes = list(classe_indices.keys())

font = cv2.FONT_HERSHEY_SIMPLEX

predicoes = list()

#Passando as 4 imagens
carros = ["bugatti", "ferrari", "lamborghini", "porsche"]
for carro in carros:
    img = image.load_img(carro+"_car_valid.png", target_size=(224,224))
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    #print(pred)

    classe_predita = classes[np.argmax(pred)]
    #print("Classe prevista:", classe_predita)
    predicoes.append(classe_predita)

#mostrando as 4 imagens e colocando as predições
bugatti = cv2.imread("bugatti_car_valid.png", 1)
bugatti = cv2.resize(bugatti, (700, 300))
bugatti = cv2.putText(bugatti, predicoes[0], (10, 30), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

ferrari = cv2.imread("ferrari_car_valid.png", 1)
ferrari = cv2.resize(ferrari, (700, 300))
ferrari = cv2.putText(ferrari, predicoes[1], (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

lamborghini = cv2.imread("lamborghini_car_valid.png", 1)
lamborghini = cv2.resize(lamborghini, (700, 300))
lamborghini = cv2.putText(lamborghini, predicoes[2], (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

porsche = cv2.imread("porsche_car_valid.png", 1)
porsche = cv2.resize(porsche, (700, 300))
porsche = cv2.putText(porsche, predicoes[3], (10, 30), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

cv2.imshow('Bugatti', bugatti)
cv2.imshow('Ferrari', ferrari)
cv2.imshow('Lamborghini', lamborghini)
cv2.imshow('Porsche', porsche)

cv2.waitKey(0)
cv2.destroyAllWindows()