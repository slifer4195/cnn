import cv2
import tensorflow as tf

CATEGORIES = ["Dog", "Cat"]

def prepare(filepath):

    IMG_SIZE = 50
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE,1)

model= tf.keras.models.load_model("64x2-CNN.model")

prediction1 = model.predict(prepare('testDog.jpg'))
prediction2 = model.predict(prepare('testCat.jpg'))

animal_1 = CATEGORIES[int(prediction1[0][0])]
animal_2 = CATEGORIES[int(prediction2[0][0])]

print(animal_1, animal_2)