"""
---------- ENUNCIADO ----------

For this challenge, you will complete the code to classify images of dogs and cats. You will use TensorFlow 2.0 and Keras to create a convolutional neural network that correctly classifies images of cats and dogs at least 63% of the time. (Extra credit if you get it to 70% accuracy!)

Some of the code is given to you but some code you must fill in to complete this challenge. Read the instruction in each text cell so you will know what you have to do in each code cell.

The first code cell imports the required libraries. The second code cell downloads the data and sets key variables. The third cell is the first place you will write your own code.

The structure of the dataset files that are downloaded looks like this (You will notice that the test directory has no subdirectories and the images are not labeled):

cats_and_dogs
|__ train:
    |______ cats: [cat.0.jpg, cat.1.jpg ...]
    |______ dogs: [dog.0.jpg, dog.1.jpg ...]
|__ validation:
    |______ cats: [cat.2000.jpg, cat.2001.jpg ...]
    |______ dogs: [dog.2000.jpg, dog.2001.jpg ...]
|__ test: [1.jpg, 2.jpg ...]
You can tweak epochs and batch size if you like, but it is not required.

The following instructions correspond to specific cell numbers, indicated with a comment at the top of the cell (such as # 3).

                                                                                                                 
Cell 3
Now it is your turn! Set each of the variables in this cell correctly. (They should no longer equal None.)

Create image generators for each of the three image data sets (train, validation, test). Use ImageDataGenerator to read / decode the images and convert them into floating point tensors. Use the rescale argument (and no other arguments for now) to rescale the tensors from values between 0 and 255 to values between 0 and 1.

For the *_data_gen variables, use the flow_from_directory method. Pass in the batch size, directory, target size ((IMG_HEIGHT, IMG_WIDTH)), class mode, and anything else required. test_data_gen will be the trickiest one. For test_data_gen, make sure to pass in shuffle=False to the flow_from_directory method. This will make sure the final predictions stay in the order that our test expects. For test_data_gen it will also be helpful to observe the directory structure.

After you run the code, the output should look like this:

Found 2000 images belonging to 2 classes.
Found 1000 images belonging to 2 classes.
Found 50 images belonging to 1 class.


Cell 4
The plotImages function will be used a few times to plot images. It takes an array of images and a probabilities list, although the probabilities list is optional. This code is given to you. If you created the train_data_gen variable correctly, then running this cell will plot five random training images.


Cell 5
Recreate the train_image_generator using ImageDataGenerator.

Since there are a small number of training examples, there is a risk of overfitting. One way to fix this problem is by creating more training data from existing training examples by using random transformations.

Add 4-6 random transformations as arguments to ImageDataGenerator. Make sure to rescale the same as before.


Cell 6
You don't have to do anything for this cell. train_data_gen is created just like before but with the new train_image_generator. Then, a single image is plotted five different times using different variations.


Cell 7
In this cell, create a model for the neural network that outputs class probabilities. It should use the Keras Sequential model. It will probably involve a stack of Conv2D and MaxPooling2D layers and then a fully connected layer on top that is activated by a ReLU activation function.

Compile the model passing the arguments to set the optimizer and loss. Also pass in metrics=['accuracy'] to view training and validation accuracy for each training epoch.


Cell 8
Use the fit method on your model to train the network. Make sure to pass in arguments for x, steps_per_epoch, epochs, validation_data, and validation_steps.


Cell 9
Run this cell to visualize the accuracy and loss of the model.


Cell 10
Now it is time to use your model to predict whether a brand new image is a cat or a dog.

In this cell, get the probability that each test image (from test_data_gen) is a dog or a cat. probabilities should be a list of integers.

Call the plotImages function and pass in the test images and the probabilities corresponding to each test image.

After you run the cell, you should see all 50 test images with a label showing the percentage of "sure" that the image is a cat or a dog. The accuracy will correspond to the accuracy shown in the graph above (after running the previous cell). More training images could lead to a higher accuracy.


Cell 11
Run this final cell to see if you passed the challenge or if you need to keep trying.
"""


import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

import zipfile
import requests

# Descargar el archivo ZIP
url = "https://cdn.freecodecamp.org/project-data/cats-and-dogs/cats_and_dogs.zip"
zip_file = "cats_and_dogs.zip"

print("Descargando archivo...")
response = requests.get(url)
with open(zip_file, "wb") as file:
    file.write(response.content)
print("Archivo descargado.")

# Descomprimir el archivo ZIP
print("Descomprimiendo el archivo...")
with zipfile.ZipFile(zip_file, "r") as zip_ref:
    zip_ref.extractall()
print("Descompresión completada.")

# Definir rutas
PATH = 'cats_and_dogs'
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
test_dir = os.path.join(PATH, 'test')

print(f"Train directory: {train_dir}")
print(f"Validation directory: {validation_dir}")
print(f"Test directory: {test_dir}")


train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
test_dir = os.path.join(PATH, 'test')

# Get number of files in each directory. The train and validation directories
# each have the subdirecories "dogs" and "cats".
total_train = sum([len(files) for r, d, files in os.walk(train_dir)])
total_val = sum([len(files) for r, d, files in os.walk(validation_dir)])
total_test = len(os.listdir(test_dir))


# Variables for pre-processing and training.
batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150


# 3
# Crea los generadores de imágenes con el rescale adecuado
train_image_generator = ImageDataGenerator(rescale=1./255)  # Reescalar las imágenes a 0-1
validation_image_generator = ImageDataGenerator(rescale=1./255)
test_image_generator = ImageDataGenerator(rescale=1./255)

# Crea los generadores de datos usando flow_from_directory
train_data_gen = train_image_generator.flow_from_directory(
    train_dir,
    batch_size=batch_size,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='binary'  # Las imágenes son de 2 clases: gatos y perros
)

val_data_gen = validation_image_generator.flow_from_directory(
    validation_dir,
    batch_size=batch_size,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='binary'
)

"""
test_data_gen = test_image_generator.flow_from_directory(
    test_dir,
    batch_size=batch_size,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode=None,  # No necesitamos las etiquetas para el test
    shuffle=False  # No barajar las imágenes para que las predicciones mantengan el mismo orden
)
"""

# Cambiar test_data_gen, ya que no tiene subdirectorios
import pandas as pd

# Lista las rutas de las imágenes en test_dir
test_image_paths = [os.path.join(test_dir, fname) for fname in os.listdir(test_dir) if fname.endswith('.jpg')]

# Crear un DataFrame con las rutas de las imágenes
test_df = pd.DataFrame({
    'filename': test_image_paths
})

# Crear el generador de imágenes para test usando flow_from_dataframe
test_data_gen = test_image_generator.flow_from_dataframe(
    dataframe=test_df,
    directory=None,  # Ya hemos incluido las rutas completas en 'filename'
    x_col='filename',  # Columna con las rutas de las imágenes
    y_col=None,  # No necesitamos las etiquetas para la prueba
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
    class_mode=None,  # No necesitamos las etiquetas para la prueba
    shuffle=False  # No barajar las imágenes para mantener el orden
)


# 4
def plotImages(images_arr, probabilities = False):
    fig, axes = plt.subplots(len(images_arr), 1, figsize=(5,len(images_arr) * 3))
    if probabilities is False:
      for img, ax in zip( images_arr, axes):
          ax.imshow(img)
          ax.axis('off')
    else:
      for img, probability, ax in zip( images_arr, probabilities, axes):
          ax.imshow(img)
          ax.axis('off')
          if probability > 0.5:
              ax.set_title("%.2f" % (probability*100) + "% dog")
          else:
              ax.set_title("%.2f" % ((1-probability)*100) + "% cat")
    plt.show()

sample_training_images, _ = next(train_data_gen)
plotImages(sample_training_images[:5])


# 5
train_image_generator = ImageDataGenerator(
  rescale=1./255,
  rotation_range=40, # Rotación aleatoria en un rango de -40° a 40°
  width_shift_range=0.2, # Desplazamiento horizontal y vertical
  height_shift_range=0.2, # aleatorio en un 20% del tamaño de la imagen
  shear_range=0.2, # Cizalladura (shearing) de la imagen, deformándola en un ángulo aleatorio dentro de un rango
  zoom_range=0.2, # Zoom aleatorio hasta un 20% (Zoom in / out)
  horizontal_flip=True, # Voltear horizontalmente la imagen de forma aleatoria
  fill_mode='nearest') # Al aplicar las transformaciones, los píxeles "vacíos" se rellenan con valores cercanos


# 6
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                     directory=train_dir,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='binary')

augmented_images = [train_data_gen[0][0][0] for i in range(5)]

plotImages(augmented_images)


# 7

"""
94% de las imágenes correctamente clasificadas
Aproximadamente 90 minutos en ejecutar

from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False  # Congelar las capas de base

model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])
  
----------------------------------------------

model = Sequential([
  Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
  MaxPooling2D(2, 2),
  Conv2D(64, (3, 3), activation='relu'),
  MaxPooling2D((2, 2)),
  Conv2D(128, (3, 3), activation='relu'),
  MaxPooling2D(2,2),
  Flatten(),
  Dense(128, activation='relu'),
  Dropout(0.5),
  Dense(1, activation = 'sigmoid')
])
"""

from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False  # Congelar las capas de base

model = models.Sequential()

# Añadir el modelo base (VGG16) sin las capas superiores
model.add(base_model)

# Añadir capas adicionales con padding='same' para evitar reducción de dimensiones
model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))  # Nueva capa de convolución con padding
model.add(layers.MaxPooling2D(2, 2))  # Nueva capa de max-pooling
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))  # Otra capa de convolución
model.add(layers.MaxPooling2D(2, 2))  # Otra capa de max-pooling

# Aplanar la salida para conectarla a capas densas
model.add(layers.Flatten())

# Añadir una capa densa
model.add(layers.Dense(256, activation='relu'))

# Capa de salida para clasificación binaria
model.add(layers.Dense(1, activation='sigmoid'))


model.summary()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# 8
steps_per_epoch = total_train // batch_size
validation_steps = total_val // batch_size

history = model.fit(
    train_data_gen,
    verbose=1,
    epochs=15,
    validation_data=val_data_gen
)


# 9
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# 10
# Realizar predicciones con el modelo
steps = len(test_data_gen)  # Asegúrate de usar el número correcto de pasos
predictions = model.predict(
    test_data_gen,
    steps=steps,
    verbose=1
)

# Aplanar las predicciones para su visualización
probabilities = predictions.flatten()

# Obtener las imágenes del generador de datos
test_images = next(test_data_gen)[0]

# Verificar que el número de imágenes y probabilidades coincidan
print(f"Cantidad de imágenes: {len(test_images)}")
print(f"Cantidad de probabilidades: {len(probabilities)}")

# Función para graficar las imágenes y sus probabilidades
def plotImages(images_arr, probabilities):
    fig, axes = plt.subplots(1, len(images_arr), figsize=(15, 15))
    for img, probability, ax in zip(images_arr, probabilities, axes):
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"Probabilidad: {probability:.2f}")
    plt.show()

# Mostrar las primeras imágenes con sus probabilidades
plotImages(test_images[:len(probabilities)], probabilities)


# 11
# Prueba del código
answers =  [1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0,
            1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0,
            1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1,
            1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 
            0, 0, 0, 0, 0, 0]

correct = 0

for probability, answer in zip(probabilities, answers):
  if round(probability) == answer:
    correct +=1

percentage_identified = (correct / len(answers)) * 100

passed_challenge = percentage_identified >= 63

print(f"Your model correctly identified {round(percentage_identified, 2)}% of the images of cats and dogs.")

if passed_challenge:
  print("You passed the challenge!")
else:
  print("You haven't passed yet. Your model should identify at least 63% of the images. Keep trying. You will get it!")