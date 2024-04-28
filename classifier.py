import os
from pathlib import Path
from keras.preprocessing import image
import matplotlib as plt
import csv
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import os
from tensorflow.keras.models import load_model
import numpy as np

def csv_to_dict(csv_file, key_column, value_column):
    result_dict = {}
    with open(csv_file, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            key = row[key_column]
            value = int(row[value_column])
            result_dict[key] = value
    return result_dict

csv_file = 'pokemon.csv'  # Replace 'your_file.csv' with the path to your CSV file
key_column = 'Name'  # Specify the column name for keys
value_column = '#'  # Specify the column name for values
label_dict = csv_to_dict(csv_file, key_column, value_column)

# Load saved model
loaded_model = load_model('pokemon_gen1_classifier.h5')

# Define a function to preprocess images
def preprocess_image(img_path, target_size=(150, 150)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values
    return img_array

# Example usage: Load and preprocess an unseen image
unseen_image_path = input('Enter Image Path:')
preprocessed_image = preprocess_image(unseen_image_path)

path = './pokemon_gen1_dbs/'
new_labels = []

for folder_name in os.listdir(path):
    folder_path = os.path.join(path, folder_name)
    if os.path.isdir(folder_path):
        # Check if the folder name exists in label_dict
        if folder_name in label_dict:
            integer_value = label_dict[folder_name]
            new_labels.append(integer_value)
        else:
            print(f"Ignoring folder '{folder_name}' as it does not have a corresponding label in label_dict.")

print("Integer List:", new_labels)


index_mapping = {}
for i, label_index in enumerate(new_labels):
    index_mapping[i] = label_index

print(index_mapping)


def get_key_from_value(dictionary, target_value):
    for key, value in dictionary.items():
        if value == target_value:
            return key
    return None

# Make predictions on the preprocessed image
predictions = loaded_model.predict(preprocessed_image)

# Get the predicted class label
predicted_class = np.argmax(predictions)
predicted_label_value = index_mapping[predicted_class]

print(f"Predicted Class Index: {get_key_from_value(label_dict,predicted_label_value)}")