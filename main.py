# =============================
# Brain Tumor Classification
# Modified to run in VS Code (Local PC)
# TensorFlow 2.9.1
# =============================

# --------- IMPORT LIBRARIES ---------
from tensorflow.keras.preprocessing import image
import os
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.callbacks import EarlyStopping

print("All modules loaded successfully")

# --------- DATASET PATHS (LOCAL) ---------
train_data_dir = "dataset/Training"
test_data_dir  = "dataset/Testing"

print("\n===== Brain Tumor Classification =====")
print("1. Train Model")
print("2. Test Image (No Training)")
print("3. Exit")

choice = input("Enter your choice (1/2/3): ")


# --------- TRAIN MODEL ---------
if choice == "1":
    print("\nTraining model...\n")

    # --------- CREATE TEST DATAFRAME ---------
    filepaths = []
    labels = []

    for folder in os.listdir(train_data_dir):
        folder_path = os.path.join(train_data_dir, folder)
        for file in os.listdir(folder_path):
            filepaths.append(os.path.join(folder_path, file))
            labels.append(folder)

    train_df = pd.DataFrame({"filepaths": filepaths, "labels": labels})
    # --------- CREATE TEST DATAFRAME ---------
    filepaths = []
    labels = []

    for folder in os.listdir(test_data_dir):
        folder_path = os.path.join(test_data_dir, folder)
        for file in os.listdir(folder_path):
            filepaths.append(os.path.join(folder_path, file))
            labels.append(folder)

    ts_df = pd.DataFrame({"filepaths": filepaths, "labels": labels})
    
    # --------- SPLIT VALIDATION & TEST ---------
    valid_df, test_df = train_test_split(ts_df, train_size=0.5, shuffle=True, random_state=123)

    # --------- IMAGE GENERATORS ---------
    batch_size = 16
    img_size = (224, 224)

    train_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
        train_df,
        x_col='filepaths',
        y_col='labels',
        target_size=img_size,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True
    )

    valid_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
        valid_df,
        x_col='filepaths',
        y_col='labels',
        target_size=img_size,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True
    )

    test_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
        test_df,
        x_col='filepaths',
        y_col='labels',
        target_size=img_size,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=False
    )

    # --------- CNN MODEL ---------
    num_classes = len(train_gen.class_indices)
    # --------- SAVE CLASS NAMES (FOR TESTING WITHOUT TRAINING) ---------
    class_names = list(train_gen.class_indices.keys())
    np.save("class_names.npy", class_names)


    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),

        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),

        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),

        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adamax(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()


    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(
        train_gen,
        epochs=10,
        validation_data=valid_gen,
        callbacks=[early_stop]
    )

    model.save('Brain_Tumors.h5')
    print("Model saved as Brain_Tumors.h5")

    # --------- PLOT ACCURACY & LOSS ---------
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss')

    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.legend()
    plt.title('Accuracy')
    plt.show()

    # --------- EVALUATION ---------
    train_score = model.evaluate(train_gen, verbose=0)
    valid_score = model.evaluate(valid_gen, verbose=0)
    test_score  = model.evaluate(test_gen, verbose=0)

    print("Train Accuracy:", train_score[1])
    print("Validation Accuracy:", valid_score[1])
    print("Test Accuracy:", test_score[1])

    # --------- CONFUSION MATRIX ---------
    y_pred = np.argmax(model.predict(test_gen), axis=1)
    cm = confusion_matrix(test_gen.classes, y_pred)

    plt.figure(figsize=(6,6))
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.show()
    print(classification_report(test_gen.classes, y_pred, target_names=list(test_gen.class_indices.keys())))

    # --------- SAVE MODEL ---------
    model.save('Brain_Tumors.h5')
    print("Model saved as Brain_Tumors.h5")

# --------- LOAD & PREDICT ONE IMAGE ---------

elif choice == "2":

    print("\nTesting image without training...\n")
    if not os.path.exists("Brain_Tumors.h5"):
        print("❌ Model not found. Train the model first (Option 1).")
        exit()
    if not os.path.exists("class_names.npy"):
        print("❌ Class names file not found. Train the model first (Option 1).")
        exit()

    model = tf.keras.models.load_model("Brain_Tumors.h5")
    class_names = np.load("class_names.npy", allow_pickle=True)

    img_path = input("Enter image path: ")
    if not os.path.exists(img_path):
        print("❌ Image path not found")
        exit()


    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    confidence = np.max(prediction) * 100

    predicted_class = class_names[np.argmax(prediction)]

    plt.imshow(img)
    plt.title(f"{predicted_class} ({confidence:.2f}%)")
    plt.axis("off")
    plt.show()

    print("Predicted Class:", predicted_class)

    # loaded_model.compile(Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
elif choice == "3":
    print("Exiting program")
    exit()
