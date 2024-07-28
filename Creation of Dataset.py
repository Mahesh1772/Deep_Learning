import os
import shutil
import random
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set up paths (using raw strings)
desktop_path = r'C:\Users\bsiva\Desktop\mahesh\Deep_Learning'
source_folder = os.path.join(desktop_path, 'Garbage classification')
destination_folder = r'C:\Users\bsiva\Desktop\mahesh\Deep_Learning\data'

# Create destination folder structure
for folder in ['data\\train', 'data\\test']:
    for class_name in ['cardboard', 'glass', 'metal', 'paper', 'plastic']:
        os.makedirs(os.path.join(destination_folder, folder, class_name), exist_ok=True)

# Function to augment images
def augment_images(image_path, num_augmented):
    img = Image.open(image_path)
    img_array = np.array(img)
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    img_array = img_array.reshape((1,) + img_array.shape)
    i = 0
    for batch in datagen.flow(img_array, batch_size=1):
        augmented_img = Image.fromarray(batch[0].astype('uint8'))
        augmented_img.save(f"{os.path.splitext(image_path)[0]}_aug_{i}.jpg")
        i += 1
        if i >= num_augmented:
            break

# Process each class
for class_name in ['cardboard', 'glass', 'metal', 'paper', 'plastic']:
    source_class_folder = os.path.join(source_folder, class_name)
    images = os.listdir(source_class_folder)
    num_images = len(images)
   
    # Augment if necessary
    if num_images < 600:
        num_to_augment = 600 - num_images
        for i in range(num_to_augment):
            img_to_augment = random.choice(images)
            augment_images(os.path.join(source_class_folder, img_to_augment), 1)
   
    # Get all images (original + augmented)
    all_images = os.listdir(source_class_folder)
    random.shuffle(all_images)
   
    # Split into train and test
    train_images = all_images[:480]  # 80% for training
    test_images = all_images[480:600]  # 20% for testing
   
    # Copy to destination folders
    for img in train_images:
        shutil.copy(os.path.join(source_class_folder, img),
                    os.path.join(destination_folder, 'data\\train', class_name))
   
    for img in test_images:
        shutil.copy(os.path.join(source_class_folder, img),
                    os.path.join(destination_folder, 'data\\test', class_name))

print("Dataset prepared successfully!")