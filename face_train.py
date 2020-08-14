from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.utils.version_utils import callbacks


# arguments
checkpoint_path = "checkpoints/training_1/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
Path(checkpoint_dir).mkdir(exist_ok=True, parents=True)

model_save_path = Path("saved_models/final_model")
model_save_path.mkdir(exist_ok=True, parents=True)

epochs = 15

# load using keras preprocessing
batch_size = 32
img_height = 256
img_width = 256

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
)

# using a training set of 80% and the rest for validation
train_generator = train_datagen.flow_from_directory(
    "dataset",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="binary",
    subset="training",
)

validation_generator = train_datagen.flow_from_directory(
    "dataset",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="binary",
    subset="validation",
)

print("Class Indices: ", train_generator.class_indices)


# # visualize dataset samples
# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(class_names[labels[i]])
#         plt.axis("off")
# plt.tight_layout()
# plt.show()

# # data augmentation
# data_augmentation = keras.Sequential(
#     [
#         layers.experimental.preprocessing.RandomFlip(
#             "horizontal", input_shape=(img_height, img_width, 3)
#         ),
#         layers.experimental.preprocessing.RandomRotation(0.1),
#         layers.experimental.preprocessing.RandomZoom(0.1),
#     ]
# )

# print("\nData augmentation layer done!")

# visualize augmentation
# plt.figure(figsize=(10, 10))
# for images, _ in train_generator:
#     print("images.shape: ", images.shape)
#     print("images.max: ", images.max())
#     for i in range(9):
#         augmented_images = data_augmentation(images)
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(augmented_images[0].numpy())
#         plt.axis("off")
#     break  # do once
# plt.show()


num_classes = len(train_generator.class_indices)
print("Number of classes: ", num_classes)

# create the model
model = Sequential(
    [
        # data_augmentation,
        layers.Conv2D(
            16,
            3,
            padding="same",
            activation="relu",
            input_shape=(img_height, img_width, 3),  # need to be in the first layer
        ),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes),
    ]
)

# Loads the weights
try:
    model.load_weights(checkpoint_path)
except Exception as e:
    print(e)
    pass

# # Re-evaluate the model
# loss,acc = model.evaluate(test_images,  test_labels, verbose=2)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.summary()

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, save_weights_only=True, verbose=1, save_best_only=True
)

# train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs,
    callbacks=[cp_callback],
)

# load weights for latest best model and save in one file
latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)
model.load_weights(latest)
model.save(model_save_path)

# visualize training results
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")
plt.show()


# # for inference
# image_path = 'path to images'

# pred_labels = [...] # list of image names

# pred_generator = DataGenerator(pred_idx, pred_labels, image_path, to_fit=False)
# pred = model.predict_generator(pred_generator)
