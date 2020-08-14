from tensorflow.keras.preprocessing.image import ImageDataGenerator

# import tensorflow as tf

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
)

train_generator = train_datagen.flow_from_directory(
    "dataset",
    target_size=(256, 256),
    batch_size=32,
    class_mode="binary",
    subset="training",
)

validation_generator = train_datagen.flow_from_directory(
    "dataset",
    target_size=(256, 256),
    batch_size=32,
    class_mode="binary",
    subset="validation",
)

# test_datagen = ImageDataGenerator(rescale=1.0 / 255)
# validation_generator = test_datagen.flow_from_directory(
#     "data/validation", target_size=(150, 150), batch_size=32, class_mode="binary"
# )
print(train_generator[:1].shape)
# print(len(train_generator))
# print(train_generator[0][0].shape)
# print(train_generator[0][1].shape)
# print(train_generator.class_indices)
# print(dir(train_generator))

# you can invert class keys and values by
# inverted = {v: k for k, v in dictionary.items()}

# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     "dataset",
#     validation_split=0.2,
#     subset="training",
#     seed=123,
#     image_size=(256, 256),
#     batch_size=32,
# )

# print(train_ds.class_names)
