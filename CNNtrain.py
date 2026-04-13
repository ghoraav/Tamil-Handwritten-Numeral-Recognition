from pathlib import Path
import tensorflow as tf

DATASET_DIR = "Dataset"
IMAGE_SIZE = (264, 264)
BATCH_SIZE = 32
EPOCHS = 15
VALIDATION_SPLIT = 0.2
SEED = 42

tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()


def numeric_sort_key(name: str):
    return int(name) if name.isdigit() else name


dataset_path = Path(DATASET_DIR)
class_names = sorted(
    [folder.name for folder in dataset_path.iterdir() if folder.is_dir()],
    key=numeric_sort_key,
)

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    class_names=class_names,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    class_names=class_names,
)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.02, seed=SEED),
    tf.keras.layers.RandomTranslation(0.03, 0.03, seed=SEED + 1),
])


model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(264, 264, 3)),
    data_augmentation,
    tf.keras.layers.Rescaling(1.0 / 255),
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(class_names), activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# early_stopping = tf.keras.callbacks.EarlyStopping(
#     monitor="val_loss",
#     patience=3,
#     restore_best_weights=True
# )

model.summary()
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

model.save("tamil_character_cnn.keras")
print("Classes:", class_names)
print("Model saved as tamil_character_cnn.keras")