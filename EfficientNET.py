import random
from pathlib import Path

import tensorflow as tf

DATASET_DIR = "Dataset"
IMAGE_SIZE = (264, 264)
BATCH_SIZE = 32
EPOCHS = 30
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42
MODEL_PATH = "tamil_character_efficientnetb0.keras"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()


def numeric_sort_key(name: str):
    return int(name) if name.isdigit() else name


def load_class_names(dataset_dir: str):
    dataset_path = Path(dataset_dir)
    return sorted(
        [folder.name for folder in dataset_path.iterdir() if folder.is_dir()],
        key=numeric_sort_key,
    )


def load_samples(dataset_dir: str, class_names: list[str]):
    dataset_path = Path(dataset_dir)
    samples = []

    for label_index, class_name in enumerate(class_names):
        class_dir = dataset_path / class_name
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((str(image_path), label_index))

    random.Random(SEED).shuffle(samples)
    return samples


def split_samples(samples):
    total_samples = len(samples)
    train_end = int(total_samples * TRAIN_SPLIT)
    val_end = train_end + int(total_samples * VAL_SPLIT)

    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]
    return train_samples, val_samples, test_samples


def load_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    return image, label


def build_dataset(samples, training=False):
    image_paths = [path for path, _ in samples]
    labels = [label for _, label in samples]

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    if training:
        dataset = dataset.shuffle(len(samples), seed=SEED, reshuffle_each_iteration=True)

    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset


class_names = load_class_names(DATASET_DIR)
samples = load_samples(DATASET_DIR, class_names)
train_samples, val_samples, test_samples = split_samples(samples)
train_ds = build_dataset(train_samples, training=True)
val_ds = build_dataset(val_samples)
test_ds = build_dataset(test_samples)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.02, seed=SEED),
    tf.keras.layers.RandomTranslation(0.03, 0.03, seed=SEED + 1),
])

base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(264, 264, 3),
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(264, 264, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
test_loss, test_accuracy = model.evaluate(test_ds)

model.save(MODEL_PATH)
print("Classes:", class_names)
print("Train samples:", len(train_samples))
print("Validation samples:", len(val_samples))
print("Test samples:", len(test_samples))
print("Test accuracy:", test_accuracy)
print("Test loss:", test_loss)
print(f"Model saved as {MODEL_PATH}")
