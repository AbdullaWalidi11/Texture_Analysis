import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

CSV_PATH = "training_data/labels.csv"
IMG_DIR = "training_data/images"
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15

print("Loading dataset metadata...")
df = pd.read_csv(CSV_PATH)

df['filepath'] = df['filename'].apply(lambda x: os.path.join(IMG_DIR, x))

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"Training on {len(train_df)} images, Validating on {len(val_df)} images.")

def process_image(filepath, score):

    img = tf.io.read_file(filepath)

    img = tf.image.decode_image(img, channels=3, expand_animations=False)

    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return img, score

def create_dataset(dataframe):
    ds = tf.data.Dataset.from_tensor_slices((dataframe['filepath'].values, dataframe['roughness_score'].values))
    ds = ds.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_dataset = create_dataset(train_df)
val_dataset = create_dataset(val_df)

print("Building EfficientNetB0 Regression Model...")

base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False 

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)

outputs = tf.keras.layers.Dense(1, activation='linear')(x) 

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='mean_absolute_error', 
    metrics=['mean_absolute_error']
)

print("Starting Initial Warm-up Training (Head Only)...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=5
)

print("Unfreezing base model for fine-tuning...")
base_model.trainable = True

for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
    loss='mean_absolute_error',
    metrics=['mean_absolute_error']
)

history_finetune = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS
)

model.save("texture_regressor.keras")
print("Model saved successfully as 'texture_regressor.keras'!")
