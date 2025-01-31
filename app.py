import numpy as np 
import pandas as pd 
import os
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import Rescaling, RandomZoom, RandomFlip, RandomRotation
from sklearn.model_selection import train_test_split
import os
from distutils.dir_util import copy_tree, remove_tree
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
base_dir = "AlzheimerDataset/"
root_dir = "./"
test_dir = base_dir + "test/"
train_dir = base_dir + "train/"
work_dir = root_dir + "dataset/"

if os.path.exists(work_dir):
    remove_tree(work_dir)
    

os.mkdir(work_dir)
copy_tree(train_dir, work_dir)
copy_tree(test_dir, work_dir)
print("Working Directory Contents:", os.listdir(work_dir))

WORK_DIR = './dataset/'
os.listdir(WORK_DIR)
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Device:', tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()
print('Number of replicas:', strategy.num_replicas_in_sync)
    
print(tf.__version__)# Constants
IMG_SIZE = 176
IMAGE_SIZE = [176, 176]
DIM = (IMG_SIZE, IMG_SIZE)
# Create the data augmentation pipeline
data_augmentation = tf.keras.Sequential([
    Rescaling(1./255),
    RandomZoom(0.2),
    RandomFlip("horizontal"),
    RandomRotation(0.2)
])
# Load images and apply data augmentation
train_data_gen = tf.keras.preprocessing.image_dataset_from_directory(
    directory=WORK_DIR,
    image_size=DIM,
    batch_size=64,
    shuffle=False,
    seed=42,
    validation_split=None,
    subset=None,
    labels="inferred"
)
train_data_augmented = train_data_gen.map(lambda x, y: (data_augmentation(x, training=True), y))
train_data = []
train_labels = []

for data, labels in train_data_augmented:
    train_data.append(data)
    train_labels.append(labels)

train_data = tf.concat(train_data, axis=0)
train_labels = tf.concat(train_labels, axis=0)
num_samples = train_data.shape[0]
train_data_2d = tf.reshape(train_data, (num_samples, -1))
#  perform over-sampling using the Synthetic Minority Over-sampling Technique for Image Data (SMOTE-IMG)
smote_img = SMOTE(sampling_strategy='minority')
train_data_resampled, train_labels_resampled = smote_img.fit_resample(train_data_2d, train_labels)
# Reshape train_data_resampled back to 4D  //4D -> Batch Size , Height , Width , Color Channels
train_data_resampled = train_data_resampled.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
# Split the augmented data into train and test sets
train_data1, test_data1, train_labels1, test_labels1 = train_test_split(train_data_resampled, train_labels_resampled, test_size=0.2, random_state=42,stratify=train_labels_resampled)
# Further split the train set into train and validation sets
train_data1, val_data1, train_labels1, val_labels1 = train_test_split(train_data1, train_labels1, test_size=0.2, random_state=42,stratify=train_labels1)

## --> Stratify == preserves the class distribution in the train and test split
from keras.utils import to_categorical

train_labels1 = to_categorical(train_labels1)
val_labels1 = to_categorical(val_labels1)
def conv_block(filters):
    block = tf.keras.Sequential([
        tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),
        tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D()
    ]
    )
    
    return block


def dense_block(units, dropout_rate):
    block = tf.keras.Sequential([
        tf.keras.layers.Dense(units, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate)
    ])
    
    return block

def build_model():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(IMAGE_SIZE[0],IMAGE_SIZE[1], 3)),
        
        tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D(),
        
        conv_block(32),
        conv_block(64),
        
        conv_block(128),
        tf.keras.layers.Dropout(0.2),
        
        conv_block(256),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Flatten(),
        dense_block(512, 0.7),
        dense_block(128, 0.5),
        dense_block(64, 0.3),
        
        tf.keras.layers.Dense(4,activation='softmax')
    ])
    
    return model


with strategy.scope():
    model = build_model()

    METRICS = [tf.keras.metrics.AUC(name='auc')]
    
    model.compile(
        optimizer='adam',
        loss=tf.losses.CategoricalCrossentropy()
        ,
        metrics=METRICS
    )
    
    
    model.summary()
    def exponential_decay(lr0, s):
        def exponential_decay_fn(epoch):
            return lr0 * 0.1 **(epoch / s)
        return exponential_decay_fn
    
exponential_decay_fn = exponential_decay(0.01, 20)
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("alzheimer_model.h5",
                                                    save_best_only=True)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10,
                                                     restore_best_weights=True)

history = model.fit(train_data1,train_labels1,
    validation_data=(val_data1, val_labels1),
    callbacks=[checkpoint_cb, early_stopping_cb, lr_scheduler],
    epochs=20
)

import tensorflow as tf
from keras.models import load_model

model = load_model("alzheimer_model.h5" , compile=False)

acc = history.history['auc']
val_acc = history.history['val_auc']

loss = history.history['loss']
val_loss = history.history['val_loss']

import matplotlib.pyplot as plt
EPOCHS = 20
plt.figure(figsize=(15, 4))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), acc, label='Training Accuracy')
plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), loss, label='Training Loss')
plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()



