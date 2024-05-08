# -*- coding: utf-8 -*-

#モデルの構築
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import matplotlib.pyplot as plt
import os

# カテゴリ
CATEGORIES = [
    u'a',
    u'i',
    u'u',
    u'e',
    u'o',
]

# 密度
DENSE_SIZE = len(CATEGORIES)
# 画像サイズ
IMG_SIZE = 150
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE,3)
# 教師データ
X_TRAIN = []
Y_TRAIN = []
# テストデータ
X_TEST = []
Y_TEST = []
# データ保存先
TRAIN_TEST_DATA = '/../AyatakaAI_py/data/train_test_data/data.npy'
# モデル保存先
MODEL_ROOT_DIR = '/../AyatakaAI_py/data/model/'


# ----- モデル構築 ----- #
model = keras.models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation="relu",input_shape=INPUT_SHAPE))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512,activation="relu"))
model.add(layers.Dense(DENSE_SIZE,activation="sigmoid"))

#モデル構成の確認
model.summary()
# ----- /モデル構築 ----- #

# ----- モデルコンパイル ----- #
model.compile(loss="binary_crossentropy",
              optimizer=keras.optimizers.RMSprop(lr=1e-4),
              metrics=["acc"])
# ----- /モデル構築 ----- #

# ----- モデル学習 ----- #
# 教師データとテストデータを読み込む
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = np.load(TRAIN_TEST_DATA, allow_pickle=True)
model = model.fit(X_TRAIN,
                  Y_TRAIN,
                  epochs=10,
                  batch_size=6,
                  validation_data=(X_TEST, Y_TEST))
# ----- /モデル学習 ----- #

# ----- 学習結果プロット ----- #
acc = model.history['acc']
val_acc = model.history['val_acc']
loss = model.history['loss']
val_loss = model.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig(os.path.join(MODEL_ROOT_DIR, 'Training_and_validation_accuracy.png'))

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig(os.path.join(MODEL_ROOT_DIR, 'Training_and_validation_loss.png'))
# ----- /学習結果プロット ----- #

# ----- モデル保存 ----- #
# モデル保存
json_string = model.model.to_json()
open(os.path.join(MODEL_ROOT_DIR, 'model_predict.json'), 'w').write(json_string)

#重み保存
model.model.save_weights(os.path.join(MODEL_ROOT_DIR, 'model_predict.hdf5'))
# ----- /モデル保存 ----- #

