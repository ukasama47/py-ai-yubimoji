# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import sys
import os

# モデル保存先
MODEL_ROOT_DIR = '/../data/model/'
MODEL_PATH = os.path.join(MODEL_ROOT_DIR, 'model_predict.json')
WEIGHT_PATH = os.path.join(MODEL_ROOT_DIR, 'model_predict.hdf5')
# カテゴリ
CATEGORIES = [
    u'a',
    u'i',
    u'u',
    u'e',
    u'o',
]

CATEGORIES_NAME = [
    u'あ',
    u'い',
    u'う',
    u'え',
    u'お',
]

# 画像サイズ
IMG_SIZE = 150
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE,3)

# モデルを読み込む
model = keras.models.model_from_json(open(MODEL_PATH).read())
model.load_weights(WEIGHT_PATH)

# 入力引数から画像を読み込む
args = sys.argv
img = keras.preprocessing.image.load_img(args[1], target_size=INPUT_SHAPE)
x = keras.preprocessing.image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# モデルで予測する
features = model.predict(x)
print("確率：")
for i in range(0, 10):
    print(str(CATEGORIES_NAME[i]) + ' ： ' + str(features[0][i]))
print("----------------------------------------------")
print("計算結果")
if np.argmax(features[0]) == 1:
    print(u'選ばれたのは綾鷹でした。')
else:
    print(u'綾鷹ではなく' + str(CATEGORIES_NAME[np.argmax(features[0])]) + 'が選ばれました。')
print("----------------------------------------------")

