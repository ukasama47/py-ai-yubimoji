# -*- coding: utf-8 -*-

from PIL import Image
import os, glob
import numpy as np
import random, math
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 画像が保存されているルートディレクトリのパス
IMG_ROOT_DIR = '../data/images/extended'
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
# 画像データ
X = []
# カテゴリデータ
Y = []
# 教師データ
X_TRAIN = []
Y_TRAIN = []
# テストデータ
X_TEST = []
Y_TEST = []
# データ保存先
TRAIN_TEST_DATA = '../data/train_test_data/data.npy'


# カテゴリごとに処理する
for idx, category in enumerate(CATEGORIES):
    # 各ラベルの画像ディレクトリ
    image_dir = os.path.join(IMG_ROOT_DIR, category)
    files = glob.glob(os.path.join(image_dir, '*.jpg'))
    for f in files:
        # 各画像をリサイズしてデータに変換する
        img = Image.open(f)
        img = img.convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))
        data = np.asarray(img)
        X.append(data)
        Y.append(idx)

X = np.array(X)
Y = np.array(Y)

# 正規化
X = X.astype('float32') /255
Y = keras.utils.to_categorical(Y, DENSE_SIZE)

# 教師データとテストデータを分ける
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, Y, test_size=0.20)
print("X_TRAIN shape:", X_TRAIN.shape)
print("X_TEST shape:", X_TEST.shape)
print("Y_TRAIN shape:", Y_TRAIN.shape)
print("Y_TEST shape:", Y_TEST.shape)



# # 教師／テストデータを保存する
# np.save(TRAIN_TEST_DATA, (X_TRAIN, X_TEST, Y_TRAIN, Y_TEST))
# print(u'教師／テストデータの作成が完了しました。: {}'.format(TRAIN_TEST_DATA))

# 教師／テストデータを保存する

# try:
#     data = [X_TRAIN, X_TEST, Y_TRAIN, Y_TEST]
#     np.savez(TRAIN_TEST_DATA, data)
#     print(u'教師／テストデータの作成が完了しました。: {}'.format(TRAIN_TEST_DATA))
# except Exception as e:
#     print("Error saving data:", e)

# 教師／テストデータを保存する
try:
    # 各データセットを個別に保存
    np.savez(TRAIN_TEST_DATA, X_TRAIN=X_TRAIN, X_TEST=X_TEST, Y_TRAIN=Y_TRAIN, Y_TEST=Y_TEST)
    print(u'教師／テストデータの作成が完了しました。: {}'.format(TRAIN_TEST_DATA))
except Exception as e:
    print("Error saving data:", e)
