# -- coding: utf-8 --
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import os
import glob
import numpy as np

CATEGORIES = [
    u'a',
    u'i',
    u'u',
    u'e',
    u'o',
    u'ka',
    u'ki',
    u'ku',
    u'ke',
    u'ko']

# 画像サイズ
IMG_SIZE = 150

# ImageDataGeneratorを定義
DATA_GENERATOR = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=0.3, zoom_range=0.1)

#画像をまとめているファイルまでのパス
IMG_ROOT_DIR = '../data/images'

for idx, category in enumerate(CATEGORIES):  #idxにはインデックス番号、categoryには'16Tea'などが入る
    # コピー元
    img_dir = os.path.join(IMG_ROOT_DIR, 'original', category)
    # コピー先
    out_dir = os.path.join(IMG_ROOT_DIR, 'extended', category)
    os.makedirs(out_dir, exist_ok=True)

    files = glob.glob(os.path.join(img_dir, '*.JPG'))
    for i, file in enumerate(files):
        img = keras.preprocessing.image.load_img(file)
        img = img.resize((IMG_SIZE, IMG_SIZE))
        x = keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        g = DATA_GENERATOR.flow(x, batch_size=1, save_to_dir=out_dir, save_prefix='img', save_format='jpg')
        for i in range(10):
            batch = g.next()
    print(u'{} : ファイル数は {} 件です。'.format(category, len(os.listdir(out_dir))))

