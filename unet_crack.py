import re
import os
import os.path

import cv2
import matplotlib.pyplot as plt

import random
import numpy as np
from glob import glob
from tqdm import tqdm

from keras import optimizers, metrics
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, AveragePooling2D
from keras.models import Model
import keras.backend as K

import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"]="2,3"

def generator(data_folder, image_shape, batch_size):
    image_paths = glob(os.path.join(data_folder, '*.jpeg'))
    random.shuffle(image_paths)
    
    while 1: 
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                gt_image_file = image_file[:-5]+'.png'
                
                image = cv2.imread(image_file)
                gt_image = cv2.imread(gt_image_file, cv2.IMREAD_GRAYSCALE).astype(np.bool)
                gt_image = gt_image.reshape(*gt_image.shape, 1)
                # gt = np.concatenate((gt_image, np.invert(gt_image)), axis=2)
                gt = np.concatenate((np.invert(gt_image), gt_image), axis=2)
                
                images.append(image)
                gt_images.append(gt)
                
            X = np.array(images).astype(np.int32)
            y = np.array(gt_images).astype(np.int32)
            yield (X, y)
            
def f1 (y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    tp = K.sum(y_true_f * y_pred_f)
    fp = K.sum(K.clip(K.clip(y_pred_f+y_true_f,0,1)-y_true_f,0,1))
    fn = K.sum(K.clip(K.clip(y_pred_f+y_true_f,0,1)-y_pred_f,0,1))
    Pr = tp/(tp+fp)
    Re = tp/(tp+fn)
    f1t = 2*(Pr*Re)/(Pr+Re)
    
    return f1t

def iou (y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    tp = K.sum(y_true_f * y_pred_f) 
    fp = K.sum(K.clip(K.clip(y_pred_f+y_true_f,0,1)-y_true_f,0,1))
    fn = K.sum(K.clip(K.clip(y_pred_f+y_true_f,0,1)-y_pred_f,0,1))
    
    return tp / (tp+fp+fn)

def mcc (y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    tp = K.sum(y_true_f * y_pred_f) 
    # tn = K.sum(y_true_f + y_pred_f) 
    fp = K.sum(K.clip(K.clip(y_pred_f+y_true_f,0,1)-y_true_f,0,1))
    fn = K.sum(K.clip(K.clip(y_pred_f+y_true_f,0,1)-y_pred_f,0,1))
    
    return (tp*tn - fp*fn) / np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))

'''
Models
'''
def unetA(img_rows, img_cols, img_channels):
    x = Input(shape=(img_rows, img_cols, img_channels))

    # Encoder 
    conv1 = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    conv2 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), padding='same', activation='relu')(pool1)
    conv4 = Conv2D(128, (3, 3), padding='same', activation='relu')(conv3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(256, (3, 3), padding='same', activation='relu')(pool2)
    conv6 = Conv2D(256, (3, 3), padding='same', activation='relu')(conv5)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv6)

    conv7 = Conv2D(512, (3, 3), padding='same', activation='relu')(pool3)
    conv8 = Conv2D(512, (3, 3), padding='same', activation='relu')(conv7)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv8)

    conv9 = Conv2D(1024, (3, 3), padding='same', activation='relu')(pool4)
    conv10 = Conv2D(1024, (3, 3), padding='same', activation='relu')(conv9)

    # Decoder
    convT1 = Conv2D(512, (2, 2), padding='same', activation='relu')(UpSampling2D(size=(2, 2))(conv10))
    merge1 = concatenate([convT1, conv8], axis=3)
    conv11 = Conv2D(512, (3, 3), padding='same', activation='relu')(merge1)
    conv12 = Conv2D(512, (3, 3), padding='same', activation='relu')(conv11)

    convT2 = Conv2D(256, (2, 2), padding='same', activation='relu')(UpSampling2D(size=(2, 2))(conv12))
    merge2 = concatenate([convT2, conv6], axis=3)
    conv13 = Conv2D(256, (3, 3), padding='same', activation='relu')(merge2)
    conv14 = Conv2D(256, (3, 3), padding='same', activation='relu')(conv13)

    convT3 = Conv2D(128, (2, 2), padding='same', activation='relu')(UpSampling2D(size=(2, 2))(conv14))
    merge3 = concatenate([convT3, conv4], axis=3)
    conv15 = Conv2D(128, (3, 3), padding='same', activation='relu')(merge3)
    conv16 = Conv2D(128, (3, 3), padding='same', activation='relu')(conv15)

    convT4 = Conv2D(64, (2, 2), padding='same', activation='relu')(UpSampling2D(size=(2, 2))(conv16))
    merge4 = concatenate([convT4, conv2], axis=3)
    conv17 = Conv2D(64, (3, 3), padding='same', activation='relu')(merge4)
    conv18 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv17)

    y = Conv2D(2, (1, 1), activation='softmax')(conv18)

    return Model(inputs=x, outputs=y)

def unetB(img_rows, img_cols, img_channels):
    x = Input(shape=(img_rows, img_cols, img_channels))

    # Encoder 
    conv1 = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    conv2 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), padding='same', activation='relu')(pool1)
    conv4 = Conv2D(128, (3, 3), padding='same', activation='relu')(conv3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(256, (3, 3), padding='same', activation='relu')(pool2)
    conv6 = Conv2D(256, (3, 3), padding='same', activation='relu')(conv5)

    # Decoder  
    convT3 = Conv2D(128, (2, 2), padding='same', activation='relu')(UpSampling2D(size=(2, 2))(conv6))
    merge3 = concatenate([convT3, conv4], axis=3)
    conv15 = Conv2D(128, (3, 3), padding='same', activation='relu')(merge3)
    conv16 = Conv2D(128, (3, 3), padding='same', activation='relu')(conv15)

    convT4 = Conv2D(64, (2, 2), padding='same', activation='relu')(UpSampling2D(size=(2, 2))(conv16))
    merge4 = concatenate([convT4, conv2], axis=3)
    conv17 = Conv2D(64, (3, 3), padding='same', activation='relu')(merge4)
    conv18 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv17)

    # Segmentation
    y = Conv2D(2, (1, 1), activation='softmax')(conv18)

    return Model(inputs=x, outputs=y)

def unetC(img_rows, img_cols, img_channels):  #lr = 0.0009
    x = Input(shape=(img_rows, img_cols, img_channels))

    # Encoder 
    conv1 = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    conv2 = Conv2D(64, (1, 1), padding='same', activation='relu')(conv1)
    pool = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), padding='same', activation='relu')(pool)
    conv4 = Conv2D(128, (1, 1), padding='same', activation='relu')(conv3)

    # Decoder   
    convT = Conv2D(64, (2, 2), padding='same', activation='relu')(UpSampling2D(size=(2, 2))(conv4))
    merge = concatenate([convT, conv2], axis=3)
    conv5 = Conv2D(64, (3, 3), padding='same', activation='relu')(merge)
    conv6 = Conv2D(64, (1, 1), padding='same', activation='relu')(conv5)

    # Segmentation
    y = Conv2D(2, (1, 1), activation='softmax')(conv6)

    return Model(inputs=x, outputs=y)

def unet256(img_rows, img_cols, img_channels):
    x = Input(shape=(img_rows, img_cols, img_channels))

    # Encoder 
    conv1 = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    conv2 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv1)
    pool1 = AveragePooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), padding='same', activation='relu')(pool1)
    conv4 = Conv2D(128, (3, 3), padding='same', activation='relu')(conv3)
    pool2 = AveragePooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(256, (3, 3), padding='same', activation='relu')(pool2)
    conv6 = Conv2D(256, (3, 3), padding='same', activation='relu')(conv5)
    pool3 = AveragePooling2D(pool_size=(2, 2))(conv6)

    conv7 = Conv2D(512, (3, 3), padding='same', activation='relu')(pool3)
    conv8 = Conv2D(512, (3, 3), padding='same', activation='relu')(conv7)

    # Decoder
    convT1 = Conv2D(256, (2, 2), padding='same', activation='relu')(UpSampling2D(size=(2, 2))(conv8))
    merge1 = concatenate([convT1, conv6], axis=3)
    conv9 = Conv2D(256, (3, 3), padding='same', activation='relu')(merge1)
    conv10 = Conv2D(256, (3, 3), padding='same', activation='relu')(conv9)

    convT2 = Conv2D(128, (2, 2), padding='same', activation='relu')(UpSampling2D(size=(2, 2))(conv10))
    merge2 = concatenate([convT2, conv4], axis=3)
    conv11 = Conv2D(128, (3, 3), padding='same', activation='relu')(merge2)
    conv12 = Conv2D(128, (3, 3), padding='same', activation='relu')(conv11)

    convT3 = Conv2D(64, (2, 2), padding='same', activation='relu')(UpSampling2D(size=(2, 2))(conv12))
    merge3 = concatenate([convT3, conv2], axis=3)
    conv13 = Conv2D(64, (3, 3), padding='same', activation='relu')(merge3)
    conv14 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv13)

    y = Conv2D(2, (1, 1), activation='softmax')(conv14)

    return Model(inputs=x, outputs=y)


def train(image_shape, data_dir, logs_dir, namefile, batch_size, params):    
    train_generator = generator(os.path.join(data_dir, 'train/'), image_shape, batch_size)
    validation_generator = generator(os.path.join(data_dir, 'val/'), image_shape, batch_size)
    test_generator = generator(os.path.join(data_dir, 'test/'), image_shape, batch_size)

    model = unetA(image_shape[0],image_shape[1],3)
    adam = optimizers.Adam(lr=params["lr"], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01) # lr = 0.0009
    model.compile(loss='categorical_crossentropy', 
                  optimizer=adam, 
                  metrics=[f1])  # metrics: f1, iou
    checkpointer = ModelCheckpoint(filepath='bestmodel_' + namefile + '.h5', 
                               monitor='val_loss', 
                               verbose=0, 
                               save_best_only=True,
                               mode='min', 
                               period=1)
    tb_callback = TensorBoard(log_dir=logs_dir, 
                              write_graph=True, 
                              update_freq="epoch")
    es_callback = EarlyStopping(monitor='val_loss', patience=3)
    
    h = model.fit_generator(generator = train_generator, 
                        steps_per_epoch=params["spe"],
                        epochs=params["ep"], 
                        verbose=1, 
                        callbacks=[checkpointer, tb_callback, es_callback], 
                        validation_data=validation_generator, 
                        validation_steps=params["vs"])

def main():    
    image_shape = (512,512)
    data_dir =  ''
    logs_dir = './logs/'
    
    umodel = 'A'
    isize = image_shape[0]
    lmodel = 'f1'
    params = {"lr":0.0001, "ep":30, "spe":10, "vs":1, "bs":5}
    namefile = namefile = f'unet_{umodel}_demo{isize}_2_{lmodel}_lr{params["lr"]}_ep{params["ep"]}_bs{params["bs"]}_spe{params["spe"]}_vs{params["vs"]}'

    train(image_shape, data_dir, logs_dir, namefile, params["bs"], params=param)

if __name__ == "__main__":
    main()