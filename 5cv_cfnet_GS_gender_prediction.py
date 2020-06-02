from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Flatten, Input, Layer, MaxPooling2D, Conv2D, Reshape, GlobalAveragePooling2D
from keras.losses import binary_crossentropy
from keras import backend as K
from keras.optimizers import Adam, SGD

import tensorflow as tf
import numpy as np
import scipy.stats as st

import sys
import argparse
import os
import glob 

import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import scipy.ndimage
from scipy.misc import imsave
import scipy as sp

from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16

#from distcorr import distcorr

def augment_by_flipping(data,sc,dx,n):
    augment_scale = 1

    if n <= data.shape[0]:
        return data
    else:
        raw_n = data.shape[0]
        m = n - raw_n
        new_data = np.zeros((m,data.shape[1],data.shape[2],data.shape[3]))
        for i in range(0,m):
	    if (i < raw_n):
		idx = i
            else:
                idx = np.random.randint(0,raw_n)

            new_sc = sc[idx]
            new_dx = dx[idx]
            new_data[i] = data[idx].copy()
            new_data[i] = np.fliplr(new_data[i])
            new_data[i] = sp.ndimage.rotate(new_data[i],np.random.uniform(-4,4),reshape=False)
            
            #imsave('test.png',new_data[i])
            #imsave('raw.png',data[idx])
            sc = np.append(sc, new_sc)
            dx = np.append(dx, new_dx)
        data = np.concatenate((data, new_data), axis=0)
        return data,sc,dx

def inv_correlation_coefficient_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return 1 - K.square(r)

def correlation_coefficient_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return K.square(r)

def inv_entropy(y_true, y_pred):
    e = tf.multiply(pred, K.log(y_pred))

    return K.sum(K.sum(e))

class GAN():
    def build_regressor(self):
        inputs_x = Input(shape=(512,))
        feature = Dense(128, activation='tanh')(inputs_x)
        cf = Dense(1)(feature)

        return Model(inputs_x, cf)

    def __init__(self):

        # Build and compile the cf predictor
        self.regressor = self.build_regressor()
        self.regressor.compile(loss='mse', optimizer='adam')
	
        ## build initial model with fixed conv layers
        input_image = Input(shape=(224,224,3),name='input_image')

        self.base_model = VGG16(weights='imagenet',include_top=False)
        feature = self.base_model(input_image)
        feature = GlobalAveragePooling2D()(feature)

        self.encoder = Model(input_image, feature)

        # For the distillation model we will only train the encoder

        self.regressor.trainable = False
        cf = self.regressor(feature)
        self.distiller = Model(input_image, cf)
        self.distiller.compile(loss=correlation_coefficient_loss, optimizer=SGD(lr=0.0001,momentum=0.9))

        # classifier
        feature_clf = Dense(128, activation='tanh')(feature)
        prediction = Dense(1, activation='sigmoid')(feature_clf)
        self.workflow = Model(inputs=input_image, outputs=prediction)

    def train(self, epochs, training, testing,  batch_size=64, fold=0):
        [train_data_aug, train_sc_aug, train_dx_aug] = training
        [test_data,  test_sc,  test_dx]  = testing
        
        ## initialize with pretrained model
        for layer in self.base_model.layers:
            layer.trainable = False

        self.workflow.compile(loss='binary_crossentropy', optimizer='rmsprop',metrics=['accuracy'])
        self.workflow.summary()

        self.workflow.fit(train_data_aug,train_dx_aug,
                     validation_data=[test_data,test_dx],
                     epochs=15,
                     batch_size=64)

        ## initialize bias predictor
        encoded_feature_train = self.encoder.predict(train_data_aug)
        encoded_feature_test = self.encoder.predict(test_data)
        self.regressor.fit(encoded_feature_train,train_sc_aug,
                           validation_data=[encoded_feature_test,test_sc],
                           epochs=15,
                           batch_size=64)

        ## fine-tune everything with cf-net
        for layer in self.base_model.layers:
            layer.trainable = True
        self.workflow.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.0001,momentum=0.9),metrics=['accuracy'])
        self.workflow.summary()

        for epoch in range(epochs):

            # Select a random batch of images
            idx_perm = np.random.permutation(int(train_data_aug.shape[0]/2))
            idx = idx_perm[:int(batch_size/2)]
            idx = np.concatenate((idx,idx+int(train_data_aug.shape[0]/2)))

            training_feature_batch = train_data_aug[idx]
            dx_batch = train_dx_aug[idx]
            sc_batch = train_sc_aug[idx]

            # ---------------------
            #  Train regressor (cf predictor)
            # ---------------------

            encoded_feature_batch = self.encoder.predict(training_feature_batch)
            r_loss = self.regressor.train_on_batch(encoded_feature_batch, sc_batch)

            # ---------------------
            #  Train Disstiller
            # ---------------------           
            g_loss = self.distiller.train_on_batch(training_feature_batch, sc_batch)

            # ---------------------
            #  Train Encoder & Classifier
            # ---------------------

            c_loss = self.workflow.train_on_batch(training_feature_batch, dx_batch)


            # Plot the progress
            if epoch % 100 == 0:
                c_loss_test = self.workflow.evaluate(test_data, test_dx, verbose = 0, batch_size = batch_size)
            	print ("Epoch:%d [Training Acc: %f, Test Acc: %f]" % (epoch, c_loss[1], c_loss_test[1]))
                pred = self.workflow.predict(test_data);

                ## save prediction intermediate predictions
                filename = sys.argv[2]+'/pred_'+str(fold)+'_'+str(epoch)+'.txt'
                np.savetxt(filename,pred);
                ## save prediction intermediate features for posthoc dcor and MI analysis
                encoded_feature_test = self.encoder.predict(test_data, batch_size = batch_size)
                filename = sys.argv[2]+'/features_'+str(fold)+'_'+str(epoch)+'.txt'
                np.savetxt(filename,encoded_feature_test);
            
        ## save ground-truth values
        filename = sys.argv[2]+'/dx_'+str(fold)+'.txt'
        np.savetxt(filename,test_dx);
        filename = sys.argv[2]+'/sc_'+str(fold)+'.txt'
        np.savetxt(filename,test_sc);

if __name__ == '__main__':
    seed = int(sys.argv[1]) 

    img_names = glob.glob("./cropped/*.png")
    img_names_sorted = sorted(img_names);
    N = len(img_names_sorted)

    data = np.zeros((N,224,224,3))
    i = 0
    for filename in img_names_sorted:
        image = load_img(filename, target_size=(224, 224))
        data[i] = preprocess_input(img_to_array(image))
        i = i + 1
   
    ## 5-fold CV
    np.random.seed(seed)
    skf = StratifiedKFold(n_splits=5,shuffle=True)

    dx = (np.loadtxt('gender.txt') > 0.5)
    sc = np.loadtxt('skincolor.txt')

    fold = 1
    for train_idx, test_idx in skf.split(data,dx):

        train_data = data[train_idx]
        train_dx = dx[train_idx]
        train_sc = sc[train_idx]   
 
        test_data = data[test_idx]
        test_dx = dx[test_idx]
        test_sc = sc[test_idx]

        print('CV fold %d' % fold)
        sys.stdout.flush()

        ## augment data manually
        train_data_pos = train_data[train_dx==1];
        train_data_neg = train_data[train_dx==0];
        train_sc_pos = train_sc[train_dx==1];
        train_sc_neg = train_sc[train_dx==0];
        train_dx_pos = train_dx[train_dx==1];
        train_dx_neg = train_dx[train_dx==0];
        [train_data_pos_aug,train_sc_pos_aug,train_dx_pos_aug] = augment_by_flipping(train_data_pos,train_sc_pos,train_dx_pos,1200)
        [train_data_neg_aug,train_sc_neg_aug,train_dx_neg_aug] = augment_by_flipping(train_data_neg,train_sc_neg,train_dx_neg,1200)

        train_data_aug = np.concatenate((train_data_neg_aug, train_data_pos_aug), axis=0)
        train_sc_aug = np.concatenate((train_sc_neg_aug, train_sc_pos_aug), axis=0)
        train_dx_aug = np.concatenate((train_dx_neg_aug, train_dx_pos_aug), axis=0)

        print("Augmenting data ...")
        sys.stdout.flush()

        gan = GAN()
        gan.train(epochs=1001, training=[train_data_aug, train_sc_aug, train_dx_aug], testing=[test_data, test_sc, test_dx],  batch_size=64, fold=fold)

        print("Training finished on fold")
        fold = fold + 1
