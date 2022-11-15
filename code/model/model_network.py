import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.models import Model
from keras_multi_head import MultiHeadAttention
from tensorflow.keras.layers import GRU,Embedding,Activation,ReLU,AveragePooling2D,MaxPool2D,BatchNormalization,Conv1D,Attention, Dense, Conv2D, Bidirectional, LSTM, Flatten, Input, Activation, Reshape, Dropout, Concatenate, AveragePooling1D, MaxPool1D, BatchNormalization, Attention, GlobalAveragePooling1D, GlobalMaxPool1D, GRU, AdditiveAttention, AlphaDropout, LeakyReLU
from tensorflow.keras.initializers import VarianceScaling,RandomUniform
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, average_precision_score
from keras_layer_normalization import LayerNormalization
from tensorflow.keras.initializers import glorot_normal
import time
import shutil
from keras_bert import get_custom_objects
# from tensorflow.keras.layers.embeddings import Embedding
# from keras.initializers import RandomUniform
from tensorflow.python.keras.layers.core import Reshape, Permute
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import multiply
from tensorflow.python.keras.layers.core import Dense, Dropout, Lambda, Flatten
# import keras


VOCAB_SIZE = 16
EMBED_SIZE = 90
MAXLEN = 23
seed = 123

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc,precision_recall_curve

# from tensorflow.python.ops.numpy_ops import np_config
# np_config.enable_numpy_behavior()

def plotPrecisionRecallCurve(
		estimators, labels,
		xtests, ytests,
		flnm, icol=1):
	indx = 0
	plt.figure()
	plt.plot([0, 1], [0, 1], 'k--')
	for estimator in estimators:
		if len(ytests[indx].shape) == 2:
			pre, rec, _ = precision_recall_curve(
				ytests[indx][:,icol],
				estimator.predict(xtests[indx])[:,icol],
				pos_label=icol)
		else:
			pre, rec, _ = precision_recall_curve(
				ytests[indx],
				estimator.predict_proba(xtests[indx])[:,icol],
				pos_label=icol)
		#
		plt.plot(
			rec, pre,
			label=labels[indx] + ' (AUC: %s \u00B1 0.001)' % (
				np.round(auc(rec, pre), 3))
		)
		indx += 1
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.legend(loc='best')
	plt.savefig(flnm)

def plotRocCurve(
		estimators, labels,
		xtests, ytests,
		flnm, icol=1):
	indx = 0
	plt.figure()
	plt.plot([0, 1], [0, 1], 'k--')
	for estimator in estimators:
		if len(ytests[indx].shape) == 2:
			fprs, tprs, _ = roc_curve(
				ytests[indx][:,icol],
				estimator.predict(xtests[indx])[:,icol]
			)
		else:
			fprs, tprs, _ = roc_curve(
				ytests[indx],
				estimator.predict_proba(xtests[indx])[:,icol]
			)
		# print(estimator)
		plt.plot(
			fprs, tprs,
			label=labels[indx] + ' (AUC: %s \u00B1 0.001)' % (
				np.round(auc(fprs, tprs), 3))
		)
		indx += 1
	#
	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	plt.legend(loc='best')
	plt.savefig(flnm)

def transformIO(xtrain, xtest, ytrain, ytest,xval,yval,seq_len , coding_dim, num_classes):
    xtrain = xtrain.reshape(xtrain.shape[0], 1,seq_len, coding_dim)
    xtest = xtest.reshape(xtest.shape[0],1,seq_len, coding_dim)
    xval = xval.reshape(xval.shape[0], 1, seq_len, coding_dim)
    input_shape = (1,seq_len, coding_dim)
    xtrain = xtrain.astype('float32')
    xtest = xtest.astype('float32')
    xval = xval.astype('float32')
    print('xtrain shape:', xtrain.shape)
    print(xtrain.shape[0], 'train samples')
    print(xtest.shape[0], 'test samples')
    print(xval.shape[0], 'val samples')

    ytrain = to_categorical(ytrain, num_classes)
    ytest = to_categorical(ytest, num_classes)
    yval = to_categorical(yval, num_classes)
    return xtrain, xtest, ytrain, ytest,xval,yval, input_shape

def CRISPR_Net_transformIO(xtrain, xtest, ytrain, ytest,xval,yval,seq_len , coding_dim, num_classes):
    xtrain = xtrain.reshape(xtrain.shape[0], 1,seq_len, coding_dim)
    xtest = xtest.reshape(xtest.shape[0],1,seq_len, coding_dim)
    xval = xval.reshape(xval.shape[0], 1, seq_len, coding_dim)
    input_shape = (1,seq_len, coding_dim)
    xtrain = xtrain.astype('float32')
    xtest = xtest.astype('float32')
    xval = xval.astype('float32')
    print('xtrain shape:', xtrain.shape)
    print(xtrain.shape[0], 'train samples')
    print(xtest.shape[0], 'test samples')
    print(xval.shape[0], 'val samples')

    ytrain = to_categorical(ytrain, num_classes)
    ytest = to_categorical(ytest, num_classes)
    yval = to_categorical(yval, num_classes)
    return xtrain, xtest, ytrain, ytest,xval,yval, input_shape

def cnn_std_transformIO(xtrain, xtest, ytrain, ytest,xval,yval,seq_len , coding_dim, num_classes):
    xtrain = xtrain.reshape(xtrain.shape[0], 1,seq_len, coding_dim)
    xtest = xtest.reshape(xtest.shape[0],1,seq_len, coding_dim)
    xval = xval.reshape(xval.shape[0], 1, seq_len, coding_dim)
    input_shape = (1,seq_len, coding_dim)
    xtrain = xtrain.astype('float32')
    xtest = xtest.astype('float32')
    xval = xval.astype('float32')
    print('xtrain shape:', xtrain.shape)
    print(xtrain.shape[0], 'train samples')
    print(xtest.shape[0], 'test samples')
    print(xval.shape[0], 'xval samples')

    ytrain = to_categorical(ytrain, num_classes)
    ytest = to_categorical(ytest, num_classes)
    yval = to_categorical(yval, num_classes)
    return xtrain, xtest, ytrain, ytest,xval,yval, input_shape

def new_transformIO(xtrain, xtest, ytrain, ytest,seq_len , coding_dim, num_classes):
    xtrain = xtrain.reshape(xtrain.shape[0], seq_len, coding_dim,1)
    xtest = xtest.reshape(xtest.shape[0],seq_len, coding_dim,1)
    input_shape = (seq_len, coding_dim,1)
    xtrain = xtrain.astype('float32')
    xtest = xtest.astype('float32')
    print('xtrain shape:', xtrain.shape)
    print(xtrain.shape[0], 'train samples')
    print(xtest.shape[0], 'test samples')

    ytrain = to_categorical(ytrain, num_classes)
    ytest = to_categorical(ytest, num_classes)
    return xtrain, xtest, ytrain, ytest, input_shape

def offt_transformIO(xtrain, xtest, ytrain, ytest ,xval,yval, num_classes):
    xtrain = xtrain.astype('float32')
    xtest = xtest.astype('float32')
    xval = xval.astype('float32')
    print('xtrain shape:', xtrain.shape)
    print(xtrain.shape[0], 'train samples')
    print(xtest.shape[0], 'test samples')
    print(xval.shape[0], 'val samples')

    ytrain = to_categorical(ytrain, num_classes)
    ytest = to_categorical(ytest, num_classes)
    yval = to_categorical(yval, num_classes)
    return xtrain, xtest, ytrain, ytest,xval,yval

def pencil_transformIO(label,xtrain, xtest, ytrain, ytest, xval,yval,seq_len , coding_dim, num_classes):
    xtrain = xtrain.reshape(xtrain.shape[0], 1,seq_len, coding_dim)
    xval = xval.reshape(xval.shape[0], 1,seq_len, coding_dim)
    xtest = xtest.reshape(xtest.shape[0],1,seq_len, coding_dim)
    input_shape = (1,seq_len, coding_dim)
    xtrain = xtrain.astype('float32')
    xval = xval.astype('float32')
    xtest = xtest.astype('float32')
    if label == 0:
        print('xtrain shape:', xtrain.shape)
        print(xtrain.shape[0], 'train samples')
    if label == 1:
        print('xtest shape:', xtest.shape)
        print(xtest.shape[0], 'test samples')
    if label == 2:
        print('xval shape:', xval.shape)
        print(xval.shape[0], 'val samples')

    ytrain = to_categorical(ytrain, num_classes)
    yval = to_categorical(yval, num_classes)
    ytest = to_categorical(ytest, num_classes)
    return xtrain, xtest, ytrain, ytest,xval,yval, input_shape

def guideseq_transformIO(x, y,seq_len , coding_dim, num_classes):
    x = x.reshape(x.shape[0], seq_len, coding_dim,1)
    input_shape = (seq_len, coding_dim,1)
    x = x.astype('float32')
    print('x shape:', x.shape)
    print(x.shape[0], 'samples')

    y = to_categorical(y, num_classes)

    return x, y, input_shape

def guideseq_pencil_transformIO(x, y,seq_len , coding_dim, num_classes):
    x = x.reshape(x.shape[0], 1,seq_len, coding_dim)
    input_shape = (1,seq_len, coding_dim)
    x = x.astype('float32')
    print('x shape:', x.shape)
    print(x.shape[0], 'samples')

    y = to_categorical(y, num_classes)

    return x, y, input_shape

def cnn_transformIO(xtrain, xtest, ytrain, ytest, seq_len , coding_dim, num_classes):
    xtrain = xtrain.reshape(xtrain.shape[0], seq_len, coding_dim)
    xtest = xtest.reshape(xtest.shape[0],seq_len, coding_dim)
    input_shape = (seq_len, coding_dim)
    xtrain = xtrain.astype('float32')
    xtest = xtest.astype('float32')
    print('xtrain shape:', xtrain.shape)
    print(xtrain.shape[0], 'train samples')
    print(xtest.shape[0], 'test samples')

    ytrain = to_categorical(ytrain, num_classes)
    ytest = to_categorical(ytest, num_classes)
    return xtrain, xtest, ytrain, ytest, input_shape

class NormalizedCrossEntropy(tf.keras.losses.Loss):
    def __init__(self,scale=1.0,**kwargs):
        super(NormalizedCrossEntropy, self).__init__()
        self.scale = scale

    def call(self, y_true, y_pred):
        nce = -1 * tf.reduce_sum(y_true * y_pred, axis=-1) / (- tf.reduce_sum(y_pred,axis=-1))
        return self.scale * tf.reduce_mean(nce)

class ReverseCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, scale=1.0,**kwargs):
        super(ReverseCrossEntropy, self).__init__()
        self.scale = scale

    def call(self, y_true, y_pred):
        y_true_2 = y_true
        y_pred_2 = y_pred

        y_true_2 = tf.clip_by_value(y_true_2, 1e-4, 1.0)

        y_pred_2 = tf.clip_by_value(y_pred_2, 1e-4, 1.0)

        return self.scale * tf.reduce_mean(-tf.reduce_sum(y_pred_2 * tf.compat.v1.log(y_true_2), axis=-1))

class MeanAbsoluteError(tf.keras.losses.Loss):
    def __init__(self, scale=1.0,**kwargs):
        super(MeanAbsoluteError, self).__init__()
        self.scale = scale

    def call(self, y_true, y_pred):
        mae = 1. - tf.reduce_sum(y_true * y_pred, axis=-1)
        return self.scale * tf.reduce_mean(mae)

class NCEandMAE(tf.keras.losses.Loss):
    def __init__(self, alpha=0.1, beta=0.1,**kwargs):
        super(NCEandMAE, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.nce = NormalizedCrossEntropy(scale=self.alpha)
        self.mae = MeanAbsoluteError(scale=self.beta)

    def call(self, y_true, y_pred):
        return self.nce(y_true, y_pred) + self.mae(y_true, y_pred)

class NCEandRCE(tf.keras.losses.Loss):
    def __init__(self, alpha=0.1, beta=0.1,**kwargs):
        super(NCEandRCE, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.nce = NormalizedCrossEntropy(scale=self.alpha)
        self.rce = ReverseCrossEntropy(scale=self.beta)

    def call(self, y_true, y_pred):
        return self.nce(y_true, y_pred) + self.rce(y_true, y_pred)


class GCE(tf.keras.losses.Loss):
    def __init__(self, q=0.7,**kwargs):
        super(GCE, self).__init__()
        self.q = q

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, clip_value_min=tf.keras.backend.epsilon(), clip_value_max=1)
        t_loss = (1 - tf.pow(tf.reduce_sum(y_true * y_pred, axis=-1), self.q)) / self.q
        return tf.reduce_mean(t_loss)

class LSR(tf.keras.losses.Loss):
    def __init__(self, epsilon=0.1,**kwargs):
        super(LSR, self).__init__()
        self.epsilon = epsilon

    def call(self, y_true, y_pred):
        y_smoothed_true = y_true * (1 - self.epsilon - self.epsilon / 10.0)
        y_smoothed_true = y_smoothed_true + self.epsilon / 10.0

        y_pred_1 = tf.clip_by_value(y_pred, 1e-7, 1.0)

        return tf.reduce_mean(-tf.reduce_sum(y_smoothed_true * tf.compat.v1.log(y_pred_1), axis=-1))

class SCE(tf.keras.losses.Loss):
    def __init__(self, alpha=0.1,beta=0.1,**kwargs):
        super(SCE, self).__init__()
        self.alpha = alpha
        self.beta = beta


    def call(self, y_true, y_pred):
        y_true_1 = y_true
        y_pred_1 = y_pred

        y_true_2 = y_true
        y_pred_2 = y_pred

        y_pred_1 = tf.clip_by_value(y_pred_1, 1e-7, 1.0)
        y_true_2 = tf.clip_by_value(y_true_2, 1e-4, 1.0)

        return self.alpha * tf.reduce_mean(-tf.reduce_sum(y_true_1 * tf.compat.v1.log(y_pred_1), axis=-1)) + self.beta * tf.reduce_mean(
            -tf.reduce_sum(y_pred_2 * tf.compat.v1.log(y_true_2), axis=-1))

#         return tf.reduce_mean(t_loss)

def CnnCrispr(embedding_weights,test_ds,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, xtest, ytest,  num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    VOCAB_SIZE = 16
    EMBED_SIZE = 100
    maxlen = 23
    if retrain or not os.path.exists('{}+CnnCrispr.h5'.format(saved_prefix)):
        input = Input(shape=(23,))
        embedding = Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=maxlen,
                            weights=[embedding_weights],
                            trainable=True)(input)
        bi_lstm = Bidirectional(LSTM(40, return_sequences=True))(embedding)
        bi_lstm_relu = Activation('relu')(bi_lstm)

        conv1 = Conv1D(10, (5))(bi_lstm_relu)
        conv1_relu = Activation('relu')(conv1)
        conv1_batch = BatchNormalization()(conv1_relu)

        conv2 = Conv1D(20, (5))(conv1_batch)
        conv2_relu = Activation('relu')(conv2)
        conv2_batch = BatchNormalization()(conv2_relu)

        conv3 = Conv1D(40, (5))(conv2_batch)
        conv3_relu = Activation('relu')(conv3)
        conv3_batch = BatchNormalization()(conv3_relu)

        conv4 = Conv1D(80, (5))(conv3_batch)
        conv4_relu = Activation('relu')(conv4)
        conv4_batch = BatchNormalization()(conv4_relu)

        conv5 = Conv1D(100, (5))(conv4_batch)
        conv5_relu = Activation('relu')(conv5)
        conv5_batch = BatchNormalization()(conv5_relu)

        flat = Flatten()(conv5_batch)
        drop = Dropout(0.3)(flat)
        dense = Dense(20)(drop)
        dense_relu = Activation('relu')(dense)
        prediction = Dense(2, activation='softmax', name='main_output')(dense_relu)
        model = Model(input, prediction)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            resampled_ds,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=test_ds,
            steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        yscore = model.predict(xtest)
        # print(yscore)
        ypred = np.argmax(yscore, axis=1)
        # print(ypred)
        yscore = yscore[:, 1]
        # print(yscore)
        ytest = np.argmax(ytest, axis=1)
        # print(ytest)
        eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score]
        eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
        eval_fun_types = [True, True, True, True, False, False]
        for index_f, function in enumerate(eval_funs):
            if eval_fun_types[index_f]:
                score = np.round(function(ytest, ypred), 4)
            else:
                score = np.round(function(ytest, yscore), 4)
            print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))
        model.save('{}+CnnCrispr.h5'.format(saved_prefix))
    else:
        model = load_model('{}+CnnCrispr.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
    return model

def conv2d_bn(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=True,
              name=None, trainable=True):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        kernel_size: kernel size as in `Conv2D`.
        strides: strides in `Conv2D`.
        padding: padding mode in `Conv2D`.
        activation: activation in `Conv2D`.
        use_bias: whether to use a bias in `Conv2D`.
        name: name of the ops; will become `name + '_ac'` for the activation
            and `name + '_bn'` for the batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    x = Conv2D(filters,
                      kernel_size,
                      strides=strides,
                      padding=padding,
                      use_bias=use_bias,
                      kernel_initializer=glorot_normal(seed=seed),
                      name=name, trainable=trainable)(x)

    # x = layers.BatchNormalization(axis=-1,scale=True)(x)
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        x = Activation(activation, name=ac_name)(x)
    return x

def saidao2_CRISPR_Net_model(test_ds,xval,yval,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, xtest, ytest, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    roc_auc = 0
    pr_auc = 0
    if retrain or not os.path.exists('{}+CRISPR_Net.h5'.format(saved_prefix)):
        inputs = Input(shape=(1, 1, 100), name='main_input')
        branch_0 = conv2d_bn(inputs, 10, (1, 1))
        print(branch_0.shape)
        branch_1 = conv2d_bn(inputs, 10, (1, 2))
        print(branch_1.shape)
        branch_2 = conv2d_bn(inputs, 10, (1, 3))
        print(branch_2.shape)
        branch_3 = conv2d_bn(inputs, 10, (1, 5))
        print(branch_3.shape)
        branches = [inputs, branch_0, branch_1, branch_2, branch_3]
        # branches = [branch_0, branch_1, branch_2, branch_3]
        mixed = Concatenate(axis=-1)(branches)
        print(mixed.shape)
        mixed = Reshape((1, 140))(mixed)
        print(mixed.shape)
        blstm_out = Bidirectional(LSTM(15, kernel_initializer=glorot_normal(seed=seed),return_sequences=True, input_shape=(1, 140), name="LSTM_out"))(mixed)
        print(blstm_out.shape)
        # inputs_rs = Reshape((24, 7))(inputs)
        # blstm_out = layers.Concatenate(axis=-1)([mixed, blstm_out])
        blstm_out = Flatten()(blstm_out)
        x = Dense(80, kernel_initializer=glorot_normal(seed=seed),activation='relu')(blstm_out)
        x = Dense(20, kernel_initializer=glorot_normal(seed=seed),activation='relu')(x)
        x = Dropout(0.35,seed=seed)(x)
        prediction = Dense(2,kernel_initializer=glorot_normal(seed=seed), activation='softmax', name='main_output')(x)
        model = Model(inputs, prediction)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            xtrain,ytrain,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=[xval,yval],
            # steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        yscore = model.predict(xtest)
        # print(yscore)
        ypred = np.argmax(yscore, axis=1)
        # print(ypred)
        yscore = yscore[:, 1]
        # print(yscore)
        ytest = np.argmax(ytest, axis=1)
        # print(ytest)
        eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score]
        eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
        eval_fun_types = [True, True, True, True, False, False]
        for index_f, function in enumerate(eval_funs):
            if eval_fun_types[index_f]:
                score = np.round(function(ytest, ypred), 4)
            else:
                score = np.round(function(ytest, yscore), 4)
                if index_f == 4:
                    roc_auc = score
                if index_f == 5:
                    pr_auc = score
            print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))
        model.save('{}+CRISPR_Net.h5'.format(saved_prefix))
    else:
        model = load_model('{}+CRISPR_Net.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
    return model

def CRISPR_Net_model(test_ds,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, xtest, ytest, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    roc_auc = 0
    pr_auc = 0
    if retrain or not os.path.exists('{}+CRISPR_Net.h5'.format(saved_prefix)):
        inputs = Input(shape=(1, 23, 6), name='main_input')
        branch_0 = conv2d_bn(inputs, 10, (1, 1))
        print(branch_0.shape)
        branch_1 = conv2d_bn(inputs, 10, (1, 2))
        print(branch_1.shape)
        branch_2 = conv2d_bn(inputs, 10, (1, 3))
        print(branch_2.shape)
        branch_3 = conv2d_bn(inputs, 10, (1, 5))
        print(branch_3.shape)
        branches = [inputs, branch_0, branch_1, branch_2, branch_3]
        # branches = [branch_0, branch_1, branch_2, branch_3]
        mixed = Concatenate(axis=-1)(branches)
        print(mixed.shape)
        mixed = Reshape((23, 46))(mixed)
        print(mixed.shape)
        blstm_out = Bidirectional(LSTM(15, kernel_initializer=glorot_normal(seed=seed),return_sequences=True, input_shape=(23, 46), name="LSTM_out"))(mixed)
        print(blstm_out.shape)
        # inputs_rs = Reshape((24, 7))(inputs)
        # blstm_out = layers.Concatenate(axis=-1)([mixed, blstm_out])
        blstm_out = Flatten()(blstm_out)
        x = Dense(80, kernel_initializer=glorot_normal(seed=seed),activation='relu')(blstm_out)
        x = Dense(20, kernel_initializer=glorot_normal(seed=seed),activation='relu')(x)
        x = Dropout(0.35,seed=seed)(x)
        prediction = Dense(2,kernel_initializer=glorot_normal(seed=seed), activation='softmax', name='main_output')(x)
        model = Model(inputs, prediction)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            resampled_ds,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=test_ds,
            steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        yscore = model.predict(xtest)
        # print(yscore)
        ypred = np.argmax(yscore, axis=1)
        # print(ypred)
        yscore = yscore[:, 1]
        # print(yscore)
        ytest = np.argmax(ytest, axis=1)
        # print(ytest)
        eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score]
        eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
        eval_fun_types = [True, True, True, True, False, False]
        for index_f, function in enumerate(eval_funs):
            if eval_fun_types[index_f]:
                score = np.round(function(ytest, ypred), 4)
            else:
                score = np.round(function(ytest, yscore), 4)
                if index_f == 4:
                    roc_auc = score
                if index_f == 5:
                    pr_auc = score
            print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))
        model.save('{}+CRISPR_Net.h5'.format(saved_prefix))
    else:
        model = load_model('{}+CRISPR_Net.h5'.format(saved_prefix))
        # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
    return model


def NCEandRCE_crispr_Net(test_ds,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, xtest, ytest, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+NCEandRCE_crispr_Net.h5'.format(saved_prefix)):
        alpha_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0, 2.0,3.0,4.0,5.0,10.0]
        beta_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,2.0,3.0,4.0,5.0,10.0,100.0]
        # alpha_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 10.0]
        # beta_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 10.0]
        best_pr = 0
        best_roc = 0
        best_beta = 0
        best_alpha = 0
        for alpha in alpha_list:
            for beta in beta_list:
                print(alpha)
                print(beta)
                inputs = Input(shape=(1, 23, 6), name='main_input')
                branch_0 = conv2d_bn(inputs, 10, (1, 1))
                print(branch_0.shape)
                branch_1 = conv2d_bn(inputs, 10, (1, 2))
                print(branch_1.shape)
                branch_2 = conv2d_bn(inputs, 10, (1, 3))
                print(branch_2.shape)
                branch_3 = conv2d_bn(inputs, 10, (1, 5))
                print(branch_3.shape)
                branches = [inputs, branch_0, branch_1, branch_2, branch_3]
                # branches = [branch_0, branch_1, branch_2, branch_3]
                mixed = Concatenate(axis=-1)(branches)
                print(mixed.shape)
                mixed = Reshape((23, 46))(mixed)
                print(mixed.shape)
                blstm_out = Bidirectional(LSTM(15, return_sequences=True, input_shape=(23, 46), name="LSTM_out"))(mixed)
                print(blstm_out.shape)
                # inputs_rs = Reshape((24, 7))(inputs)
                # blstm_out = layers.Concatenate(axis=-1)([mixed, blstm_out])
                blstm_out = Flatten()(blstm_out)
                x = Dense(80, activation='relu')(blstm_out)
                x = Dense(20, activation='relu')(x)
                x = Dropout(0.35)(x)
                prediction = Dense(2, activation='softmax', name='main_output')(x)
                model = Model(inputs, prediction)
                model.compile(tf.keras.optimizers.Adam(), loss=NCEandRCE(alpha=alpha,beta=beta), metrics=['accuracy'])#Adam是0.001，SGD是0.01
                history_model = model.fit(
                    resampled_ds,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=test_ds,
                    steps_per_epoch=resampled_steps_per_epoch,
                    callbacks=callbacks
                )
                yscore = model.predict(xtest)
                ypred = np.argmax(yscore, axis=1)
                print(ypred)
                yscore = yscore[:, 1]
                # print(yscore)
                gcetest = np.argmax(ytest, axis=1)
                print(gcetest)
                eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score]
                eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
                eval_fun_types = [True, True, True, True, False, False]
                for index_f, function in enumerate(eval_funs):
                    if eval_fun_types[index_f]:
                        score = np.round(function(gcetest, ypred), 4)
                    else:
                        score = np.round(function(gcetest, yscore), 4)
                        if index_f == 5:
                            if score > best_pr and np.round(roc_auc_score(gcetest, yscore), 4) > best_roc:
                                best_pr = score
                                best_roc = np.round(roc_auc_score(gcetest, yscore), 4)
                                best_alpha = alpha
                                best_beta = beta
                                model.save('{}+NCEandRCE_crispr_Net.h5'.format(saved_prefix))
                    print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))
                # model.save('{}+new_crispr_ip.h5'.format(saved_prefix))
    # else:
    #     # print('cunzai')
    #     model = load_model('{}+NCEandRCE_crispr_Net.h5'.format(saved_prefix),custom_objects={'PositionalEncoding': PositionalEncoding})
    # model = load_model('{}+sce_crispr_ip.h5'.format(saved_prefix))
    custom_objects = get_custom_objects()
    my_objects = {'PositionalEncoding': PositionalEncoding,'NCEandRCE':NCEandRCE}
    custom_objects.update(my_objects)
    model = load_model('{}+NCEandRCE_crispr_Net.h5'.format(saved_prefix),
                       custom_objects=custom_objects)
    return model

def NCEandMAE_crispr_Net(test_ds,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, xtest, ytest, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+NCEandMAE_crispr_Net.h5'.format(saved_prefix)):
        # alpha_list = [0.1, 1.0, 10.0]
        # beta_list = [0.1,1.0,10.0,100.0]
        alpha_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0]
        beta_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0,100.0]
        # alpha_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 10.0]
        # beta_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 10.0]
        # beta_list = [0.1]
        best_pr = 0
        best_roc = 0
        best_beta = 0
        best_alpha = 0
        for alpha in alpha_list:
            for beta in beta_list:
                print(alpha)
                print(beta)
                inputs = Input(shape=(1, 23, 6), name='main_input')
                branch_0 = conv2d_bn(inputs, 10, (1, 1))
                print(branch_0.shape)
                branch_1 = conv2d_bn(inputs, 10, (1, 2))
                print(branch_1.shape)
                branch_2 = conv2d_bn(inputs, 10, (1, 3))
                print(branch_2.shape)
                branch_3 = conv2d_bn(inputs, 10, (1, 5))
                print(branch_3.shape)
                branches = [inputs, branch_0, branch_1, branch_2, branch_3]
                # branches = [branch_0, branch_1, branch_2, branch_3]
                mixed = Concatenate(axis=-1)(branches)
                print(mixed.shape)
                mixed = Reshape((23, 46))(mixed)
                print(mixed.shape)
                blstm_out = Bidirectional(LSTM(15, return_sequences=True, input_shape=(23, 46), name="LSTM_out"))(mixed)
                print(blstm_out.shape)
                # inputs_rs = Reshape((24, 7))(inputs)
                # blstm_out = layers.Concatenate(axis=-1)([mixed, blstm_out])
                blstm_out = Flatten()(blstm_out)
                x = Dense(80, activation='relu')(blstm_out)
                x = Dense(20, activation='relu')(x)
                x = Dropout(0.35)(x)
                prediction = Dense(2, activation='softmax', name='main_output')(x)
                model = Model(inputs, prediction)
                model.compile(tf.keras.optimizers.Adam(), loss=NCEandMAE(alpha=alpha,beta=beta), metrics=['accuracy'])#Adam是0.001，SGD是0.01
                history_model = model.fit(
                    resampled_ds,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=test_ds,
                    steps_per_epoch=resampled_steps_per_epoch,
                    callbacks=callbacks
                )
                yscore = model.predict(xtest)
                ypred = np.argmax(yscore, axis=1)
                print(ypred)
                yscore = yscore[:, 1]
                # print(yscore)
                gcetest = np.argmax(ytest, axis=1)
                print(gcetest)
                eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score]
                eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
                eval_fun_types = [True, True, True, True, False, False]
                for index_f, function in enumerate(eval_funs):
                    if eval_fun_types[index_f]:
                        score = np.round(function(gcetest, ypred), 4)
                    else:
                        score = np.round(function(gcetest, yscore), 4)
                        if index_f == 5:
                            if score > best_pr and np.round(roc_auc_score(gcetest, yscore), 4) > best_roc:
                                best_pr = score
                                best_roc = np.round(roc_auc_score(gcetest, yscore), 4)
                                best_alpha = alpha
                                best_beta = beta
                                model.save('{}+NCEandMAE_crispr_Net.h5'.format(saved_prefix))
                    print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))
                # model.save('{}+new_crispr_ip.h5'.format(saved_prefix))
    # else:
    #     # print('cunzai')
    #     model = load_model('{}+NCEandMAE_crispr_Net.h5'.format(saved_prefix),custom_objects={'PositionalEncoding': PositionalEncoding})
    # model = load_model('{}+sce_crispr_ip.h5'.format(saved_prefix))
    custom_objects = get_custom_objects()
    my_objects = {'PositionalEncoding': PositionalEncoding,'NCEandMAE':NCEandMAE}
    custom_objects.update(my_objects)
    model = load_model('{}+NCEandMAE_crispr_Net.h5'.format(saved_prefix),
                       custom_objects=custom_objects)
    return model

def sce_crispr_Net(test_ds,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, xtest, ytest, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+sce_crispr_Net.h5'.format(saved_prefix)):
        beta_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,2.0,3.0,4.0,5.0]
        alpha_list = [0.01,0.05,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0]
        # alpha_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        # beta_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        # beta_list = [0.1]
        best_pr = 0
        best_roc = 0
        best_beta = 0
        best_alpha = 0
        for alpha in alpha_list:
            for beta in beta_list:
                print(alpha)
                print(beta)
                inputs = Input(shape=(1, 23, 6), name='main_input')
                branch_0 = conv2d_bn(inputs, 10, (1, 1))
                print(branch_0.shape)
                branch_1 = conv2d_bn(inputs, 10, (1, 2))
                print(branch_1.shape)
                branch_2 = conv2d_bn(inputs, 10, (1, 3))
                print(branch_2.shape)
                branch_3 = conv2d_bn(inputs, 10, (1, 5))
                print(branch_3.shape)
                branches = [inputs, branch_0, branch_1, branch_2, branch_3]
                # branches = [branch_0, branch_1, branch_2, branch_3]
                mixed = Concatenate(axis=-1)(branches)
                print(mixed.shape)
                mixed = Reshape((23, 46))(mixed)
                print(mixed.shape)
                blstm_out = Bidirectional(LSTM(15, return_sequences=True, input_shape=(23, 46), name="LSTM_out"))(mixed)
                print(blstm_out.shape)
                # inputs_rs = Reshape((24, 7))(inputs)
                # blstm_out = layers.Concatenate(axis=-1)([mixed, blstm_out])
                blstm_out = Flatten()(blstm_out)
                x = Dense(80, activation='relu')(blstm_out)
                x = Dense(20, activation='relu')(x)
                x = Dropout(0.35)(x)
                prediction = Dense(2, activation='softmax', name='main_output')(x)
                model = Model(inputs, prediction)
                model.compile(tf.keras.optimizers.Adam(), loss=SCE(alpha=alpha,beta=beta), metrics=['accuracy'])#Adam是0.001，SGD是0.01
                history_model = model.fit(
                    resampled_ds,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=test_ds,
                    steps_per_epoch=resampled_steps_per_epoch,
                    callbacks=callbacks
                )
                yscore = model.predict(xtest)
                ypred = np.argmax(yscore, axis=1)
                print(ypred)
                yscore = yscore[:, 1]
                # print(yscore)
                gcetest = np.argmax(ytest, axis=1)
                print(gcetest)
                eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score]
                eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
                eval_fun_types = [True, True, True, True, False, False]
                for index_f, function in enumerate(eval_funs):
                    if eval_fun_types[index_f]:
                        score = np.round(function(gcetest, ypred), 4)
                    else:
                        score = np.round(function(gcetest, yscore), 4)
                        if index_f == 5:
                            if score > best_pr and np.round(roc_auc_score(gcetest, yscore), 4) > best_roc:
                                best_pr = score
                                best_roc = np.round(roc_auc_score(gcetest, yscore), 4)
                                best_alpha = alpha
                                best_beta = beta
                                model.save('{}+sce_crispr_Net.h5'.format(saved_prefix))
                    print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))
                # model.save('{}+new_crispr_ip.h5'.format(saved_prefix))
    # else:
    #     # print('cunzai')
    #     model = load_model('{}+sce_crispr_Net.h5'.format(saved_prefix),custom_objects={'PositionalEncoding': PositionalEncoding})
    # model = load_model('{}+sce_crispr_ip.h5'.format(saved_prefix))
    custom_objects = get_custom_objects()
    my_objects = {'PositionalEncoding': PositionalEncoding,'SCE':SCE}
    custom_objects.update(my_objects)
    model = load_model('{}+sce_crispr_Net.h5'.format(saved_prefix),
                       custom_objects=custom_objects)
    return model

def gce_crispr_Net(test_ds,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, xtest, ytest, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+gce_crispr_Net.h5'.format(saved_prefix)):
        q_list = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        best_pr = 0
        best_roc = 0
        best_q = 0
        for q in q_list:
            print(q)
            inputs = Input(shape=(1, 23, 6), name='main_input')
            branch_0 = conv2d_bn(inputs, 10, (1, 1))
            print(branch_0.shape)
            branch_1 = conv2d_bn(inputs, 10, (1, 2))
            print(branch_1.shape)
            branch_2 = conv2d_bn(inputs, 10, (1, 3))
            print(branch_2.shape)
            branch_3 = conv2d_bn(inputs, 10, (1, 5))
            print(branch_3.shape)
            branches = [inputs, branch_0, branch_1, branch_2, branch_3]
            # branches = [branch_0, branch_1, branch_2, branch_3]
            mixed = Concatenate(axis=-1)(branches)
            print(mixed.shape)
            mixed = Reshape((23, 46))(mixed)
            print(mixed.shape)
            blstm_out = Bidirectional(LSTM(15, return_sequences=True, input_shape=(23, 46), name="LSTM_out"))(mixed)
            print(blstm_out.shape)
            # inputs_rs = Reshape((24, 7))(inputs)
            # blstm_out = layers.Concatenate(axis=-1)([mixed, blstm_out])
            blstm_out = Flatten()(blstm_out)
            x = Dense(80, activation='relu')(blstm_out)
            x = Dense(20, activation='relu')(x)
            x = Dropout(0.35)(x)
            prediction = Dense(2, activation='softmax', name='main_output')(x)
            model = Model(inputs, prediction)
            model.compile(tf.keras.optimizers.Adam(), loss=GCE(q=q), metrics=['accuracy'])#Adam是0.001，SGD是0.01
            history_model = model.fit(
                resampled_ds,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=test_ds,
                steps_per_epoch=resampled_steps_per_epoch,
                callbacks=callbacks
            )
            yscore = model.predict(xtest)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            yscore = yscore[:, 1]
            # print(yscore)
            gcetest = np.argmax(ytest, axis=1)
            print(gcetest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(gcetest, ypred), 4)
                else:
                    score = np.round(function(gcetest, yscore), 4)
                    if index_f == 5:
                        if score > best_pr and np.round(roc_auc_score(gcetest, yscore), 4) > best_roc:
                            best_pr = score
                            best_roc = np.round(roc_auc_score(gcetest, yscore), 4)
                            best_q = q
                            model.save('{}+gce_crispr_Net.h5'.format(saved_prefix))
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))
            # model.save('{}+new_crispr_ip.h5'.format(saved_prefix))
    # else:
    #     # print('cunzai')
    #     model = load_model('{}+gce_crispr_Net.h5'.format(saved_prefix),custom_objects={'PositionalEncoding': PositionalEncoding})
    # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    # model = load_model('{}+gce_crispr_ip.h5'.format(saved_prefix),
    #                    custom_objects={'PositionalEncoding': PositionalEncoding})
    custom_objects = get_custom_objects()
    my_objects = {'PositionalEncoding': PositionalEncoding, 'GCE': GCE}
    custom_objects.update(my_objects)
    model = load_model('{}+gce_crispr_Net.h5'.format(saved_prefix),
                       custom_objects=custom_objects)
    return model


def crispr_ip(test_ds,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, xtest, ytest, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+crispr_ip.h5'.format(saved_prefix)):
        initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
        input_value = Input(shape=input_shape)
        conv_1_output = Conv2D(60, (1, input_shape[-1]), padding='valid', data_format='channels_first',
                               kernel_initializer=initializer)(input_value)
        conv_1_output_reshape = Reshape(tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(
            conv_1_output)
        conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0, 2, 1])
        conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
        conv_1_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_1_output_reshape2)
        bidirectional_1_output = Bidirectional(
            LSTM(30, return_sequences=True, dropout=0.25, kernel_initializer=initializer))(
            Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max]))
        attention_1_output = Attention()([bidirectional_1_output, bidirectional_1_output])
        average_1_output = GlobalAveragePooling1D(data_format='channels_last')(attention_1_output)
        max_1_output = GlobalMaxPool1D(data_format='channels_last')(attention_1_output)
        concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
        flatten_output = Flatten()(concat_output)
        linear_1_output = BatchNormalization()(
            Dense(200, activation='relu', kernel_initializer=initializer)(flatten_output))
        linear_2_output = Dense(100, activation='relu', kernel_initializer=initializer)(linear_1_output)
        linear_2_output_dropout = Dropout(0.9)(linear_2_output)
        linear_3_output = Dense(num_classes, activation='softmax', kernel_initializer=initializer)(
            linear_2_output_dropout)
        model = Model(input_value, linear_3_output)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history_model = model.fit(
            resampled_ds,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=test_ds,
            steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        yscore = model.predict(xtest)
        # print(yscore)
        ypred = np.argmax(yscore, axis=1)
        # print(ypred)
        yscore = yscore[:, 1]
        # print(yscore)
        ytest = np.argmax(ytest, axis=1)
        # print(ytest)
        eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score]
        eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
        eval_fun_types = [True, True, True, True, False, False]
        for index_f, function in enumerate(eval_funs):
            if eval_fun_types[index_f]:
                score = np.round(function(ytest, ypred), 4)
            else:
                score = np.round(function(ytest, yscore), 4)
            print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))
        model.save('{}+crispr_ip.h5'.format(saved_prefix))
    else:
        model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
    # model = load_model('{}+crispr_ip.h5'.format(saved_prefix))
    return model

def NCEandRCE_crispr_ip_z(test_ds,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, xtest, ytest, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+NCEandRCE_crispr_ip.h5'.format(saved_prefix)):
        alpha_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0, 2.0,3.0,4.0,5.0,10.0]
        beta_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,2.0,3.0,4.0,5.0,10.0,100.0]
        # alpha_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 10.0]
        # beta_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 10.0]
        best_pr = 0
        best_roc = 0
        best_beta = 0
        best_alpha = 0
        for alpha in alpha_list:
            for beta in beta_list:
                print(alpha)
                print(beta)
                initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
                input_value = Input(shape=input_shape)
                conv_1_output = Conv2D(60, (1, input_shape[-1]), padding='valid', data_format='channels_first',
                                       kernel_initializer=initializer)(input_value)
                conv_1_output_reshape = Reshape(
                    tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(
                    conv_1_output)
                conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0, 2, 1])
                conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
                conv_1_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_1_output_reshape2)
                bidirectional_1_output = Bidirectional(
                    LSTM(30, return_sequences=True, dropout=0.25, kernel_initializer=initializer))(
                    Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max]))
                attention_1_output = Attention()([bidirectional_1_output, bidirectional_1_output])
                average_1_output = GlobalAveragePooling1D(data_format='channels_last')(attention_1_output)
                max_1_output = GlobalMaxPool1D(data_format='channels_last')(attention_1_output)
                concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
                flatten_output = Flatten()(concat_output)
                linear_1_output = BatchNormalization()(
                    Dense(200, activation='relu', kernel_initializer=initializer)(flatten_output))
                linear_2_output = Dense(100, activation='relu', kernel_initializer=initializer)(linear_1_output)
                linear_2_output_dropout = Dropout(0.9)(linear_2_output)
                linear_3_output = Dense(num_classes, activation='softmax', kernel_initializer=initializer)(
                    linear_2_output_dropout)
                model = Model(input_value, linear_3_output)
                model.compile(tf.keras.optimizers.Adam(), loss=NCEandRCE(alpha=alpha,beta=beta), metrics=['accuracy'])#Adam是0.001，SGD是0.01
                history_model = model.fit(
                    resampled_ds,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=test_ds,
                    steps_per_epoch=resampled_steps_per_epoch,
                    callbacks=callbacks
                )
                yscore = model.predict(xtest)
                ypred = np.argmax(yscore, axis=1)
                print(ypred)
                yscore = yscore[:, 1]
                # print(yscore)
                gcetest = np.argmax(ytest, axis=1)
                print(gcetest)
                eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score]
                eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
                eval_fun_types = [True, True, True, True, False, False]
                for index_f, function in enumerate(eval_funs):
                    if eval_fun_types[index_f]:
                        score = np.round(function(gcetest, ypred), 4)
                    else:
                        score = np.round(function(gcetest, yscore), 4)
                        if index_f == 5:
                            if score > best_pr and np.round(roc_auc_score(gcetest, yscore), 4) > best_roc:
                                best_pr = score
                                best_roc = np.round(roc_auc_score(gcetest, yscore), 4)
                                best_alpha = alpha
                                best_beta = beta
                                model.save('{}+NCEandRCE_crispr_ip.h5'.format(saved_prefix))
                    print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))
                # model.save('{}+new_crispr_ip.h5'.format(saved_prefix))
    # else:
    #     # print('cunzai')
    #     model = load_model('{}+NCEandRCE_crispr_ip.h5'.format(saved_prefix),custom_objects={'PositionalEncoding': PositionalEncoding})
    # model = load_model('{}+sce_crispr_ip.h5'.format(saved_prefix))
    custom_objects = get_custom_objects()
    my_objects = {'PositionalEncoding': PositionalEncoding,'NCEandRCE':NCEandRCE}
    custom_objects.update(my_objects)
    model = load_model('{}+NCEandRCE_crispr_ip.h5'.format(saved_prefix),
                       custom_objects=custom_objects)
    return model

def NCEandMAE_crispr_ip_z(test_ds,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, xtest, ytest, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+NCEandMAE_crispr_ip.h5'.format(saved_prefix)):
        # alpha_list = [0.1, 1.0, 10.0]
        # beta_list = [0.1,1.0,10.0,100.0]
        alpha_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0]
        beta_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0,100.0]
        # alpha_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 10.0]
        # beta_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 10.0]
        # beta_list = [0.1]
        best_pr = 0
        best_roc = 0
        best_beta = 0
        best_alpha = 0
        for alpha in alpha_list:
            for beta in beta_list:
                print(alpha)
                print(beta)
                initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
                input_value = Input(shape=input_shape)
                conv_1_output = Conv2D(60, (1, input_shape[-1]), padding='valid', data_format='channels_first',
                                       kernel_initializer=initializer)(input_value)
                conv_1_output_reshape = Reshape(
                    tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(
                    conv_1_output)
                conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0, 2, 1])
                conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
                conv_1_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_1_output_reshape2)
                bidirectional_1_output = Bidirectional(
                    LSTM(30, return_sequences=True, dropout=0.25, kernel_initializer=initializer))(
                    Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max]))
                attention_1_output = Attention()([bidirectional_1_output, bidirectional_1_output])
                average_1_output = GlobalAveragePooling1D(data_format='channels_last')(attention_1_output)
                max_1_output = GlobalMaxPool1D(data_format='channels_last')(attention_1_output)
                concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
                flatten_output = Flatten()(concat_output)
                linear_1_output = BatchNormalization()(
                    Dense(200, activation='relu', kernel_initializer=initializer)(flatten_output))
                linear_2_output = Dense(100, activation='relu', kernel_initializer=initializer)(linear_1_output)
                linear_2_output_dropout = Dropout(0.9)(linear_2_output)
                linear_3_output = Dense(num_classes, activation='softmax', kernel_initializer=initializer)(
                    linear_2_output_dropout)
                model = Model(input_value, linear_3_output)
                model.compile(tf.keras.optimizers.Adam(), loss=NCEandMAE(alpha=alpha,beta=beta), metrics=['accuracy'])#Adam是0.001，SGD是0.01
                history_model = model.fit(
                    resampled_ds,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=test_ds,
                    steps_per_epoch=resampled_steps_per_epoch,
                    callbacks=callbacks
                )
                yscore = model.predict(xtest)
                ypred = np.argmax(yscore, axis=1)
                print(ypred)
                yscore = yscore[:, 1]
                # print(yscore)
                gcetest = np.argmax(ytest, axis=1)
                print(gcetest)
                eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score]
                eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
                eval_fun_types = [True, True, True, True, False, False]
                for index_f, function in enumerate(eval_funs):
                    if eval_fun_types[index_f]:
                        score = np.round(function(gcetest, ypred), 4)
                    else:
                        score = np.round(function(gcetest, yscore), 4)
                        if index_f == 5:
                            if score > best_pr and np.round(roc_auc_score(gcetest, yscore), 4) > best_roc:
                                best_pr = score
                                best_roc = np.round(roc_auc_score(gcetest, yscore), 4)
                                best_alpha = alpha
                                best_beta = beta
                                model.save('{}+NCEandMAE_crispr_ip.h5'.format(saved_prefix))
                    print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))
                # model.save('{}+new_crispr_ip.h5'.format(saved_prefix))
    # else:
    #     # print('cunzai')
    #     model = load_model('{}+NCEandMAE_crispr_ip.h5'.format(saved_prefix),custom_objects={'PositionalEncoding': PositionalEncoding})
    # model = load_model('{}+sce_crispr_ip.h5'.format(saved_prefix))
    custom_objects = get_custom_objects()
    my_objects = {'PositionalEncoding': PositionalEncoding,'NCEandMAE':NCEandMAE}
    custom_objects.update(my_objects)
    model = load_model('{}+NCEandMAE_crispr_ip.h5'.format(saved_prefix),
                       custom_objects=custom_objects)
    return model

def sce_crispr_ip_z(test_ds,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, xtest, ytest, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+sce_crispr_ip.h5'.format(saved_prefix)):
        beta_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,2.0,3.0,4.0,5.0]
        alpha_list = [0.01,0.05,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0]
        # alpha_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        # beta_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        # beta_list = [0.1]
        best_pr = 0
        best_roc = 0
        best_beta = 0
        best_alpha = 0
        for alpha in alpha_list:
            for beta in beta_list:
                print(beta)
                initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
                input_value = Input(shape=input_shape)
                conv_1_output = Conv2D(60, (1, input_shape[-1]), padding='valid', data_format='channels_first',
                                       kernel_initializer=initializer)(input_value)
                conv_1_output_reshape = Reshape(
                    tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(
                    conv_1_output)
                conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0, 2, 1])
                conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
                conv_1_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_1_output_reshape2)
                bidirectional_1_output = Bidirectional(
                    LSTM(30, return_sequences=True, dropout=0.25, kernel_initializer=initializer))(
                    Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max]))
                attention_1_output = Attention()([bidirectional_1_output, bidirectional_1_output])
                average_1_output = GlobalAveragePooling1D(data_format='channels_last')(attention_1_output)
                max_1_output = GlobalMaxPool1D(data_format='channels_last')(attention_1_output)
                concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
                flatten_output = Flatten()(concat_output)
                linear_1_output = BatchNormalization()(
                    Dense(200, activation='relu', kernel_initializer=initializer)(flatten_output))
                linear_2_output = Dense(100, activation='relu', kernel_initializer=initializer)(linear_1_output)
                linear_2_output_dropout = Dropout(0.9)(linear_2_output)
                linear_3_output = Dense(num_classes, activation='softmax', kernel_initializer=initializer)(
                    linear_2_output_dropout)
                model = Model(input_value, linear_3_output)
                model.compile(tf.keras.optimizers.Adam(), loss=SCE(alpha=alpha,beta=beta), metrics=['accuracy'])#Adam是0.001，SGD是0.01
                history_model = model.fit(
                    resampled_ds,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=test_ds,
                    steps_per_epoch=resampled_steps_per_epoch,
                    callbacks=callbacks
                )
                yscore = model.predict(xtest)
                ypred = np.argmax(yscore, axis=1)
                print(ypred)
                yscore = yscore[:, 1]
                # print(yscore)
                gcetest = np.argmax(ytest, axis=1)
                print(gcetest)
                eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score]
                eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
                eval_fun_types = [True, True, True, True, False, False]
                for index_f, function in enumerate(eval_funs):
                    if eval_fun_types[index_f]:
                        score = np.round(function(gcetest, ypred), 4)
                    else:
                        score = np.round(function(gcetest, yscore), 4)
                        if index_f == 5:
                            if score > best_pr and np.round(roc_auc_score(gcetest, yscore), 4) > best_roc:
                                best_pr = score
                                best_roc = np.round(roc_auc_score(gcetest, yscore), 4)
                                best_alpha = alpha
                                best_beta = beta
                                model.save('{}+sce_crispr_ip.h5'.format(saved_prefix))
                    print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))
                # model.save('{}+new_crispr_ip.h5'.format(saved_prefix))
    # else:
    #     # print('cunzai')
    #     model = load_model('{}+sce_crispr_ip.h5'.format(saved_prefix),custom_objects={'PositionalEncoding': PositionalEncoding})
    # model = load_model('{}+sce_crispr_ip.h5'.format(saved_prefix))
    custom_objects = get_custom_objects()
    my_objects = {'PositionalEncoding': PositionalEncoding,'SCE':SCE}
    custom_objects.update(my_objects)
    model = load_model('{}+sce_crispr_ip.h5'.format(saved_prefix),
                       custom_objects=custom_objects)
    return model

def gce_crispr_ip_z(test_ds,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, xtest, ytest, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+gce_crispr_ip.h5'.format(saved_prefix)):
        q_list = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        best_pr = 0
        best_roc = 0
        best_q = 0
        for q in q_list:
            print(q)
            initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
            input_value = Input(shape=input_shape)
            conv_1_output = Conv2D(60, (1, input_shape[-1]), padding='valid', data_format='channels_first',
                                   kernel_initializer=initializer)(input_value)
            conv_1_output_reshape = Reshape(
                tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(
                conv_1_output)
            conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0, 2, 1])
            conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
            conv_1_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_1_output_reshape2)
            bidirectional_1_output = Bidirectional(
                LSTM(30, return_sequences=True, dropout=0.25, kernel_initializer=initializer))(
                Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max]))
            attention_1_output = Attention()([bidirectional_1_output, bidirectional_1_output])
            average_1_output = GlobalAveragePooling1D(data_format='channels_last')(attention_1_output)
            max_1_output = GlobalMaxPool1D(data_format='channels_last')(attention_1_output)
            concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
            flatten_output = Flatten()(concat_output)
            linear_1_output = BatchNormalization()(
                Dense(200, activation='relu', kernel_initializer=initializer)(flatten_output))
            linear_2_output = Dense(100, activation='relu', kernel_initializer=initializer)(linear_1_output)
            linear_2_output_dropout = Dropout(0.9)(linear_2_output)
            linear_3_output = Dense(num_classes, activation='softmax', kernel_initializer=initializer)(
                linear_2_output_dropout)
            model = Model(input_value, linear_3_output)
            model.compile(tf.keras.optimizers.Adam(), loss=GCE(q=q), metrics=['accuracy'])#Adam是0.001，SGD是0.01
            history_model = model.fit(
                resampled_ds,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=test_ds,
                steps_per_epoch=resampled_steps_per_epoch,
                callbacks=callbacks
            )
            yscore = model.predict(xtest)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            yscore = yscore[:, 1]
            # print(yscore)
            gcetest = np.argmax(ytest, axis=1)
            print(gcetest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(gcetest, ypred), 4)
                else:
                    score = np.round(function(gcetest, yscore), 4)
                    if index_f == 5:
                        if score > best_pr and np.round(roc_auc_score(gcetest, yscore), 4) > best_roc:
                            best_pr = score
                            best_roc = np.round(roc_auc_score(gcetest, yscore), 4)
                            best_q = q
                            model.save('{}+gce_crispr_ip.h5'.format(saved_prefix))
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))
            # model.save('{}+new_crispr_ip.h5'.format(saved_prefix))
    # else:
    #     # print('cunzai')
    #     model = load_model('{}+gce_crispr_ip.h5'.format(saved_prefix),custom_objects={'PositionalEncoding': PositionalEncoding})
    # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    # model = load_model('{}+gce_crispr_ip.h5'.format(saved_prefix),
    #                    custom_objects={'PositionalEncoding': PositionalEncoding})
    custom_objects = get_custom_objects()
    my_objects = {'PositionalEncoding': PositionalEncoding, 'GCE': GCE}
    custom_objects.update(my_objects)
    model = load_model('{}+gce_crispr_ip.h5'.format(saved_prefix),
                       custom_objects=custom_objects)
    return model

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, sequence_len=None, embedding_dim=None,**kwargs):
        super(PositionalEncoding, self).__init__()
        self.sequence_len = sequence_len
        self.embedding_dim = embedding_dim

    def call(self, x):
        # print('x:')
        # print(x.shape)
        position_embedding = np.array([
            [pos / np.power(10000, 2. * i / self.embedding_dim) for i in range(self.embedding_dim)]
            for pos in range(self.sequence_len)])

        position_embedding[:, 0::2] = np.sin(position_embedding[:, 0::2])  # dim 2i
        position_embedding[:, 1::2] = np.cos(position_embedding[:, 1::2])  # dim 2i+1
        # print('position_embedding')
        # print(position_embedding.shape)
        position_embedding = tf.cast(position_embedding, dtype=tf.float32)
        # print((position_embedding+x).shape)
        return position_embedding+x
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'sequence_len' : self.sequence_len,
            'embedding_dim' : self.embedding_dim,
        })
        return config


# branch_0 = conv2d_bn(inputs, 10, (1, 1))
#         branch_1 = conv2d_bn(inputs, 10, (1, 2))
#         branch_2 = conv2d_bn(inputs, 10, (1, 3))
#         branch_3 = conv2d_bn(inputs, 10, (1, 5))
#         branches = [inputs, branch_0, branch_1, branch_2, branch_3]
#         # branches = [branch_0, branch_1, branch_2, branch_3]
#         mixed = Concatenate(axis=-1)(branches)
#         mixed = Reshape((23, 46))(mixed)
#         blstm_out = Bidirectional(LSTM(15, return_sequences=True, input_shape=(23, 46), name="LSTM_out"))(mixed)
#         # inputs_rs = Reshape((24, 7))(inputs)
#         # blstm_out = layers.Concatenate(axis=-1)([mixed, blstm_out])
#         blstm_out = Flatten()(blstm_out)
def saidao2_new_crispr_ip(test_ds,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, xtest, ytest, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+new_crispr_ip.h5'.format(saved_prefix)):
        # initializer = VarianceScaling(mode='fan_avg', distribution='uniform')  # 初始化器能够根据权值的尺寸调整其规模
        # input_value = Input(shape=input_shape)  # (1,23,14)
        # print(input_shape)
        # print(input_value)
        # conv_1_output = Conv2D(10, (1, 7), padding='same', activation='relu',
        #                        kernel_initializer=initializer)(input_value)
        # conv_1_output_bn = BatchNormalization()(conv_1_output)
        #
        # conv_2_output = Conv2D(10, (1, 2), padding='same',activation='relu',
        #                        kernel_initializer=initializer)(input_value)
        # conv_2_output_bn = BatchNormalization()(conv_2_output)
        #
        # conv_3_output = Conv2D(10, (1, 3), padding='same',activation='relu',
        #                        kernel_initializer=initializer)(input_value)
        # conv_3_output_bn = BatchNormalization()(conv_3_output)
        #
        # conv_4_output = Conv2D(10, (1, 5), padding='same',activation='relu',
        #                        kernel_initializer=initializer)(input_value)
        # conv_4_output_bn = BatchNormalization()(conv_4_output)
        #
        # branches = [input_value, conv_1_output_bn, conv_2_output_bn, conv_3_output_bn, conv_4_output_bn]
        # # branches = [input_value, conv_1_output, conv_2_output, conv_3_output, conv_4_output]
        # mixed = Concatenate(axis=-1)(branches)
        # mixed = Reshape((23, 54))(mixed)
        #
        # bidirectional_1_output = Bidirectional(
        #     LSTM(32, return_sequences=True, input_shape=(23, 54), kernel_initializer=initializer))(mixed)

        initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
        input_value = Input(shape=input_shape)

        flatten_output = Flatten()(input_value)
        linear_1_output = (Dense(256, activation='relu', kernel_initializer=initializer)(flatten_output))
        # linear_1_output = BatchNormalization()(linear_1_output)
        # linear_1_output = Dropout(0.25)(linear_1_output)
        linear_2_output = Dense(64, activation='relu', kernel_initializer=initializer)(linear_1_output)
        linear_2_output_dropout = Dropout(0.25)(linear_2_output)
        linear_3_output = Dense(2, activation='softmax', kernel_initializer=initializer)(linear_2_output_dropout)
        model = Model(input_value, linear_3_output)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])#Adam是0.001，SGD是0.01
        history_model = model.fit(
            resampled_ds,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=test_ds,
            steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        yscore = model.predict(xtest)
        ypred = np.argmax(yscore, axis=1)
        print(ypred)
        yscore = yscore[:, 1]
        # print(yscore)
        ytest = np.argmax(ytest, axis=1)
        print(ytest)
        eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score]
        eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
        eval_fun_types = [True, True, True, True, False, False]
        for index_f, function in enumerate(eval_funs):
            if eval_fun_types[index_f]:
                score = np.round(function(ytest, ypred), 4)
            else:
                score = np.round(function(ytest, yscore), 4)
            print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))
        model.save('{}+new_crispr_ip.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix),custom_objects={'PositionalEncoding': PositionalEncoding,'MultiHeadAttention':MultiHeadAttention})
    # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model



def new_crispr_ip(test_ds,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, xtest, ytest, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+new_crispr_ip.h5'.format(saved_prefix)):
        # initializer = VarianceScaling(mode='fan_avg', distribution='uniform')  # 初始化器能够根据权值的尺寸调整其规模
        # input_value = Input(shape=input_shape)  # (1,23,14)
        # print(input_shape)
        # print(input_value)
        # conv_1_output = Conv2D(10, (1, 7), padding='same', activation='relu',
        #                        kernel_initializer=initializer)(input_value)
        # conv_1_output_bn = BatchNormalization()(conv_1_output)
        #
        # conv_2_output = Conv2D(10, (1, 2), padding='same',activation='relu',
        #                        kernel_initializer=initializer)(input_value)
        # conv_2_output_bn = BatchNormalization()(conv_2_output)
        #
        # conv_3_output = Conv2D(10, (1, 3), padding='same',activation='relu',
        #                        kernel_initializer=initializer)(input_value)
        # conv_3_output_bn = BatchNormalization()(conv_3_output)
        #
        # conv_4_output = Conv2D(10, (1, 5), padding='same',activation='relu',
        #                        kernel_initializer=initializer)(input_value)
        # conv_4_output_bn = BatchNormalization()(conv_4_output)
        #
        # branches = [input_value, conv_1_output_bn, conv_2_output_bn, conv_3_output_bn, conv_4_output_bn]
        # # branches = [input_value, conv_1_output, conv_2_output, conv_3_output, conv_4_output]
        # mixed = Concatenate(axis=-1)(branches)
        # mixed = Reshape((23, 54))(mixed)
        #
        # bidirectional_1_output = Bidirectional(
        #     LSTM(32, return_sequences=True, input_shape=(23, 54), kernel_initializer=initializer))(mixed)

        initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
        input_value = Input(shape=input_shape)
        conv_1_output = Conv2D(64, (1, input_shape[-1]), activation='relu', padding='valid', data_format='channels_first',
                               kernel_initializer=initializer)(input_value)
        conv_1_output = BatchNormalization()(conv_1_output)
        conv_1_output_reshape = Reshape(tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(
            conv_1_output)
        conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0, 2, 1])
        conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
        conv_1_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_1_output_reshape2)
        input_value1 = Reshape((23,input_shape[-1]))(input_value)
        bidirectional_1_output = Bidirectional(
            LSTM(32, return_sequences=True, dropout=0.2,activation='relu',kernel_initializer=initializer))(
            Concatenate(axis=-1)([input_value1,conv_1_output_reshape_average, conv_1_output_reshape_max]))

        # bidirectional_1_output = Bidirectional(
        #     LSTM(32, return_sequences=True, dropout=0.25, kernel_initializer=initializer))(Concatenate(axis=-2)(
        #     [conv_1_output_reshape_average, conv_1_output_reshape_max, conv_2_output_reshape_average,
        #      conv_2_output_reshape_max, conv_3_output_reshape_average, conv_3_output_reshape_max]))

        # bidirectional_1_output = Activation('relu')(bidirectional_1_output)
        bidirectional_1_output_ln = LayerNormalization()(bidirectional_1_output)
        # print(bidirectional_1_output_ln.shape)  # bidirectional_2_output = Dropout(0.9)(bidirectional_1_output)
        # print(bidirectional_1_output.shape)
        # bidirectional_1_output = Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max])
        pos_embedding = PositionalEncoding(sequence_len=23, embedding_dim=64)(bidirectional_1_output_ln)
        # print(pos_embedding.shape)
        attention_1_output = MultiHeadAttention(head_num=8)([pos_embedding, pos_embedding, pos_embedding])
        print(attention_1_output.shape)
        residual1 = attention_1_output + pos_embedding
        print('residual1.shape')
        print(residual1.shape)
        laynorm1 = LayerNormalization()(residual1)
        linear1 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm1)
        linear2 = Dense(64, activation='relu', kernel_initializer=initializer)(linear1)
        residual2 = laynorm1 + linear2
        laynorm2 = LayerNormalization()(residual2)
        print(laynorm2.shape)
        attention_2_output = MultiHeadAttention(head_num=8)([laynorm2, laynorm2, laynorm2])
        residual3 = attention_2_output + laynorm2
        laynorm3 = LayerNormalization()(residual3)
        linear3 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm3)
        linear4 = Dense(64, activation='relu', kernel_initializer=initializer)(linear3)
        residual4 = laynorm3 + linear4
        laynorm4 = LayerNormalization()(residual4)
        print(laynorm4.shape)

        # average_1_output = GlobalAveragePooling1D(data_format='channels_last')(laynorm4)
        # print(average_1_output.shape)
        # max_1_output = GlobalMaxPool1D(data_format='channels_last')(laynorm4)
        # print(max_1_output.shape)
        # concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
        # print(concat_output.shape)
        flatten_output = Flatten()(laynorm4)
        linear_1_output = (Dense(256, activation='relu', kernel_initializer=initializer)(flatten_output))
        # linear_1_output = BatchNormalization()(linear_1_output)
        # linear_1_output = Dropout(0.25)(linear_1_output)
        linear_2_output = Dense(64, activation='relu', kernel_initializer=initializer)(linear_1_output)
        linear_2_output_dropout = Dropout(0.25)(linear_2_output)
        linear_3_output = Dense(2, activation='softmax', kernel_initializer=initializer)(linear_2_output_dropout)
        model = Model(input_value, linear_3_output)
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])#Adam是0.001，SGD是0.01
        history_model = model.fit(
            resampled_ds,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=test_ds,
            steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        yscore = model.predict(xtest)
        ypred = np.argmax(yscore, axis=1)
        print(ypred)
        yscore = yscore[:, 1]
        # print(yscore)
        ytest = np.argmax(ytest, axis=1)
        print(ytest)
        eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score]
        eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
        eval_fun_types = [True, True, True, True, False, False]
        for index_f, function in enumerate(eval_funs):
            if eval_fun_types[index_f]:
                score = np.round(function(ytest, ypred), 4)
            else:
                score = np.round(function(ytest, yscore), 4)
            print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))
        model.save('{}+new_crispr_ip.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix),custom_objects={'PositionalEncoding': PositionalEncoding,'MultiHeadAttention':MultiHeadAttention})
    # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model

def NCEandRCE_new_crispr_ip(test_ds,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, xtest, ytest, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+NCEandRCE_new_crispr_ip.h5'.format(saved_prefix)):
        # alpha_list = [0.1, 1.0, 10.0]
        # beta_list = [0.1,1.0,10.0,100.0]
        alpha_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0]
        beta_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 100.0]
        best_pr = 0
        best_roc = 0
        best_beta = 0
        best_alpha = 0
        for alpha in alpha_list:
            for beta in beta_list:
                print(alpha)
                print(beta)
                initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
                input_value = Input(shape=input_shape)
                conv_1_output = Conv2D(64, (1, input_shape[-1]), activation='relu', padding='valid',
                                       data_format='channels_first',
                                       kernel_initializer=initializer)(input_value)
                conv_1_output = BatchNormalization()(conv_1_output)
                conv_1_output_reshape = Reshape(
                    tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(
                    conv_1_output)
                conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0, 2, 1])
                conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
                conv_1_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_1_output_reshape2)
                input_value1 = Reshape((23, 14))(input_value)
                bidirectional_1_output = Bidirectional(
                    LSTM(32, return_sequences=True, dropout=0.2, activation='relu', kernel_initializer=initializer))(
                    Concatenate(axis=-1)([input_value1, conv_1_output_reshape_average, conv_1_output_reshape_max]))

                # bidirectional_1_output = Bidirectional(
                #     LSTM(32, return_sequences=True, dropout=0.25, kernel_initializer=initializer))(Concatenate(axis=-2)(
                #     [conv_1_output_reshape_average, conv_1_output_reshape_max, conv_2_output_reshape_average,
                #      conv_2_output_reshape_max, conv_3_output_reshape_average, conv_3_output_reshape_max]))

                # bidirectional_1_output = Activation('relu')(bidirectional_1_output)
                bidirectional_1_output_ln = LayerNormalization()(bidirectional_1_output)
                # print(bidirectional_1_output_ln.shape)  # bidirectional_2_output = Dropout(0.9)(bidirectional_1_output)
                # print(bidirectional_1_output.shape)
                # bidirectional_1_output = Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max])
                pos_embedding = PositionalEncoding(sequence_len=23, embedding_dim=64)(bidirectional_1_output_ln)
                # print(pos_embedding.shape)
                attention_1_output = MultiHeadAttention(head_num=8)([pos_embedding, pos_embedding, pos_embedding])
                print(attention_1_output.shape)
                residual1 = attention_1_output + pos_embedding
                print('residual1.shape')
                print(residual1.shape)
                laynorm1 = LayerNormalization()(residual1)
                linear1 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm1)
                linear2 = Dense(64, activation='relu', kernel_initializer=initializer)(linear1)
                residual2 = laynorm1 + linear2
                laynorm2 = LayerNormalization()(residual2)
                print(laynorm2.shape)
                attention_2_output = MultiHeadAttention(head_num=8)([laynorm2, laynorm2, laynorm2])
                residual3 = attention_2_output + laynorm2
                laynorm3 = LayerNormalization()(residual3)
                linear3 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm3)
                linear4 = Dense(64, activation='relu', kernel_initializer=initializer)(linear3)
                residual4 = laynorm3 + linear4
                laynorm4 = LayerNormalization()(residual4)
                print(laynorm4.shape)

                # average_1_output = GlobalAveragePooling1D(data_format='channels_last')(laynorm4)
                # print(average_1_output.shape)
                # max_1_output = GlobalMaxPool1D(data_format='channels_last')(laynorm4)
                # print(max_1_output.shape)
                # concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
                # print(concat_output.shape)
                flatten_output = Flatten()(laynorm4)
                linear_1_output = (Dense(256, activation='relu', kernel_initializer=initializer)(flatten_output))
                # linear_1_output = BatchNormalization()(linear_1_output)
                # linear_1_output = Dropout(0.25)(linear_1_output)
                linear_2_output = Dense(64, activation='relu', kernel_initializer=initializer)(linear_1_output)
                linear_2_output_dropout = Dropout(0.25)(linear_2_output)
                linear_3_output = Dense(2, activation='softmax', kernel_initializer=initializer)(
                    linear_2_output_dropout)
                model = Model(input_value, linear_3_output)
                model.compile(tf.keras.optimizers.Adam(), loss=NCEandRCE(alpha=alpha,beta=beta), metrics=['accuracy'])#Adam是0.001，SGD是0.01
                history_model = model.fit(
                    resampled_ds,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=test_ds,
                    steps_per_epoch=resampled_steps_per_epoch,
                    callbacks=callbacks
                )
                yscore = model.predict(xtest)
                ypred = np.argmax(yscore, axis=1)
                print(ypred)
                yscore = yscore[:, 1]
                # print(yscore)
                gcetest = np.argmax(ytest, axis=1)
                print(gcetest)
                eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score]
                eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
                eval_fun_types = [True, True, True, True, False, False]
                for index_f, function in enumerate(eval_funs):
                    if eval_fun_types[index_f]:
                        score = np.round(function(gcetest, ypred), 4)
                    else:
                        score = np.round(function(gcetest, yscore), 4)
                        if index_f == 5:
                            if score > best_pr and np.round(roc_auc_score(gcetest,yscore),4) > best_roc:
                                best_pr = score
                                best_roc = np.round(roc_auc_score(gcetest,yscore),4)
                                best_alpha = alpha
                                best_beta = beta
                                model.save('{}+NCEandRCE_new_crispr_ip.h5'.format(saved_prefix))
                    print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))
                # model.save('{}+new_crispr_ip.h5'.format(saved_prefix))
    # else:
    #     # print('cunzai')
    #     model = load_model('{}+NCEandRCE_new_crispr_ip.h5'.format(saved_prefix),custom_objects={'PositionalEncoding': PositionalEncoding})
    # model = load_model('{}+sce_crispr_ip.h5'.format(saved_prefix))
    custom_objects = get_custom_objects()
    my_objects = {'PositionalEncoding': PositionalEncoding,'NCEandRCE':NCEandRCE}
    custom_objects.update(my_objects)
    model = load_model('{}+NCEandRCE_new_crispr_ip.h5'.format(saved_prefix),
                       custom_objects=custom_objects)
    return model

def NCEandMAE_new_crispr_ip(test_ds,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, xtest, ytest, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+NCEandMAE_new_crispr_ip.h5'.format(saved_prefix)):
        # alpha_list = [0.1, 1.0, 10.0]
        # beta_list = [0.1,1.0,10.0,100.0]
        alpha_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0]
        beta_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 100.0]
        # beta_list = [0.1]
        best_pr = 0
        best_roc = 0
        best_beta = 0
        best_alpha = 0
        for alpha in alpha_list:
            for beta in beta_list:
                print(alpha)
                print(beta)
                initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
                input_value = Input(shape=input_shape)
                conv_1_output = Conv2D(64, (1, input_shape[-1]), activation='relu', padding='valid',
                                       data_format='channels_first',
                                       kernel_initializer=initializer)(input_value)
                conv_1_output = BatchNormalization()(conv_1_output)
                conv_1_output_reshape = Reshape(
                    tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(
                    conv_1_output)
                conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0, 2, 1])
                conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
                conv_1_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_1_output_reshape2)
                input_value1 = Reshape((23, 14))(input_value)
                bidirectional_1_output = Bidirectional(
                    LSTM(32, return_sequences=True, dropout=0.2, activation='relu', kernel_initializer=initializer))(
                    Concatenate(axis=-1)([input_value1, conv_1_output_reshape_average, conv_1_output_reshape_max]))

                # bidirectional_1_output = Bidirectional(
                #     LSTM(32, return_sequences=True, dropout=0.25, kernel_initializer=initializer))(Concatenate(axis=-2)(
                #     [conv_1_output_reshape_average, conv_1_output_reshape_max, conv_2_output_reshape_average,
                #      conv_2_output_reshape_max, conv_3_output_reshape_average, conv_3_output_reshape_max]))

                # bidirectional_1_output = Activation('relu')(bidirectional_1_output)
                bidirectional_1_output_ln = LayerNormalization()(bidirectional_1_output)
                # print(bidirectional_1_output_ln.shape)  # bidirectional_2_output = Dropout(0.9)(bidirectional_1_output)
                # print(bidirectional_1_output.shape)
                # bidirectional_1_output = Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max])
                pos_embedding = PositionalEncoding(sequence_len=23, embedding_dim=64)(bidirectional_1_output_ln)
                # print(pos_embedding.shape)
                attention_1_output = MultiHeadAttention(head_num=8)([pos_embedding, pos_embedding, pos_embedding])
                print(attention_1_output.shape)
                residual1 = attention_1_output + pos_embedding
                print('residual1.shape')
                print(residual1.shape)
                laynorm1 = LayerNormalization()(residual1)
                linear1 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm1)
                linear2 = Dense(64, activation='relu', kernel_initializer=initializer)(linear1)
                residual2 = laynorm1 + linear2
                laynorm2 = LayerNormalization()(residual2)
                print(laynorm2.shape)
                attention_2_output = MultiHeadAttention(head_num=8)([laynorm2, laynorm2, laynorm2])
                residual3 = attention_2_output + laynorm2
                laynorm3 = LayerNormalization()(residual3)
                linear3 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm3)
                linear4 = Dense(64, activation='relu', kernel_initializer=initializer)(linear3)
                residual4 = laynorm3 + linear4
                laynorm4 = LayerNormalization()(residual4)
                print(laynorm4.shape)

                # average_1_output = GlobalAveragePooling1D(data_format='channels_last')(laynorm4)
                # print(average_1_output.shape)
                # max_1_output = GlobalMaxPool1D(data_format='channels_last')(laynorm4)
                # print(max_1_output.shape)
                # concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
                # print(concat_output.shape)
                flatten_output = Flatten()(laynorm4)
                linear_1_output = (Dense(256, activation='relu', kernel_initializer=initializer)(flatten_output))
                # linear_1_output = BatchNormalization()(linear_1_output)
                # linear_1_output = Dropout(0.25)(linear_1_output)
                linear_2_output = Dense(64, activation='relu', kernel_initializer=initializer)(linear_1_output)
                linear_2_output_dropout = Dropout(0.25)(linear_2_output)
                linear_3_output = Dense(2, activation='softmax', kernel_initializer=initializer)(
                    linear_2_output_dropout)
                model = Model(input_value, linear_3_output)
                model.compile(tf.keras.optimizers.Adam(), loss=NCEandMAE(alpha=alpha,beta=beta), metrics=['accuracy'])#Adam是0.001，SGD是0.01
                history_model = model.fit(
                    resampled_ds,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=test_ds,
                    steps_per_epoch=resampled_steps_per_epoch,
                    callbacks=callbacks
                )
                yscore = model.predict(xtest)
                ypred = np.argmax(yscore, axis=1)
                print(ypred)
                yscore = yscore[:, 1]
                # print(yscore)
                gcetest = np.argmax(ytest, axis=1)
                print(gcetest)
                eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score]
                eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
                eval_fun_types = [True, True, True, True, False, False]
                for index_f, function in enumerate(eval_funs):
                    if eval_fun_types[index_f]:
                        score = np.round(function(gcetest, ypred), 4)
                    else:
                        score = np.round(function(gcetest, yscore), 4)
                        if index_f == 5:
                            if score > best_pr and np.round(roc_auc_score(gcetest, yscore), 4) > best_roc:
                                best_pr = score
                                best_roc = np.round(roc_auc_score(gcetest, yscore), 4)
                                best_alpha = alpha
                                best_beta = beta
                                model.save('{}+NCEandMAE_new_crispr_ip.h5'.format(saved_prefix))
                    print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))
                # model.save('{}+new_crispr_ip.h5'.format(saved_prefix))
    # else:
    #     # print('cunzai')
    #     model = load_model('{}+NCEandMAE_new_crispr_ip.h5'.format(saved_prefix),custom_objects={'PositionalEncoding': PositionalEncoding})
    # model = load_model('{}+sce_crispr_ip.h5'.format(saved_prefix))
    custom_objects = get_custom_objects()
    my_objects = {'PositionalEncoding': PositionalEncoding,'NCEandMAE':NCEandMAE}
    custom_objects.update(my_objects)
    model = load_model('{}+NCEandMAE_new_crispr_ip.h5'.format(saved_prefix),
                       custom_objects=custom_objects)
    return model

def sce_new_crispr_ip(test_ds,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, xtest, ytest, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+sce_new_crispr_ip.h5'.format(saved_prefix)):
        beta_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0]
        alpha_list = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0]
        # beta_list = [0.1]
        best_pr = 0
        best_roc = 0
        best_beta = 0
        best_alpha = 0
        for alpha in alpha_list:
            for beta in beta_list:
                print(alpha)
                print(beta)
                initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
                input_value = Input(shape=input_shape)
                conv_1_output = Conv2D(64, (1, input_shape[-1]), activation='relu', padding='valid',
                                       data_format='channels_first',
                                       kernel_initializer=initializer)(input_value)
                conv_1_output = BatchNormalization()(conv_1_output)
                conv_1_output_reshape = Reshape(
                    tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(
                    conv_1_output)
                conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0, 2, 1])
                conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
                conv_1_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_1_output_reshape2)
                input_value1 = Reshape((23, 14))(input_value)
                bidirectional_1_output = Bidirectional(
                    LSTM(32, return_sequences=True, dropout=0.2, activation='relu', kernel_initializer=initializer))(
                    Concatenate(axis=-1)([input_value1, conv_1_output_reshape_average, conv_1_output_reshape_max]))

                # bidirectional_1_output = Bidirectional(
                #     LSTM(32, return_sequences=True, dropout=0.25, kernel_initializer=initializer))(Concatenate(axis=-2)(
                #     [conv_1_output_reshape_average, conv_1_output_reshape_max, conv_2_output_reshape_average,
                #      conv_2_output_reshape_max, conv_3_output_reshape_average, conv_3_output_reshape_max]))

                # bidirectional_1_output = Activation('relu')(bidirectional_1_output)
                bidirectional_1_output_ln = LayerNormalization()(bidirectional_1_output)
                # print(bidirectional_1_output_ln.shape)  # bidirectional_2_output = Dropout(0.9)(bidirectional_1_output)
                # print(bidirectional_1_output.shape)
                # bidirectional_1_output = Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max])
                pos_embedding = PositionalEncoding(sequence_len=23, embedding_dim=64)(bidirectional_1_output_ln)
                # print(pos_embedding.shape)
                attention_1_output = MultiHeadAttention(head_num=8)([pos_embedding, pos_embedding, pos_embedding])
                print(attention_1_output.shape)
                residual1 = attention_1_output + pos_embedding
                print('residual1.shape')
                print(residual1.shape)
                laynorm1 = LayerNormalization()(residual1)
                linear1 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm1)
                linear2 = Dense(64, activation='relu', kernel_initializer=initializer)(linear1)
                residual2 = laynorm1 + linear2
                laynorm2 = LayerNormalization()(residual2)
                print(laynorm2.shape)
                attention_2_output = MultiHeadAttention(head_num=8)([laynorm2, laynorm2, laynorm2])
                residual3 = attention_2_output + laynorm2
                laynorm3 = LayerNormalization()(residual3)
                linear3 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm3)
                linear4 = Dense(64, activation='relu', kernel_initializer=initializer)(linear3)
                residual4 = laynorm3 + linear4
                laynorm4 = LayerNormalization()(residual4)
                print(laynorm4.shape)

                # average_1_output = GlobalAveragePooling1D(data_format='channels_last')(laynorm4)
                # print(average_1_output.shape)
                # max_1_output = GlobalMaxPool1D(data_format='channels_last')(laynorm4)
                # print(max_1_output.shape)
                # concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
                # print(concat_output.shape)
                flatten_output = Flatten()(laynorm4)
                linear_1_output = (Dense(256, activation='relu', kernel_initializer=initializer)(flatten_output))
                # linear_1_output = BatchNormalization()(linear_1_output)
                # linear_1_output = Dropout(0.25)(linear_1_output)
                linear_2_output = Dense(64, activation='relu', kernel_initializer=initializer)(linear_1_output)
                linear_2_output_dropout = Dropout(0.25)(linear_2_output)
                linear_3_output = Dense(2, activation='softmax', kernel_initializer=initializer)(
                    linear_2_output_dropout)
                model = Model(input_value, linear_3_output)
                model.compile(tf.keras.optimizers.Adam(), loss=SCE(alpha=alpha,beta=beta), metrics=['accuracy'])#Adam是0.001，SGD是0.01
                history_model = model.fit(
                    resampled_ds,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=test_ds,
                    steps_per_epoch=resampled_steps_per_epoch,
                    callbacks=callbacks
                )
                yscore = model.predict(xtest)
                ypred = np.argmax(yscore, axis=1)
                print(ypred)
                yscore = yscore[:, 1]
                # print(yscore)
                gcetest = np.argmax(ytest, axis=1)
                print(gcetest)
                eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score]
                eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
                eval_fun_types = [True, True, True, True, False, False]
                for index_f, function in enumerate(eval_funs):
                    if eval_fun_types[index_f]:
                        score = np.round(function(gcetest, ypred), 4)
                    else:
                        score = np.round(function(gcetest, yscore), 4)
                        if index_f == 5:
                            if score > best_pr and np.round(roc_auc_score(gcetest, yscore), 4) > best_roc:
                                best_pr = score
                                best_roc = np.round(roc_auc_score(gcetest, yscore), 4)
                                best_alpha = alpha
                                best_beta = beta
                                model.save('{}+sce_new_crispr_ip.h5'.format(saved_prefix))
                    print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))
                # model.save('{}+new_crispr_ip.h5'.format(saved_prefix))
    # else:
    #     # print('cunzai')
    #     model = load_model('{}+sce_new_crispr_ip.h5'.format(saved_prefix),custom_objects={'PositionalEncoding': PositionalEncoding})
    # model = load_model('{}+sce_crispr_ip.h5'.format(saved_prefix))
    custom_objects = get_custom_objects()
    my_objects = {'PositionalEncoding': PositionalEncoding,'SCE':SCE}
    custom_objects.update(my_objects)
    model = load_model('{}+sce_new_crispr_ip.h5'.format(saved_prefix),
                       custom_objects=custom_objects)
    return model

def gce_new_crispr_ip(test_ds,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, xtest, ytest, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+gce_new_crispr_ip.h5'.format(saved_prefix)):
        q_list = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        best_pr = 0
        best_roc = 0
        best_q = 0
        for q in q_list:
            print(q)
            initializer = VarianceScaling(mode='fan_avg', distribution='uniform')
            input_value = Input(shape=input_shape)
            conv_1_output = Conv2D(64, (1, input_shape[-1]), activation='relu', padding='valid',
                                   data_format='channels_first',
                                   kernel_initializer=initializer)(input_value)
            conv_1_output = BatchNormalization()(conv_1_output)
            conv_1_output_reshape = Reshape(
                tuple([x for x in conv_1_output.shape.as_list() if x != 1 and x is not None]))(
                conv_1_output)
            conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0, 2, 1])
            conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape2)
            conv_1_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_1_output_reshape2)
            input_value1 = Reshape((23, 14))(input_value)
            bidirectional_1_output = Bidirectional(
                LSTM(32, return_sequences=True, dropout=0.2, activation='relu', kernel_initializer=initializer))(
                Concatenate(axis=-1)([input_value1, conv_1_output_reshape_average, conv_1_output_reshape_max]))

            # bidirectional_1_output = Bidirectional(
            #     LSTM(32, return_sequences=True, dropout=0.25, kernel_initializer=initializer))(Concatenate(axis=-2)(
            #     [conv_1_output_reshape_average, conv_1_output_reshape_max, conv_2_output_reshape_average,
            #      conv_2_output_reshape_max, conv_3_output_reshape_average, conv_3_output_reshape_max]))

            # bidirectional_1_output = Activation('relu')(bidirectional_1_output)
            bidirectional_1_output_ln = LayerNormalization()(bidirectional_1_output)
            # print(bidirectional_1_output_ln.shape)  # bidirectional_2_output = Dropout(0.9)(bidirectional_1_output)
            # print(bidirectional_1_output.shape)
            # bidirectional_1_output = Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max])
            pos_embedding = PositionalEncoding(sequence_len=23, embedding_dim=64)(bidirectional_1_output_ln)
            # print(pos_embedding.shape)
            attention_1_output = MultiHeadAttention(head_num=8)([pos_embedding, pos_embedding, pos_embedding])
            print(attention_1_output.shape)
            residual1 = attention_1_output + pos_embedding
            print('residual1.shape')
            print(residual1.shape)
            laynorm1 = LayerNormalization()(residual1)
            linear1 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm1)
            linear2 = Dense(64, activation='relu', kernel_initializer=initializer)(linear1)
            residual2 = laynorm1 + linear2
            laynorm2 = LayerNormalization()(residual2)
            print(laynorm2.shape)
            attention_2_output = MultiHeadAttention(head_num=8)([laynorm2, laynorm2, laynorm2])
            residual3 = attention_2_output + laynorm2
            laynorm3 = LayerNormalization()(residual3)
            linear3 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm3)
            linear4 = Dense(64, activation='relu', kernel_initializer=initializer)(linear3)
            residual4 = laynorm3 + linear4
            laynorm4 = LayerNormalization()(residual4)
            print(laynorm4.shape)

            # average_1_output = GlobalAveragePooling1D(data_format='channels_last')(laynorm4)
            # print(average_1_output.shape)
            # max_1_output = GlobalMaxPool1D(data_format='channels_last')(laynorm4)
            # print(max_1_output.shape)
            # concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
            # print(concat_output.shape)
            flatten_output = Flatten()(laynorm4)
            linear_1_output = (Dense(256, activation='relu', kernel_initializer=initializer)(flatten_output))
            # linear_1_output = BatchNormalization()(linear_1_output)
            # linear_1_output = Dropout(0.25)(linear_1_output)
            linear_2_output = Dense(64, activation='relu', kernel_initializer=initializer)(linear_1_output)
            linear_2_output_dropout = Dropout(0.25)(linear_2_output)
            linear_3_output = Dense(2, activation='softmax', kernel_initializer=initializer)(linear_2_output_dropout)
            model = Model(input_value, linear_3_output)
            model.compile(tf.keras.optimizers.Adam(), loss=GCE(q=q), metrics=['accuracy'])#Adam是0.001，SGD是0.01
            history_model = model.fit(
                resampled_ds,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=test_ds,
                steps_per_epoch=resampled_steps_per_epoch,
                callbacks=callbacks
            )
            yscore = model.predict(xtest)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            yscore = yscore[:, 1]
            # print(yscore)
            gcetest = np.argmax(ytest, axis=1)
            print(gcetest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(gcetest, ypred), 4)
                else:
                    score = np.round(function(gcetest, yscore), 4)
                    if index_f == 5:
                        if score > best_pr and np.round(roc_auc_score(gcetest, yscore), 4) > best_roc:
                            best_pr = score
                            best_roc = np.round(roc_auc_score(gcetest, yscore), 4)
                            best_q = q
                            model.save('{}+gce_new_crispr_ip.h5'.format(saved_prefix))
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))
            # model.save('{}+new_crispr_ip.h5'.format(saved_prefix))
    # else:
    #     # print('cunzai')
    #     model = load_model('{}+gce_new_crispr_ip.h5'.format(saved_prefix),custom_objects={'PositionalEncoding': PositionalEncoding})
    # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    # model = load_model('{}+gce_crispr_ip.h5'.format(saved_prefix),
    #                    custom_objects={'PositionalEncoding': PositionalEncoding})
    custom_objects = get_custom_objects()
    my_objects = {'PositionalEncoding': PositionalEncoding, 'GCE': GCE}
    custom_objects.update(my_objects)
    model = load_model('{}+gce_new_crispr_ip.h5'.format(saved_prefix),
                       custom_objects=custom_objects)
    return model

# def attention(x, g, TIME_STEPS):
#     """
#     inputs.shape = (batch_size, time_steps, input_dim)
#     """
#     input_dim = int(x.shape[2])
#     x1 = K.permute_dimensions(x, (0, 2, 1))
#     g1 = K.permute_dimensions(g, (0, 2, 1))
#
#     x2 = Reshape((input_dim, TIME_STEPS))(x1)
#     g2 = Reshape((input_dim, TIME_STEPS))(g1)
#
#     x3 = Dense(TIME_STEPS, kernel_initializer=RandomUniform(seed=2020))(x2)
#     g3 = Dense(TIME_STEPS, kernel_initializer=RandomUniform(seed=2020))(g2)
#     x4 = tf.keras.layers.add([x3, g3])
#     a = Dense(TIME_STEPS, activation="softmax", use_bias=False)(x4)
#     a_probs = Permute((2, 1))(a)
#     output_attention_mul = multiply([x, a_probs])
#     return output_attention_mul

def crispr_offt(test_ds,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, xtest, ytest, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+crispr_offt.h5'.format(saved_prefix)):
        input = Input(shape=(23,))
        embedded = Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=MAXLEN)(input)

        conv1 = Conv1D(20, 5, activation="relu", name="conv1")(embedded)
        batchnor1 = BatchNormalization()(conv1)

        conv2 = Conv1D(40, 5, activation="relu", name="conv2")(batchnor1)
        batchnor2 = BatchNormalization()(conv2)

        conv3 = Conv1D(80, 5, activation="relu", name="conv3")(batchnor2)
        batchnor3 = BatchNormalization()(conv3)

        conv11 = Conv1D(80, 9, name="conv11")(batchnor1)
        x = Attention()([conv11, batchnor3])

        flat = Flatten()(x)
        dense1 = Dense(40, activation="relu", name="dense1")(flat)
        drop1 = Dropout(0.2)(dense1)

        dense2 = Dense(20, activation="relu", name="dense2")(drop1)
        drop2 = Dropout(0.2)(dense2)

        output = Dense(2, activation="softmax", name="dense3")(drop2)
        model = Model(inputs=[input], outputs=[output])
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])  # Adam是0.001，SGD是0.01
        history_model = model.fit(
            resampled_ds,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=test_ds,
            steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        yscore = model.predict(xtest)
        ypred = np.argmax(yscore, axis=1)
        print(ypred)
        yscore = yscore[:, 1]
        # print(yscore)
        ytest = np.argmax(ytest, axis=1)
        print(ytest)
        eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score]
        eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
        eval_fun_types = [True, True, True, True, False, False]
        for index_f, function in enumerate(eval_funs):
            if eval_fun_types[index_f]:
                score = np.round(function(ytest, ypred), 4)
            else:
                score = np.round(function(ytest, yscore), 4)
            print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))
        model.save('{}+crispr_offt.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+crispr_offt.h5'.format(saved_prefix),
                           custom_objects={'PositionalEncoding': PositionalEncoding})
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model

def NCEandRCE_crispr_offt(test_ds,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, xtest, ytest, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+NCEandRCE_crispr_offt.h5'.format(saved_prefix)):
        alpha_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0, 2.0,3.0,4.0,5.0,10.0]
        beta_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,2.0,3.0,4.0,5.0,10.0,100.0]
        # alpha_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 10.0]
        # beta_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 10.0]
        best_pr = 0
        best_roc = 0
        best_beta = 0
        best_alpha = 0
        for alpha in alpha_list:
            for beta in beta_list:
                print(alpha)
                print(beta)
                input = Input(shape=(23,))
                embedded = Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=MAXLEN)(input)

                conv1 = Conv1D(20, 5, activation="relu", name="conv1")(embedded)
                batchnor1 = BatchNormalization()(conv1)

                conv2 = Conv1D(40, 5, activation="relu", name="conv2")(batchnor1)
                batchnor2 = BatchNormalization()(conv2)

                conv3 = Conv1D(80, 5, activation="relu", name="conv3")(batchnor2)
                batchnor3 = BatchNormalization()(conv3)

                conv11 = Conv1D(80, 9, name="conv11")(batchnor1)
                x = Attention()([conv11, batchnor3])

                flat = Flatten()(x)
                dense1 = Dense(40, activation="relu", name="dense1")(flat)
                drop1 = Dropout(0.2)(dense1)

                dense2 = Dense(20, activation="relu", name="dense2")(drop1)
                drop2 = Dropout(0.2)(dense2)

                output = Dense(2, activation="softmax", name="dense3")(drop2)
                model = Model(inputs=[input], outputs=[output])
                model.compile(tf.keras.optimizers.Adam(), loss=NCEandRCE(alpha=alpha,beta=beta), metrics=['accuracy'])#Adam是0.001，SGD是0.01
                history_model = model.fit(
                    resampled_ds,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=test_ds,
                    steps_per_epoch=resampled_steps_per_epoch,
                    callbacks=callbacks
                )
                yscore = model.predict(xtest)
                ypred = np.argmax(yscore, axis=1)
                print(ypred)
                yscore = yscore[:, 1]
                # print(yscore)
                gcetest = np.argmax(ytest, axis=1)
                print(gcetest)
                eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score]
                eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
                eval_fun_types = [True, True, True, True, False, False]
                for index_f, function in enumerate(eval_funs):
                    if eval_fun_types[index_f]:
                        score = np.round(function(gcetest, ypred), 4)
                    else:
                        score = np.round(function(gcetest, yscore), 4)
                        if index_f == 5:
                            if score > best_pr and np.round(roc_auc_score(gcetest, yscore), 4) > best_roc:
                                best_pr = score
                                best_roc = np.round(roc_auc_score(gcetest, yscore), 4)
                                best_alpha = alpha
                                best_beta = beta
                                model.save('{}+NCEandRCE_crispr_offt.h5'.format(saved_prefix))
                    print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))
                # model.save('{}+new_crispr_ip.h5'.format(saved_prefix))
    # else:
    #     # print('cunzai')
    #     model = load_model('{}+NCEandRCE_crispr_offt.h5'.format(saved_prefix),custom_objects={'PositionalEncoding': PositionalEncoding})
    # model = load_model('{}+sce_crispr_ip.h5'.format(saved_prefix))
    custom_objects = get_custom_objects()
    my_objects = {'PositionalEncoding': PositionalEncoding,'NCEandRCE':NCEandRCE}
    custom_objects.update(my_objects)
    model = load_model('{}+NCEandRCE_crispr_offt.h5'.format(saved_prefix),
                       custom_objects=custom_objects)
    return model

def NCEandMAE_crispr_offt(test_ds,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, xtest, ytest, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+NCEandMAE_crispr_offt.h5'.format(saved_prefix)):
        # alpha_list = [0.1, 1.0, 10.0]
        # beta_list = [0.1,1.0,10.0,100.0]
        alpha_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0]
        beta_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0,100.0]
        # alpha_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 10.0]
        # beta_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 10.0]
        # beta_list = [0.1]
        best_pr = 0
        best_roc = 0
        best_beta = 0
        best_alpha = 0
        for alpha in alpha_list:
            for beta in beta_list:
                print(alpha)
                print(beta)
                input = Input(shape=(23,))
                embedded = Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=MAXLEN)(input)

                conv1 = Conv1D(20, 5, activation="relu", name="conv1")(embedded)
                batchnor1 = BatchNormalization()(conv1)

                conv2 = Conv1D(40, 5, activation="relu", name="conv2")(batchnor1)
                batchnor2 = BatchNormalization()(conv2)

                conv3 = Conv1D(80, 5, activation="relu", name="conv3")(batchnor2)
                batchnor3 = BatchNormalization()(conv3)

                conv11 = Conv1D(80, 9, name="conv11")(batchnor1)
                x = Attention()([conv11, batchnor3])

                flat = Flatten()(x)
                dense1 = Dense(40, activation="relu", name="dense1")(flat)
                drop1 = Dropout(0.2)(dense1)

                dense2 = Dense(20, activation="relu", name="dense2")(drop1)
                drop2 = Dropout(0.2)(dense2)

                output = Dense(2, activation="softmax", name="dense3")(drop2)
                model = Model(inputs=[input], outputs=[output])
                model.compile(tf.keras.optimizers.Adam(), loss=NCEandMAE(alpha=alpha,beta=beta), metrics=['accuracy'])#Adam是0.001，SGD是0.01
                history_model = model.fit(
                    resampled_ds,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=test_ds,
                    steps_per_epoch=resampled_steps_per_epoch,
                    callbacks=callbacks
                )
                yscore = model.predict(xtest)
                ypred = np.argmax(yscore, axis=1)
                print(ypred)
                yscore = yscore[:, 1]
                # print(yscore)
                gcetest = np.argmax(ytest, axis=1)
                print(gcetest)
                eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score]
                eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
                eval_fun_types = [True, True, True, True, False, False]
                for index_f, function in enumerate(eval_funs):
                    if eval_fun_types[index_f]:
                        score = np.round(function(gcetest, ypred), 4)
                    else:
                        score = np.round(function(gcetest, yscore), 4)
                        if index_f == 5:
                            if score > best_pr and np.round(roc_auc_score(gcetest, yscore), 4) > best_roc:
                                best_pr = score
                                best_roc = np.round(roc_auc_score(gcetest, yscore), 4)
                                best_alpha = alpha
                                best_beta = beta
                                model.save('{}+NCEandMAE_crispr_offt.h5'.format(saved_prefix))
                    print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))
                # model.save('{}+new_crispr_ip.h5'.format(saved_prefix))
    # else:
    #     # print('cunzai')
    #     model = load_model('{}+NCEandMAE_crispr_offt.h5'.format(saved_prefix),custom_objects={'PositionalEncoding': PositionalEncoding})
    # model = load_model('{}+sce_crispr_ip.h5'.format(saved_prefix))
    custom_objects = get_custom_objects()
    my_objects = {'PositionalEncoding': PositionalEncoding,'NCEandMAE':NCEandMAE}
    custom_objects.update(my_objects)
    model = load_model('{}+NCEandMAE_crispr_offt.h5'.format(saved_prefix),
                       custom_objects=custom_objects)
    return model

def sce_crispr_offt(test_ds,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, xtest, ytest, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+sce_crispr_offt.h5'.format(saved_prefix)):
        beta_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,2.0,3.0,4.0,5.0]
        alpha_list = [0.01,0.05,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0]
        # alpha_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        # beta_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        # beta_list = [0.1]
        best_pr = 0
        best_roc = 0
        best_beta = 0
        best_alpha = 0
        for alpha in alpha_list:
            for beta in beta_list:
                print(beta)
                input = Input(shape=(23,))
                embedded = Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=MAXLEN)(input)

                conv1 = Conv1D(20, 5, activation="relu", name="conv1")(embedded)
                batchnor1 = BatchNormalization()(conv1)

                conv2 = Conv1D(40, 5, activation="relu", name="conv2")(batchnor1)
                batchnor2 = BatchNormalization()(conv2)

                conv3 = Conv1D(80, 5, activation="relu", name="conv3")(batchnor2)
                batchnor3 = BatchNormalization()(conv3)

                conv11 = Conv1D(80, 9, name="conv11")(batchnor1)
                x = Attention()([conv11, batchnor3])

                flat = Flatten()(x)
                dense1 = Dense(40, activation="relu", name="dense1")(flat)
                drop1 = Dropout(0.2)(dense1)

                dense2 = Dense(20, activation="relu", name="dense2")(drop1)
                drop2 = Dropout(0.2)(dense2)

                output = Dense(2, activation="softmax", name="dense3")(drop2)
                model = Model(inputs=[input], outputs=[output])
                model.compile(tf.keras.optimizers.Adam(), loss=SCE(alpha=alpha,beta=beta), metrics=['accuracy'])#Adam是0.001，SGD是0.01
                history_model = model.fit(
                    resampled_ds,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=test_ds,
                    steps_per_epoch=resampled_steps_per_epoch,
                    callbacks=callbacks
                )
                yscore = model.predict(xtest)
                ypred = np.argmax(yscore, axis=1)
                print(ypred)
                yscore = yscore[:, 1]
                # print(yscore)
                gcetest = np.argmax(ytest, axis=1)
                print(gcetest)
                eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score]
                eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
                eval_fun_types = [True, True, True, True, False, False]
                for index_f, function in enumerate(eval_funs):
                    if eval_fun_types[index_f]:
                        score = np.round(function(gcetest, ypred), 4)
                    else:
                        score = np.round(function(gcetest, yscore), 4)
                        if index_f == 5:
                            if score > best_pr and np.round(roc_auc_score(gcetest, yscore), 4) > best_roc:
                                best_pr = score
                                best_roc = np.round(roc_auc_score(gcetest, yscore), 4)
                                best_alpha = alpha
                                best_beta = beta
                                model.save('{}+sce_crispr_offt.h5'.format(saved_prefix))
                    print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))
                # model.save('{}+new_crispr_ip.h5'.format(saved_prefix))
    # else:
    #     # print('cunzai')
    #     model = load_model('{}+sce_crispr_offt.h5'.format(saved_prefix),custom_objects={'PositionalEncoding': PositionalEncoding})
    # model = load_model('{}+sce_crispr_ip.h5'.format(saved_prefix))
    custom_objects = get_custom_objects()
    my_objects = {'PositionalEncoding': PositionalEncoding,'SCE':SCE}
    custom_objects.update(my_objects)
    model = load_model('{}+sce_crispr_offt.h5'.format(saved_prefix),
                       custom_objects=custom_objects)
    return model

def gce_crispr_offt(test_ds,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, xtest, ytest, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+gce_crispr_offt.h5'.format(saved_prefix)):
        q_list = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        best_pr = 0
        best_roc = 0
        best_q = 0
        for q in q_list:
            print(q)
            input = Input(shape=(23,))
            embedded = Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=MAXLEN)(input)

            conv1 = Conv1D(20, 5, activation="relu", name="conv1")(embedded)
            batchnor1 = BatchNormalization()(conv1)

            conv2 = Conv1D(40, 5, activation="relu", name="conv2")(batchnor1)
            batchnor2 = BatchNormalization()(conv2)

            conv3 = Conv1D(80, 5, activation="relu", name="conv3")(batchnor2)
            batchnor3 = BatchNormalization()(conv3)

            conv11 = Conv1D(80, 9, name="conv11")(batchnor1)
            x = Attention()([conv11, batchnor3])

            flat = Flatten()(x)
            dense1 = Dense(40, activation="relu", name="dense1")(flat)
            drop1 = Dropout(0.2)(dense1)

            dense2 = Dense(20, activation="relu", name="dense2")(drop1)
            drop2 = Dropout(0.2)(dense2)

            output = Dense(2, activation="softmax", name="dense3")(drop2)
            model = Model(inputs=[input], outputs=[output])
            model.compile(tf.keras.optimizers.Adam(), loss=GCE(q=q), metrics=['accuracy'])#Adam是0.001，SGD是0.01
            history_model = model.fit(
                resampled_ds,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=test_ds,
                steps_per_epoch=resampled_steps_per_epoch,
                callbacks=callbacks
            )
            yscore = model.predict(xtest)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            yscore = yscore[:, 1]
            # print(yscore)
            gcetest = np.argmax(ytest, axis=1)
            print(gcetest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(gcetest, ypred), 4)
                else:
                    score = np.round(function(gcetest, yscore), 4)
                    if index_f == 5:
                        if score > best_pr and np.round(roc_auc_score(gcetest, yscore), 4) > best_roc:
                            best_pr = score
                            best_roc = np.round(roc_auc_score(gcetest, yscore), 4)
                            best_q = q
                            model.save('{}+gce_crispr_offt.h5'.format(saved_prefix))
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))
            # model.save('{}+new_crispr_ip.h5'.format(saved_prefix))
    # else:
    #     # print('cunzai')
    #     model = load_model('{}+gce_crispr_offt.h5'.format(saved_prefix),custom_objects={'PositionalEncoding': PositionalEncoding})
    # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    # model = load_model('{}+gce_crispr_ip.h5'.format(saved_prefix),
    #                    custom_objects={'PositionalEncoding': PositionalEncoding})
    custom_objects = get_custom_objects()
    my_objects = {'PositionalEncoding': PositionalEncoding, 'GCE': GCE}
    custom_objects.update(my_objects)
    model = load_model('{}+gce_crispr_offt.h5'.format(saved_prefix),
                       custom_objects=custom_objects)
    return model

def cnn_std(test_ds,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, xtest, ytest, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+cnn_std.h5'.format(saved_prefix)):
        input = Input(shape=(1, 23, 4))
        conv_1 = Conv2D(10, (1, 1), padding='same', activation='relu')(input)
        conv_2 = Conv2D(10, (1, 2), padding='same', activation='relu')(input)
        conv_3 = Conv2D(10, (1, 3), padding='same', activation='relu')(input)
        conv_4 = Conv2D(10, (1, 5), padding='same', activation='relu')(input)

        conv_output = tensorflow.keras.layers.concatenate([conv_1, conv_2, conv_3, conv_4])

        bn_output = BatchNormalization()(conv_output)

        pooling_output = MaxPool2D(pool_size=(1, 5), strides=None, padding='valid')(bn_output)

        flatten_output = Flatten()(pooling_output)

        x = Dense(100, activation='relu')(flatten_output)
        x = Dense(23, activation='relu')(x)
        x = Dropout(rate=0.15)(x)

        output = Dense(2, activation="softmax")(x)

        model = Model(inputs=[input], outputs=[output])
        model.compile(tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])  # Adam是0.001，SGD是0.01
        history_model = model.fit(
            resampled_ds,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=test_ds,
            steps_per_epoch=resampled_steps_per_epoch,
            callbacks=callbacks
        )
        yscore = model.predict(xtest)
        ypred = np.argmax(yscore, axis=1)
        print(ypred)
        yscore = yscore[:, 1]
        # print(yscore)
        ytest = np.argmax(ytest, axis=1)
        print(ytest)
        eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score]
        eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
        eval_fun_types = [True, True, True, True, False, False]
        for index_f, function in enumerate(eval_funs):
            if eval_fun_types[index_f]:
                score = np.round(function(ytest, ypred), 4)
            else:
                score = np.round(function(ytest, yscore), 4)
            print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))
        model.save('{}+cnn_std.h5'.format(saved_prefix))
    else:
        # print('cunzai')
        model = load_model('{}+cnn_std.h5'.format(saved_prefix),
                           custom_objects={'PositionalEncoding': PositionalEncoding})
        # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    return model

def NCEandRCE_cnn_std(test_ds,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, xtest, ytest, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+NCEandRCE_cnn_std.h5'.format(saved_prefix)):
        alpha_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0, 2.0,3.0,4.0,5.0,10.0]
        beta_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,2.0,3.0,4.0,5.0,10.0,100.0]
        # alpha_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 10.0]
        # beta_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 10.0]
        best_pr = 0
        best_roc = 0
        best_beta = 0
        best_alpha = 0
        for alpha in alpha_list:
            for beta in beta_list:
                print(alpha)
                print(beta)
                input = Input(shape=(1, 23, 4), name='main_input')
                conv_1 = Conv2D(10, (1, 1), padding='same', activation='relu')(input)
                conv_2 = Conv2D(10, (1, 2), padding='same', activation='relu')(input)
                conv_3 = Conv2D(10, (1, 3), padding='same', activation='relu')(input)
                conv_4 = Conv2D(10, (1, 5), padding='same', activation='relu')(input)

                conv_output = tensorflow.keras.layers.concatenate([conv_1, conv_2, conv_3, conv_4])

                bn_output = BatchNormalization()(conv_output)

                pooling_output = MaxPool2D(pool_size=(1, 5), strides=None, padding='valid')(bn_output)

                flatten_output = Flatten()(pooling_output)

                x = Dense(100, activation='relu')(flatten_output)
                x = Dense(23, activation='relu')(x)
                x = Dropout(rate=0.15)(x)

                output = Dense(2, activation="softmax")(x)
                model = Model(inputs=[input], outputs=[output])
                model.compile(tf.keras.optimizers.Adam(), loss=NCEandRCE(alpha=alpha,beta=beta), metrics=['accuracy'])#Adam是0.001，SGD是0.01
                history_model = model.fit(
                    resampled_ds,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=test_ds,
                    steps_per_epoch=resampled_steps_per_epoch,
                    callbacks=callbacks
                )
                yscore = model.predict(xtest)
                ypred = np.argmax(yscore, axis=1)
                print(ypred)
                yscore = yscore[:, 1]
                # print(yscore)
                gcetest = np.argmax(ytest, axis=1)
                print(gcetest)
                eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score]
                eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
                eval_fun_types = [True, True, True, True, False, False]
                for index_f, function in enumerate(eval_funs):
                    if eval_fun_types[index_f]:
                        score = np.round(function(gcetest, ypred), 4)
                    else:
                        score = np.round(function(gcetest, yscore), 4)
                        if index_f == 5:
                            if score > best_pr and np.round(roc_auc_score(gcetest, yscore), 4) > best_roc:
                                best_pr = score
                                best_roc = np.round(roc_auc_score(gcetest, yscore), 4)
                                best_alpha = alpha
                                best_beta = beta
                                model.save('{}+NCEandRCE_cnn_std.h5'.format(saved_prefix))
                    print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))
                # model.save('{}+new_crispr_ip.h5'.format(saved_prefix))
    # else:
    #     # print('cunzai')
    #     model = load_model('{}+NCEandRCE_cnn_std.h5'.format(saved_prefix),custom_objects={'PositionalEncoding': PositionalEncoding})
    # model = load_model('{}+sce_crispr_ip.h5'.format(saved_prefix))
    custom_objects = get_custom_objects()
    my_objects = {'PositionalEncoding': PositionalEncoding,'NCEandRCE':NCEandRCE}
    custom_objects.update(my_objects)
    model = load_model('{}+NCEandRCE_cnn_std.h5'.format(saved_prefix),
                       custom_objects=custom_objects)
    return model

def NCEandMAE_cnn_std(test_ds,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, xtest, ytest, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+NCEandMAE_cnn_std.h5'.format(saved_prefix)):
        # alpha_list = [0.1, 1.0, 10.0]
        # beta_list = [0.1,1.0,10.0,100.0]
        alpha_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0]
        beta_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0,100.0]
        # alpha_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 10.0]
        # beta_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 10.0]
        # beta_list = [0.1]
        best_pr = 0
        best_roc = 0
        best_beta = 0
        best_alpha = 0
        for alpha in alpha_list:
            for beta in beta_list:
                print(alpha)
                print(beta)
                input = Input(shape=(1, 23, 4), name='main_input')
                conv_1 = Conv2D(10, (1, 1), padding='same', activation='relu')(input)
                conv_2 = Conv2D(10, (1, 2), padding='same', activation='relu')(input)
                conv_3 = Conv2D(10, (1, 3), padding='same', activation='relu')(input)
                conv_4 = Conv2D(10, (1, 5), padding='same', activation='relu')(input)

                conv_output = tensorflow.keras.layers.concatenate([conv_1, conv_2, conv_3, conv_4])

                bn_output = BatchNormalization()(conv_output)

                pooling_output = MaxPool2D(pool_size=(1, 5), strides=None, padding='valid')(bn_output)

                flatten_output = Flatten()(pooling_output)

                x = Dense(100, activation='relu')(flatten_output)
                x = Dense(23, activation='relu')(x)
                x = Dropout(rate=0.15)(x)

                output = Dense(2, activation="softmax")(x)
                model = Model(inputs=[input], outputs=[output])
                model.compile(tf.keras.optimizers.Adam(), loss=NCEandMAE(alpha=alpha,beta=beta), metrics=['accuracy'])#Adam是0.001，SGD是0.01
                history_model = model.fit(
                    resampled_ds,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=test_ds,
                    steps_per_epoch=resampled_steps_per_epoch,
                    callbacks=callbacks
                )
                yscore = model.predict(xtest)
                ypred = np.argmax(yscore, axis=1)
                print(ypred)
                yscore = yscore[:, 1]
                # print(yscore)
                gcetest = np.argmax(ytest, axis=1)
                print(gcetest)
                eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score]
                eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
                eval_fun_types = [True, True, True, True, False, False]
                for index_f, function in enumerate(eval_funs):
                    if eval_fun_types[index_f]:
                        score = np.round(function(gcetest, ypred), 4)
                    else:
                        score = np.round(function(gcetest, yscore), 4)
                        if index_f == 5:
                            if score > best_pr and np.round(roc_auc_score(gcetest, yscore), 4) > best_roc:
                                best_pr = score
                                best_roc = np.round(roc_auc_score(gcetest, yscore), 4)
                                best_alpha = alpha
                                best_beta = beta
                                model.save('{}+NCEandMAE_cnn_std.h5'.format(saved_prefix))
                    print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))
                # model.save('{}+new_crispr_ip.h5'.format(saved_prefix))
    # else:
    #     # print('cunzai')
    #     model = load_model('{}+NCEandMAE_cnn_std.h5'.format(saved_prefix),custom_objects={'PositionalEncoding': PositionalEncoding})
    # model = load_model('{}+sce_crispr_ip.h5'.format(saved_prefix))
    custom_objects = get_custom_objects()
    my_objects = {'PositionalEncoding': PositionalEncoding,'NCEandMAE':NCEandMAE}
    custom_objects.update(my_objects)
    model = load_model('{}+NCEandMAE_cnn_std.h5'.format(saved_prefix),
                       custom_objects=custom_objects)
    return model

def sce_cnn_std(test_ds,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, xtest, ytest, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+sce_cnn_std.h5'.format(saved_prefix)):
        beta_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,2.0,3.0,4.0,5.0]
        alpha_list = [0.01,0.05,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0]
        # alpha_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        # beta_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        # beta_list = [0.1]
        best_pr = 0
        best_roc = 0
        best_beta = 0
        best_alpha = 0
        for alpha in alpha_list:
            for beta in beta_list:
                print(beta)
                input = Input(shape=(1, 23, 4), name='main_input')
                conv_1 = Conv2D(10, (1, 1), padding='same', activation='relu')(input)
                conv_2 = Conv2D(10, (1, 2), padding='same', activation='relu')(input)
                conv_3 = Conv2D(10, (1, 3), padding='same', activation='relu')(input)
                conv_4 = Conv2D(10, (1, 5), padding='same', activation='relu')(input)

                conv_output = tensorflow.keras.layers.concatenate([conv_1, conv_2, conv_3, conv_4])

                bn_output = BatchNormalization()(conv_output)

                pooling_output = MaxPool2D(pool_size=(1, 5), strides=None, padding='valid')(bn_output)

                flatten_output = Flatten()(pooling_output)

                x = Dense(100, activation='relu')(flatten_output)
                x = Dense(23, activation='relu')(x)
                x = Dropout(rate=0.15)(x)

                output = Dense(2, activation="softmax")(x)
                model = Model(inputs=[input], outputs=[output])
                model.compile(tf.keras.optimizers.Adam(), loss=SCE(alpha=alpha,beta=beta), metrics=['accuracy'])#Adam是0.001，SGD是0.01
                history_model = model.fit(
                    resampled_ds,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=test_ds,
                    steps_per_epoch=resampled_steps_per_epoch,
                    callbacks=callbacks
                )
                yscore = model.predict(xtest)
                ypred = np.argmax(yscore, axis=1)
                print(ypred)
                yscore = yscore[:, 1]
                # print(yscore)
                gcetest = np.argmax(ytest, axis=1)
                print(gcetest)
                eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score]
                eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
                eval_fun_types = [True, True, True, True, False, False]
                for index_f, function in enumerate(eval_funs):
                    if eval_fun_types[index_f]:
                        score = np.round(function(gcetest, ypred), 4)
                    else:
                        score = np.round(function(gcetest, yscore), 4)
                        if index_f == 5:
                            if score > best_pr and np.round(roc_auc_score(gcetest, yscore), 4) > best_roc:
                                best_pr = score
                                best_roc = np.round(roc_auc_score(gcetest, yscore), 4)
                                best_alpha = alpha
                                best_beta = beta
                                model.save('{}+sce_cnn_std.h5'.format(saved_prefix))
                    print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))
                # model.save('{}+new_crispr_ip.h5'.format(saved_prefix))
    # else:
    #     # print('cunzai')
    #     model = load_model('{}+sce_cnn_std.h5'.format(saved_prefix),custom_objects={'PositionalEncoding': PositionalEncoding})
    # model = load_model('{}+sce_crispr_ip.h5'.format(saved_prefix))
    custom_objects = get_custom_objects()
    my_objects = {'PositionalEncoding': PositionalEncoding,'SCE':SCE}
    custom_objects.update(my_objects)
    model = load_model('{}+sce_cnn_std.h5'.format(saved_prefix),
                       custom_objects=custom_objects)
    return model

def gce_cnn_std(test_ds,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, xtest, ytest, num_classes, batch_size, epochs, callbacks, saved_prefix, retrain=False):
    if retrain or not os.path.exists('{}+gce_cnn_std.h5'.format(saved_prefix)):
        q_list = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        best_pr = 0
        best_roc = 0
        best_q = 0
        for q in q_list:
            print(q)
            input = Input(shape=(1, 23, 4), name='main_input')
            conv_1 = Conv2D(10, (1, 1), padding='same', activation='relu')(input)
            conv_2 = Conv2D(10, (1, 2), padding='same', activation='relu')(input)
            conv_3 = Conv2D(10, (1, 3), padding='same', activation='relu')(input)
            conv_4 = Conv2D(10, (1, 5), padding='same', activation='relu')(input)

            conv_output = tensorflow.keras.layers.concatenate([conv_1, conv_2, conv_3, conv_4])

            bn_output = BatchNormalization()(conv_output)

            pooling_output = MaxPool2D(pool_size=(1, 5), strides=None, padding='valid')(bn_output)

            flatten_output = Flatten()(pooling_output)

            x = Dense(100, activation='relu')(flatten_output)
            x = Dense(23, activation='relu')(x)
            x = Dropout(rate=0.15)(x)

            output = Dense(2, activation="softmax")(x)
            model = Model(inputs=[input], outputs=[output])
            model.compile(tf.keras.optimizers.Adam(), loss=GCE(q=q), metrics=['accuracy'])#Adam是0.001，SGD是0.01
            history_model = model.fit(
                resampled_ds,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=test_ds,
                steps_per_epoch=resampled_steps_per_epoch,
                callbacks=callbacks
            )
            yscore = model.predict(xtest)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            yscore = yscore[:, 1]
            # print(yscore)
            gcetest = np.argmax(ytest, axis=1)
            print(gcetest)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(gcetest, ypred), 4)
                else:
                    score = np.round(function(gcetest, yscore), 4)
                    if index_f == 5:
                        if score > best_pr and np.round(roc_auc_score(gcetest, yscore), 4) > best_roc:
                            best_pr = score
                            best_roc = np.round(roc_auc_score(gcetest, yscore), 4)
                            best_q = q
                            model.save('{}+gce_cnn_std.h5'.format(saved_prefix))
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))
            # model.save('{}+new_crispr_ip.h5'.format(saved_prefix))
    # else:
    #     # print('cunzai')
    #     model = load_model('{}+gce_cnn_std.h5'.format(saved_prefix),custom_objects={'PositionalEncoding': PositionalEncoding})
    # model = load_model('{}+new_crispr_ip.h5'.format(saved_prefix))
    # model = load_model('{}+gce_crispr_ip.h5'.format(saved_prefix),
    #                    custom_objects={'PositionalEncoding': PositionalEncoding})
    custom_objects = get_custom_objects()
    my_objects = {'PositionalEncoding': PositionalEncoding, 'GCE': GCE}
    custom_objects.update(my_objects)
    model = load_model('{}+gce_cnn_std.h5'.format(saved_prefix),
                       custom_objects=custom_objects)
    return model


# lr = 0.001
# lr2 = 0.0005
# momentum = 0.9
# weight_decay = 1e-4
# start_epoch = 0
# epochs = 320
# stage1 = 70
# stage2 = 200
# alpha = 0.1
# beta = 0.4
# lambda1 = 600
# print_freq = 50
# evaluate = 0
#
# best_prec1 = 0
def pencil_crispr_ip(test_ds,resampled_steps_per_epoch,resampled_ds,xtrain, ytrain, xtest, ytest, input_shape, num_classes, batch_size, epochs, callbacks, saved_prefix, datanum,retrain=False):
    beta_list = [3.0]
    print('beta')
    for beta in beta_list:
        print(beta)
        k = 10
        lr = 0.003
        lr1 = 0.003
        lr2 = 0.001
        momentum = 0.9
        weight_decay = 1e-4
        start_epoch = 0
        epochs = 320
        stage1 = 70
        stage2 = 200
        alpha = 0.7
        # beta = 0
        lambda1 = 700
        print_freq = 50
        evaluate = 0

        best_prec1 = 0
        if retrain or not os.path.exists('{}+crispr_ip.h5'.format(saved_prefix)):
            initializer = VarianceScaling(mode='fan_avg', distribution='uniform')  # 初始化器能够根据权值的尺寸调整其规模
            input_value = Input(shape=input_shape)  # (9,23,1)
            print(input_shape)
            print(input_value)
            conv_1_output = Conv2D(64, (input_shape[0], 1), padding='valid', activation='relu',
                                   kernel_initializer=initializer)(input_value)
            conv_1_output_bn = BatchNormalization()(conv_1_output)

            # conv_2_output = Conv2D(64, (input_shape[0], 2), padding='valid',activation='relu',
            #                        kernel_initializer=initializer)(input_value)
            # conv_2_output_bn = BatchNormalization()(conv_2_output)
            #
            # conv_3_output = Conv2D(64, (input_shape[0], 3), padding='valid',activation='relu',
            #                        kernel_initializer=initializer)(input_value)
            # conv_3_output_bn = BatchNormalization()(conv_3_output)

            # conv_4_output = Conv2D(64, (input_shape[0], 4), padding='valid',activation='relu',
            #                        kernel_initializer=initializer)(input_value)
            # conv_4_output_bn = BatchNormalization()(conv_4_output)
            #
            conv_1_output_reshape = Reshape(
                tuple([x for x in conv_1_output_bn.shape.as_list() if x != 1 and x is not None]))(
                conv_1_output_bn)  # 去掉维度=1的维度
            # conv_2_output_reshape = Reshape(tuple([x for x in conv_2_output_bn.shape.as_list() if x != 1 and x is not None]))(
            #     conv_2_output_bn)  # 去掉维度=1的维度
            # conv_3_output_reshape = Reshape(tuple([x for x in conv_3_output_bn.shape.as_list() if x != 1 and x is not None]))(
            #     conv_3_output_bn)  # 去掉维度=1的维度
            # conv_4_output_reshape = Reshape(tuple([x for x in conv_4_output_bn.shape.as_list() if x != 1 and x is not None]))(
            #     conv_4_output_bn)  # 去掉维度=1的维度
            # conv_1_output_reshape = tf.transpose(conv_1_output, perm=[0,2,3,1])#转置运算
            # print(conv_1_output_reshape.shape)
            # conv_2_output = Conv2D(64, (2, 2), activation='relu', padding='valid',
            #                        kernel_initializer=initializer)(conv_1_output_reshape)
            # conv_1_output_reshape2 = tf.transpose(conv_1_output_reshape, perm=[0,2,1])#转置运算
            # print(conv_1_output_reshape2.shape)
            conv_1_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_1_output_reshape)
            print(conv_1_output_reshape_average.shape)
            conv_1_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_1_output_reshape)
            print(conv_1_output_reshape_max.shape)
            print(Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max]).shape)

            # conv_2_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_2_output_reshape)
            # print(conv_2_output_reshape_average.shape)
            # conv_2_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_2_output_reshape)
            # print(conv_2_output_reshape_max.shape)
            # print(Concatenate(axis=-2)([conv_2_output_reshape_average, conv_2_output_reshape_max]).shape)
            #
            # conv_3_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_3_output_reshape)
            # print(conv_3_output_reshape_average.shape)
            # conv_3_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_3_output_reshape)
            # print(conv_3_output_reshape_max.shape)
            # print(Concatenate(axis=-2)([conv_3_output_reshape_average, conv_3_output_reshape_max]).shape)

            # conv_4_output_reshape_average = AveragePooling1D(data_format='channels_first')(conv_4_output_reshape)
            # print(conv_4_output_reshape_average.shape)
            # conv_4_output_reshape_max = MaxPool1D(data_format='channels_first')(conv_4_output_reshape)
            # print(conv_4_output_reshape_max.shape)
            # print(Concatenate(axis=-2)([conv_4_output_reshape_average, conv_4_output_reshape_max]).shape)

            bidirectional_1_output = Bidirectional(
                LSTM(32, return_sequences=True, dropout=0.25, kernel_initializer=initializer))(
                Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max]))

            # bidirectional_1_output = Bidirectional(
            #     LSTM(32, return_sequences=True, dropout=0.25, kernel_initializer=initializer))(Concatenate(axis=-2)(
            #     [conv_1_output_reshape_average, conv_1_output_reshape_max, conv_2_output_reshape_average,
            #      conv_2_output_reshape_max, conv_3_output_reshape_average, conv_3_output_reshape_max]))

            bidirectional_1_output = Activation('relu')(bidirectional_1_output)
            bidirectional_1_output_ln = LayerNormalization()(bidirectional_1_output)
            print(bidirectional_1_output_ln.shape)  # bidirectional_2_output = Dropout(0.9)(bidirectional_1_output)
            # print(bidirectional_1_output.shape)
            # bidirectional_1_output = Concatenate(axis=-1)([conv_1_output_reshape_average, conv_1_output_reshape_max])
            pos_embedding = PositionalEncoding(sequence_len=23, embedding_dim=64)(bidirectional_1_output_ln)
            # print(pos_embedding.shape)
            attention_1_output = MultiHeadAttention(head_num=8)([pos_embedding, pos_embedding, pos_embedding])
            print(attention_1_output.shape)
            residual1 = attention_1_output + bidirectional_1_output_ln
            print('residual1.shape')
            print(residual1.shape)
            laynorm1 = LayerNormalization()(residual1)
            linear1 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm1)
            linear2 = Dense(64, activation='relu', kernel_initializer=initializer)(linear1)
            residual2 = laynorm1 + linear2
            laynorm2 = LayerNormalization()(residual2)
            print(laynorm2.shape)
            attention_2_output = MultiHeadAttention(head_num=8)([laynorm2, laynorm2, laynorm2])
            residual3 = attention_2_output + laynorm2
            laynorm3 = LayerNormalization()(residual3)
            linear3 = Dense(512, activation='relu', kernel_initializer=initializer)(laynorm3)
            linear4 = Dense(64, activation='relu', kernel_initializer=initializer)(linear3)
            residual4 = laynorm3 + linear4
            laynorm4 = LayerNormalization()(residual4)
            print(laynorm4.shape)
            # average_1_output = GlobalAveragePooling1D(data_format='channels_last')(laynorm2)
            # print(average_1_output.shape)
            # max_1_output = GlobalMaxPool1D(data_format='channels_last')(attention_1_output)
            # print(max_1_output.shape)
            # concat_output = Concatenate(axis=-1)([average_1_output, max_1_output])
            # print(concat_output.shape)
            flatten_output = Flatten()(laynorm4)
            linear_1_output = (Dense(256, activation='relu', kernel_initializer=initializer)(flatten_output))
            batch_norm = BatchNormalization()(linear_1_output)
            # batch_norm = Dropout(0.2)(linear_1_output)
            linear_2_output = Dense(64, activation='relu', kernel_initializer=initializer)(batch_norm)
            linear_2_output_dropout = Dropout(0.2)(linear_2_output)
            linear_3_output = Dense(2, kernel_initializer=initializer)(linear_2_output_dropout)
            model = Model(input_value, linear_3_output)


            # Instantiate an optimizer.
            # optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            # scheduler = tf.keras.optimizers.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
            # Instantiate a loss function.
            criterion = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

            checkpoint_dir = "checkpoint.pth.tar"
            modelbest_dir = "model_best.pth.tar"
            y_file = "y.npy"
            # epochs = 3
            for epoch in range(start_epoch, epochs):
                # print('Start of epoch %d' % (epoch,))

                optimizer = adjust_learning_rate(epoch,lr,lr1,lr2,momentum,weight_decay,epochs,stage1,stage2)
                # print(optimizer)

                if os.path.isfile(y_file):
                    y = np.load(y_file)
                else:
                    y = []

                train(k,resampled_ds, resampled_steps_per_epoch,model,criterion, optimizer, epoch, y,datanum,stage1,stage2,alpha,beta,lambda1,print_freq)


            print('test:\n')
            yscore = model.predict(xtest)
            ypred = np.argmax(yscore, axis=1)
            print(ypred)
            yscore = yscore[:, 1]
            # print(yscore)
            ytest1 = np.argmax(ytest, axis=1)
            print(ytest1)
            eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score]
            eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
            eval_fun_types = [True, True, True, True, False, False]
            for index_f, function in enumerate(eval_funs):
                if eval_fun_types[index_f]:
                    score = np.round(function(ytest1, ypred), 4)
                else:
                    score = np.round(function(ytest1, yscore), 4)
                print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

            model.save('{}+pencil_crispr_ip.h5'.format(saved_prefix))
        else:
            # print('cunzai')
            model = load_model('{}+pencil_crispr_ip.h5'.format(saved_prefix),
                               custom_objects={'PositionalEncoding': PositionalEncoding})
        # model = load_model('{}+pencil_crispr_ip.h5'.format(saved_prefix))
    return model


def save_checkpoint(state, is_best, filename='', modelbest = ''):
    tf.save(state, filename)
    if is_best:
        shutil.copyfile(filename, modelbest)

def train(k,train_loader, resampled_steps_per_epoch,model, criterion, optimizer, epoch, y,datanum,stage1,stage2,alpha,beta,lambda1,print_freq):

    new_y = np.zeros([datanum, 2],dtype='float32')

    for i, (input, target, index) in enumerate(train_loader):
        # measure data loading time

        # print(i)
        # print(input[0])
        if i == resampled_steps_per_epoch:
            break
        # print(index[0])
        index = index.numpy()
        # target1 = target.cuda(async=True)#另外，一旦固定了张量或存储，就可以使用异步的GPU副本。只需传递一个额外的async=True参数到cuda()的调用。这可以用于将数据传输与计算重叠。
        input_var = tf.Variable(input)#autograd.Variable 是包的中央类, 它包裹着Tensor, 支持几乎所有Tensor的操作,并附加额外的属性, 在进行操作以后, 通过调用.backward()来计算梯度, 通过.data来访问原始raw data (tensor), 并将变量梯度累加到.grad
        target_var = tf.Variable(target)

        # compute output
        # logsoftmax = tf.nn.log_softmax(logits)
        # softmax = tf.nn.softmax()

        with tf.GradientTape(persistent=True) as tape:  # 梯度记录
            output = model(input_var)


            if epoch < stage1:
                # lc is classification loss
                lc = criterion(target_var,output)
                # init y_tilde, let softmax(y_tilde) is noisy labels
                onehot = target*k#论文需求，where K is a large constant ( K = 10 in our experiments)
                # print(onehot)
                onehot = onehot.numpy()
                new_y[index, :] = onehot
            else:
                yy = y
                yy = yy[index,:]
                # yy = yy.astype(np.float32)
                # print(yy.dtype)
                # yy = torch.FloatTensor(yy)
                # yy = yy.cuda(async = True)
                yy = tf.Variable(yy)
                # print(yy.dtype)
                # obtain label distributions (y_hat)
                last_y_var = tf.nn.softmax(yy)

                # log_last_y_var = tf.cast(tf.math.log(last_y_var), dtype=tf.float32)
                #
                # print(tf.nn.softmax(output).dtype)
                # print(tf.nn.log_softmax(output).dtype)
                # print(tf.compat.v1.log(last_y_var).dtype)

                lc = tf.reduce_mean(tf.nn.softmax(output)*(tf.nn.log_softmax(output)-tf.compat.v1.log(last_y_var)))
                # lo is compatibility loss
                lo = criterion(target_var,last_y_var)
            # le is entropy loss
            le = - tf.reduce_mean(tf.multiply(tf.nn.softmax(output), tf.nn.log_softmax(output)))

            if epoch < stage1:
                loss = lc
            elif epoch < stage2:
                loss = lc + alpha * lo + beta * le
            else:
                loss = lc

        # 计算梯度
        grads = tape.gradient(loss, model.trainable_variables)
        # 更新参数
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))


        if epoch >= stage1 and epoch < stage2:
            # update y_tilde by back-propagation
            yy_grad = tape.gradient(loss, yy)
            yy = yy.numpy()
            yy-=lambda1*yy_grad.numpy()#求导后的值

            new_y[index,:] = yy

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss:.4f}\t'.format(
                epoch, i, resampled_steps_per_epoch, loss=loss))

        del tape

    if epoch < stage2:
        # save y_tilde
        y = new_y
        y_file = "y.npy"
        np.save(y_file,y)

def validate(val_loader, model, criterion):


    for i, (input, target) in enumerate(val_loader):

        output = model(input)
        loss = criterion(output, target)

        if i == 0:
            yscore = output
            ytarget = target
        else:
            yscore = np.concatenate((yscore, output), axis=0)
            ytarget = np.concatenate((ytarget, target), axis=0)

    ypred = np.argmax(yscore, axis=1)
    # print(ypred)
    yscore = yscore[:, 1]
    # print(yscore)
    ytest = np.argmax(ytarget, axis=1)
    # print(ytest)
    eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score]
    eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
    eval_fun_types = [True, True, True, True, False, False]
    for index_f, function in enumerate(eval_funs):
        if eval_fun_types[index_f]:
            score = np.round(function(ytest, ypred), 4)
        else:
            score = np.round(function(ytest, yscore), 4)
        print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))


def adjust_learning_rate(epoch,lr,lr1,lr2,momentum,weight_decay,epochs,stage1,stage2):
    """Sets the learning rate"""
    if epoch < stage1 :
        lrr = lr
    elif epoch < stage2:
        lrr = lr1
    elif epoch < (epochs - stage2)//3 + stage2:
        lrr = lr2
    elif epoch < 2 * (epochs - stage2)//3 + stage2:
        lrr = lr2//10
    else:
        lrr = lr2//100
    return tf.keras.optimizers.Adam(learning_rate=lrr)
