import pandas as pd
import numpy as np
# from codes.encoding import my_encode_on_off_dim
import newnetwork
import tensorflow as tf
import os
import sklearn
import pickle as pkl
from sklearn.model_selection import (train_test_split, GridSearchCV)
from tensorflow.python.keras import backend as K
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, average_precision_score
from imblearn.over_sampling import SMOTE
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from sklearn.model_selection import KFold

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

# import tensorflow as tf
#
# gpu_device_name = tf.test.gpu_device_name()
# print(gpu_device_name)

def loadGlove(inputpath, outputpath=""):
    data_list = []
    wordEmb = {}
    with open(inputpath) as f:
        for line in f:
            # 基本的数据整理
            ll = line.strip().split(',')
            ll[0] = str(int(float(ll[0])))
            data_list.append(ll)

            # 构建wordembeding的选项
            ll_new = [float(i) for i in ll]
            emb = np.array(ll_new[1:], dtype="float32")
            wordEmb[str(int(ll_new[0]))] = emb

    if outputpath != "":
        with open(outputpath) as f:
            for data in data_list:
                f.writelines(' '.join(data))
        # data_list = [float(i) for i in data_list]
    return wordEmb

seed = 123
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

# gpus = tf.config.experimental.list_physical_devices(devices='0', device_type='GPU')
# print(os.environ['CUDA_VISIBLE_DEVICES'])
import random

random.seed(seed)
os.environ['PYTHONHASHSEED']=str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# print(K.image_data_format())

# Incorporating reduced learning and early stopping for NN callback
early_stopping = tf.keras.callbacks.EarlyStopping(#monitor监控的数据接口,verbose是否输出更多的调试信息
    monitor='val_loss', min_delta=0.0001,#0.0001
    patience=10, verbose=0, mode='auto')
callbacks = [early_stopping]

# callbacks = []

# list_dataset = ['k562','crispor','hek293t']
list_dataset = ['1&2&3&4']
# list_dataset = ['Listgarten_22gRNA','Kleinstiver_5gRNA','SITE-Seq_offTarget','Tasi']
# list_dataset = ['SITE-Seq_offTarget']
# list_dataset = ['crispor']
# list_type = ['8x23','14x23']
list_type = ['14x23']
num_classes = 2
epochs = 500
batch_size = 128#64
retrain=True
flpath = '../data/'

# encoder_shape1=(9,23)
# seg_len1, coding_dim1 = encoder_shape1
# encoder_shape2=(10,23)
# seg_len2, coding_dim2 = encoder_shape2
# encoder_shape3=(13,23)
# seg_len3, coding_dim3 = encoder_shape3

flag = 0
if flag == 5:
    # epochs = 100
    # retrain = False
    for dataset in list_dataset:
        if dataset == 'hek293t':
            batch_size = 10000
        else:
            batch_size = 10000
        #
        print('CRISPR_Net')
        # type = '14x23'
        open_name = 'encoded6x23' + dataset + 'withoutTsai.pkl'
        encoder_shape = (23, 6)
        seg_len, coding_dim = encoder_shape

        print('load data!')
        print('load data!')
        print(open_name)

        loaddata = pkl.load(
            open(flpath + open_name, 'rb'),
            encoding='latin1'
        )

        x_train, x_test, y_train, y_test = train_test_split(
            loaddata.images,
            loaddata.target,  # loaddata.target,
            stratify=pd.Series(loaddata.target),
            test_size=0.2,
            shuffle=True,
            random_state=42)

        x_train, x_val, y_train, y_val = train_test_split(
            x_train,
            y_train,  # loaddata.target,
            stratify=pd.Series(y_train),
            test_size=0.2,
            shuffle=True,
            random_state=42)

        neg = 0
        for i in y_train:
            if i == 0:
                neg += 1
        print(neg)

        xtrain, xtest5, ytrain, ytest5,xval,yval, inputshape = newnetwork.CRISPR_Net_transformIO(
            x_train, x_test, y_train, y_test,x_val,y_val, seg_len, coding_dim, num_classes)

        #
        # print(ytrain)
        # print(guideseq_y)

        # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
        # print(train_ds)
        pos_indices = y_train == 1
        # print(pos_indices)
        pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
        pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
        # print(pos_y)
        # print(1)
        # print(pos_y)
        print(len(pos_y))
        print(len(neg_y))

        pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
        # print(pos_ds)
        neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

        resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5],seed=seed)
        resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
        # print(resampled_ds)
        # for features, labels in resampled_ds:
        #     print(labels)
        resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
        print(resampled_steps_per_epoch)

        test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
        test_ds = test_ds.batch(batch_size)

        print('Training!!')
        retrain = False

        CRISPR_Net_model = newnetwork.CRISPR_Net_model(test_ds,xval,yval,resampled_steps_per_epoch, resampled_ds, xtrain, ytrain,
                                                       xtest5,
                                                       ytest5,
                                                       inputshape, num_classes, batch_size, epochs, callbacks,
                                                       open_name, retrain)

        NCEandRCE_model = newnetwork.NCEandRCE_crispr_Net(test_ds,resampled_steps_per_epoch, resampled_ds, xtrain, ytrain,
                                               xtest5,
                                               ytest5,
                                               inputshape, num_classes, batch_size, epochs, callbacks,
                                               open_name, retrain)

        NCEandMAE_model = newnetwork.NCEandMAE_crispr_Net(test_ds,resampled_steps_per_epoch, resampled_ds, xtrain, ytrain,
                                               xtest5,
                                               ytest5,
                                               inputshape, num_classes, batch_size, epochs, callbacks,
                                               open_name, retrain)

        sce_model = newnetwork.sce_crispr_Net(test_ds,resampled_steps_per_epoch, resampled_ds, xtrain, ytrain,
                                               xtest5,
                                               ytest5,
                                               inputshape, num_classes, batch_size, epochs, callbacks,
                                               open_name, retrain)

        retrain = True
        gce_model = newnetwork.gce_crispr_Net(test_ds,resampled_steps_per_epoch, resampled_ds, xtrain, ytrain,
                                               xtest5,
                                               ytest5,
                                               inputshape, num_classes, batch_size, epochs, callbacks,
                                               open_name, retrain)

        models = [CRISPR_Net_model, NCEandRCE_model, NCEandMAE_model, sce_model, gce_model]

        labels = ['CRISPR_Net', 'NCEandRCE_Net', 'NCEandMAE_Net', 'SCE_Net', 'GCE_Net']

        xtests = [xtest5, xtest5, xtest5, xtest5, xtest5]

        ytests = [ytest5, ytest5, ytest5, ytest5, ytest5]

        roc_name = 'roccurve_CRISPR_Net_' + dataset + '.pdf'
        pr_name = 'precisionrecallcurve_CRISPR_Net_' + dataset + '.pdf'

        newnetwork.plotRocCurve(models, labels, xtests, ytests, roc_name)

        newnetwork.plotPrecisionRecallCurve(models, labels, xtests, ytests, pr_name)

if flag == 4:
    # epochs = 100
    # retrain = True
    for dataset in list_dataset:
        if dataset == 'hek293t':
            batch_size = 10000
        else:
            batch_size = 10000

        print('crispr_ip_model')
        encoder_shape = (23,9)
        seg_len, coding_dim = encoder_shape

        open_name = 'encodedposition9x23'+ dataset + 'withoutTsai.pkl'
        # guideseq_name = 'encodedmismatchtype' + type + 'guideseqwithoutTsai.pkl'

        print('load data!')
        print('load data!')
        print(open_name)

        loaddata = pkl.load(
            open(flpath + open_name, 'rb'),
            encoding='latin1'
        )
        # guideseq_loaddata = pkl.load(
        #     open(flpath + guideseq_name, 'rb'),
        #     encoding='latin1'
        # )

        x_train, x_test, y_train, y_test = train_test_split(
            np.array(loaddata.images),
            loaddata.target,  # loaddata.target,
            stratify=pd.Series(loaddata.target),
            test_size=0.2,
            shuffle=True,
            random_state=42)

        x_train, x_val, y_train, y_val = train_test_split(
            x_train,
            y_train,  # loaddata.target,
            stratify=pd.Series(y_train),
            test_size=0.2,
            shuffle=True,
            random_state=42)

        neg = 0
        for i in y_train:
            if i == 0:
                neg += 1
        print(neg)

        # x,y = np.array(loaddata.images),loaddata.target

        # guideseq_x, guideseq_y, inputshape = newnetwork.guideseq_transformIO(guideseq_loaddata.images,
        #                                                                      pd.Series(guideseq_loaddata.target),
        #                                                                      seg_len, coding_dim, num_classes)
        #
        # print(ytrain)
        # print(guideseq_y)

        xtrain, xtest1, ytrain, ytest1, xval, yval, inputshape = newnetwork.transformIO(
            x_train, x_test, y_train, y_test, x_val, y_val, seg_len, coding_dim, num_classes)

        pos_indices = y_train == 1
        # print(pos_indices)
        pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
        pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
        # print(pos_y)
        # print(1)
        # print(pos_y)
        print(len(pos_y))
        print(len(neg_y))

        pos_index = np.array(range(0, len(pos_y)))
        neg_index = np.array(range(len(pos_y), len(pos_y) + len(neg_y)))

        pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y, pos_index)).repeat()
        # print(pos_ds)
        neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y, neg_index)).repeat()

        resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5],seed=seed)
        resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
        # print(resampled_ds)
        # for features, labels in resampled_ds:
        #     print(labels)
        resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
        print(resampled_steps_per_epoch)

        test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
        test_ds = test_ds.batch(batch_size)

        # xtrain1, xtest1, ytrain1, ytest1, inputshape1 = newnetwork.transformIO(
        #     x_train1, x_test1, y_train1, y_test1, seg_len1, coding_dim1, num_classes)
        #
        # xtrain2, xtest2, ytrain2, ytest2, inputshape2 = newnetwork.transformIO(
        #     x_train2, x_test2, y_train2, y_test2, seg_len2, coding_dim2, num_classes)
        #
        # xtrain3, xtest3, ytrain3, ytest3, inputshape3 = newnetwork.transformIO(
        #     x_train3, x_test3, y_train3, y_test3, seg_len3, coding_dim3, num_classes)

        print('Training!!')

        # model = newnetwork.cnn_crispr(xtrain, ytrain, xtest, ytest, inputshape,num_classes, batch_size, epochs, callbacks,
        #                             'example_saved/example', retrain)

        # model = newnetwork.crispr_ip(xtrain, ytrain, xtest, ytest, inputshape,num_classes, batch_size, epochs, callbacks,
        #                             'example_saved/example', retrain)
        # pencil_model = newnetwork.pencil_crispr_ip(resampled_steps_per_epoch,resampled_ds,test_ds,xtrain, ytrain, xtest, ytest,inputshape, num_classes, batch_size, epochs, callbacks,
        #                         open_name,len(xtrain),retrain)
        # model = newnetwork.crispr_ip(resampled_steps_per_epoch, resampled_ds, test_ds, xtrain, ytrain, xtest, ytest,
        #                              inputshape, num_classes, batch_size, epochs, callbacks,
        #                              open_name, retrain)
        retrain = False
        crispr_ip_model = newnetwork.crispr_ip(test_ds, resampled_steps_per_epoch, resampled_ds, xtrain, ytrain,
                                                    xtest1,
                                                    ytest1,
                                                    inputshape, num_classes, batch_size, epochs, callbacks,
                                                    open_name, retrain)
        retrain = True
        NCEandRCE_model = newnetwork.NCEandRCE_crispr_ip_z(test_ds, resampled_steps_per_epoch, resampled_ds, xtrain, ytrain,
                                               xtest1,
                                               ytest1,
                                               inputshape, num_classes, batch_size, epochs, callbacks,
                                               open_name, retrain)

        NCEandMAE_model = newnetwork.NCEandMAE_crispr_ip_z(test_ds, resampled_steps_per_epoch, resampled_ds, xtrain, ytrain,
                                               xtest1,
                                               ytest1,
                                               inputshape, num_classes, batch_size, epochs, callbacks,
                                               open_name, retrain)

        sce_model = newnetwork.sce_crispr_ip_z(test_ds, resampled_steps_per_epoch, resampled_ds, xtrain, ytrain,
                                               xtest1,
                                               ytest1,
                                               inputshape, num_classes, batch_size, epochs, callbacks,
                                               open_name, retrain)

        gce_model = newnetwork.gce_crispr_ip_z(test_ds, resampled_steps_per_epoch, resampled_ds, xtrain, ytrain,
                                               xtest1,
                                               ytest1,
                                               inputshape, num_classes, batch_size, epochs, callbacks,
                                               open_name, retrain)

        models = [crispr_ip_model, NCEandRCE_model, NCEandMAE_model, sce_model, gce_model]

        labels = ['CRISPR_IP', 'NCEandRCE_IP', 'NCEandMAE_IP', 'SCE_IP', 'GCE_IP']

        xtests = [xtest1, xtest1, xtest1, xtest1, xtest1]

        ytests = [ytest1, ytest1, ytest1, ytest1, ytest1]

        roc_name = 'roccurve_crispr_ip_' + dataset + '.pdf'
        pr_name = 'precisionrecallcurve_crispr_ip_' + dataset + '.pdf'

        newnetwork.plotRocCurve(models, labels, xtests, ytests, roc_name)

        newnetwork.plotPrecisionRecallCurve(models, labels, xtests, ytests, pr_name)

if flag == 3:
    # epochs = 100
    retrain = False
    for dataset in list_dataset:
        if dataset == 'hek293t':
            batch_size = 10000
        else:
            batch_size = 10000
        #
        print('CRISPR_Net')
        # type = '14x23'
        open_name = 'encoded6x23' + dataset + 'withoutTsai.pkl'
        encoder_shape = (23, 6)
        seg_len, coding_dim = encoder_shape

        print('load data!')
        print('load data!')
        print(open_name)

        loaddata = pkl.load(
            open(flpath + open_name, 'rb'),
            encoding='latin1'
        )

        x_train, x_test, y_train, y_test = train_test_split(
            loaddata.images,
            loaddata.target,  # loaddata.target,
            stratify=pd.Series(loaddata.target),
            test_size=0.2,
            shuffle=True,
            random_state=42)

        x_train, x_val, y_train, y_val = train_test_split(
            x_train,
            y_train,  # loaddata.target,
            stratify=pd.Series(y_train),
            test_size=0.2,
            shuffle=True,
            random_state=42)

        neg = 0
        for i in y_train:
            if i == 0:
                neg += 1
        print(neg)

        xtrain, xtest5, ytrain, ytest5, xval, yval, inputshape = newnetwork.CRISPR_Net_transformIO(
            x_train, x_test, y_train, y_test, x_val, y_val, seg_len, coding_dim, num_classes)

        #
        # print(ytrain)
        # print(guideseq_y)

        # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
        # print(train_ds)
        pos_indices = y_train == 1
        # print(pos_indices)
        pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
        pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
        # print(pos_y)
        # print(1)
        # print(pos_y)
        print(len(pos_y))
        print(len(neg_y))

        pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
        # print(pos_ds)
        neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

        resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
        resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
        # print(resampled_ds)
        # for features, labels in resampled_ds:
        #     print(labels)
        resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
        print(resampled_steps_per_epoch)

        test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
        test_ds = test_ds.batch(batch_size)

        print('Training!!')

        CRISPR_Net_model = newnetwork.CRISPR_Net_model(test_ds, resampled_steps_per_epoch, resampled_ds,
                                                       xtrain, ytrain,
                                                       xtest5,
                                                       ytest5,
                                                       inputshape, num_classes, batch_size, epochs, callbacks,
                                                       open_name, retrain)
        yscore = CRISPR_Net_model.predict(xtest5)
        # print(yscore)
        ypred = np.argmax(yscore, axis=1)
        # print(ypred)
        yscore = yscore[:, 1]
        # print(yscore)
        ytest = np.argmax(ytest5, axis=1)
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


        print('new_model')
        # type = '14x23'
        open_name = 'encodedmismatchtype14x23' + dataset + 'withoutTsai.pkl'
        encoder_shape = (23, 14)
        seg_len, coding_dim = encoder_shape

        print('load data!')
        print('load data!')
        print(open_name)

        loaddata = pkl.load(
            open(flpath + open_name, 'rb'),
            encoding='latin1'
        )

        x_train, x_test, y_train, y_test = train_test_split(
            loaddata.images,
            loaddata.target,  # loaddata.target,
            stratify=pd.Series(loaddata.target),
            test_size=0.2,
            shuffle=True,
            random_state=42)

        x_train, x_val, y_train, y_val = train_test_split(
            x_train,
            y_train,  # loaddata.target,
            stratify=pd.Series(y_train),
            test_size=0.2,
            shuffle=True,
            random_state=42)

        neg = 0
        for i in y_train:
            if i == 0:
                neg += 1
        print(neg)

        xtrain, xtest4, ytrain, ytest4, xval, yval, inputshape = newnetwork.transformIO(
            x_train, x_test, y_train, y_test, x_val, y_val, seg_len, coding_dim, num_classes)

        #
        # print(ytrain)
        # print(guideseq_y)

        # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
        # print(train_ds)
        pos_indices = y_train == 1
        # print(pos_indices)
        pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
        pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
        # print(pos_y)
        # print(1)
        # print(pos_y)
        print(len(pos_y))
        print(len(neg_y))

        pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
        # print(pos_ds)
        neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

        resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
        resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
        # print(resampled_ds)
        # for features, labels in resampled_ds:
        #     print(labels)
        resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
        print(resampled_steps_per_epoch)

        test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
        test_ds = test_ds.batch(batch_size)

        print('Training!!')

        new_model = newnetwork.new_crispr_ip(test_ds, resampled_steps_per_epoch, resampled_ds, xtrain, ytrain,
                                             xtest4,
                                             ytest4,
                                             inputshape, num_classes, batch_size, epochs, callbacks,
                                             open_name, retrain)

        yscore = new_model.predict(xtest4)
        # print(yscore)
        ypred = np.argmax(yscore, axis=1)
        # print(ypred)
        yscore = yscore[:, 1]
        # print(yscore)
        ytest = np.argmax(ytest4, axis=1)
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

        print('cnn_crispr model')
        print("GloVe model loaded")
        VOCAB_SIZE = 16  # 4**3
        EMBED_SIZE = 100
        glove_inputpath = "../data/keras_GloVeVec_" + dataset + "_5_100_10000.csv"
        # load GloVe model
        model_glove = loadGlove(glove_inputpath)
        embedding_weights = np.zeros((VOCAB_SIZE, EMBED_SIZE))  # 词语的数量×嵌入的维度
        for i in range(VOCAB_SIZE):
            embedding_weights[i, :] = model_glove[str(i)]

        open_name = 'encoded_CnnCrispr_' + dataset + 'withoutTsai.pkl'
        # guideseq_name = 'encodedmismatchtype' + type + 'guideseqwithoutTsai.pkl'

        print('load data!')
        print('load data!')
        print(open_name)

        loaddata = pkl.load(
            open(flpath + open_name, 'rb'),
            encoding='latin1'
        )

        x_train, x_test, y_train, y_test = train_test_split(
            np.array(loaddata.images),
            loaddata.target,  # loaddata.target,
            stratify=pd.Series(loaddata.target),
            test_size=0.2,
            shuffle=True,
            random_state=42)

        x_train, x_val, y_train, y_val = train_test_split(
            x_train,
            y_train,  # loaddata.target,
            stratify=pd.Series(y_train),
            test_size=0.2,
            shuffle=True,
            random_state=42)

        neg = 0
        for i in y_train:
            if i == 0:
                neg += 1
        print(neg)

        xtrain, xtest6, ytrain, ytest6, xval, yval = newnetwork.offt_transformIO(x_train, x_test, y_train, y_test,
                                                                                 x_val, y_val, num_classes)

        pos_indices = y_train == 1
        # print(pos_indices)
        pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
        pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
        # print(pos_y)
        # print(1)
        # print(pos_y)
        print(len(pos_y))
        print(len(neg_y))

        pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
        # print(pos_ds)
        neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

        resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
        resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
        # print(resampled_ds)
        # for features, labels in resampled_ds:
        #     print(labels)
        resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
        print(resampled_steps_per_epoch)

        test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
        test_ds = test_ds.batch(batch_size)

        print('Training!!')

        CnnCrispr_model = newnetwork.CnnCrispr(embedding_weights, test_ds, resampled_steps_per_epoch, resampled_ds,
                                               xtrain, ytrain,
                                               xtest6,
                                               ytest6, num_classes, batch_size, epochs, callbacks,
                                               open_name, retrain)

        yscore = CnnCrispr_model.predict(xtest6)
        # print(yscore)
        ypred = np.argmax(yscore, axis=1)
        # print(ypred)
        yscore = yscore[:, 1]
        # print(yscore)
        ytest = np.argmax(ytest6, axis=1)
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

        print('crispr_ip_model')
        encoder_shape = (23, 9)
        seg_len, coding_dim = encoder_shape

        open_name = 'encodedposition9x23' + dataset + 'withoutTsai.pkl'
        # guideseq_name = 'encodedmismatchtype' + type + 'guideseqwithoutTsai.pkl'

        print('load data!')
        print('load data!')
        print(open_name)

        loaddata = pkl.load(
            open(flpath + open_name, 'rb'),
            encoding='latin1'
        )
        # guideseq_loaddata = pkl.load(
        #     open(flpath + guideseq_name, 'rb'),
        #     encoding='latin1'
        # )

        x_train, x_test, y_train, y_test = train_test_split(
            np.array(loaddata.images),
            loaddata.target,  # loaddata.target,
            stratify=pd.Series(loaddata.target),
            test_size=0.2,
            shuffle=True,
            random_state=42)

        x_train, x_val, y_train, y_val = train_test_split(
            x_train,
            y_train,  # loaddata.target,
            stratify=pd.Series(y_train),
            test_size=0.2,
            shuffle=True,
            random_state=42)

        neg = 0
        for i in y_train:
            if i == 0:
                neg += 1
        print(neg)

        # x,y = np.array(loaddata.images),loaddata.target

        # guideseq_x, guideseq_y, inputshape = newnetwork.guideseq_transformIO(guideseq_loaddata.images,
        #                                                                      pd.Series(guideseq_loaddata.target),
        #                                                                      seg_len, coding_dim, num_classes)
        #
        # print(ytrain)
        # print(guideseq_y)

        xtrain, xtest1, ytrain, ytest1, xval, yval, inputshape = newnetwork.transformIO(
            x_train, x_test, y_train, y_test, x_val, y_val, seg_len, coding_dim, num_classes)

        pos_indices = y_train == 1
        # print(pos_indices)
        pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
        pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
        # print(pos_y)
        # print(1)
        # print(pos_y)
        print(len(pos_y))
        print(len(neg_y))

        pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
        # print(pos_ds)
        neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

        resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
        resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
        # print(resampled_ds)
        # for features, labels in resampled_ds:
        #     print(labels)
        resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
        print(resampled_steps_per_epoch)

        test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
        test_ds = test_ds.batch(batch_size)

        # xtrain1, xtest1, ytrain1, ytest1, inputshape1 = newnetwork.transformIO(
        #     x_train1, x_test1, y_train1, y_test1, seg_len1, coding_dim1, num_classes)
        #
        # xtrain2, xtest2, ytrain2, ytest2, inputshape2 = newnetwork.transformIO(
        #     x_train2, x_test2, y_train2, y_test2, seg_len2, coding_dim2, num_classes)
        #
        # xtrain3, xtest3, ytrain3, ytest3, inputshape3 = newnetwork.transformIO(
        #     x_train3, x_test3, y_train3, y_test3, seg_len3, coding_dim3, num_classes)

        print('Training!!')

        crispr_ip_model = newnetwork.crispr_ip(test_ds, resampled_steps_per_epoch, resampled_ds, xtrain, ytrain,
                                               xtest1,
                                               ytest1,
                                               inputshape, num_classes, batch_size, epochs, callbacks,
                                               open_name, retrain)

        yscore = crispr_ip_model.predict(xtest1)
        # print(yscore)
        ypred = np.argmax(yscore, axis=1)
        # print(ypred)
        yscore = yscore[:, 1]
        # print(yscore)
        ytest = np.argmax(ytest1, axis=1)
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

        print('cnn_std')
        encoder_shape = (23, 4)
        seg_len, coding_dim = encoder_shape
        open_name = 'encoded4x23' + dataset + 'withoutTsai.pkl'
        # guideseq_name = 'encodedmismatchtype' + type + 'guideseqwithoutTsai.pkl'

        print('load data!')
        print('load data!')
        print(open_name)

        loaddata = pkl.load(
            open(flpath + open_name, 'rb'),
            encoding='latin1'
        )
        # guideseq_loaddata = pkl.load(
        #     open(flpath + guideseq_name, 'rb'),
        #     encoding='latin1'
        # )

        x_train, x_test, y_train, y_test = train_test_split(
            np.array(loaddata.images),
            loaddata.target,  # loaddata.target,
            stratify=pd.Series(loaddata.target),
            test_size=0.2,
            shuffle=True,
            random_state=42)

        x_train, x_val, y_train, y_val = train_test_split(
            x_train,
            y_train,  # loaddata.target,
            stratify=pd.Series(y_train),
            test_size=0.2,
            shuffle=True,
            random_state=42)

        neg = 0
        for i in y_train:
            if i == 0:
                neg += 1
        print(neg)

        xtrain, xtest2, ytrain, ytest2, xval, yval, input_shape = newnetwork.cnn_std_transformIO(x_train, x_test,
                                                                                                 y_train,
                                                                                                 y_test, x_val, y_val,
                                                                                                 seg_len, coding_dim,
                                                                                                 num_classes)
        pos_indices = y_train == 1
        # print(pos_indices)
        pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
        pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
        # print(pos_y)
        # print(1)
        # print(pos_y)
        print(len(pos_y))
        print(len(neg_y))

        pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
        # print(pos_ds)
        neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

        resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
        resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
        # print(resampled_ds)
        # for features, labels in resampled_ds:
        #     print(labels)
        resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
        print(resampled_steps_per_epoch)

        test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
        test_ds = test_ds.batch(batch_size)

        print('Training!!')

        cnn_std_model = newnetwork.cnn_std(test_ds, resampled_steps_per_epoch, resampled_ds, xtrain, ytrain,
                                           xtest2,
                                           ytest2, num_classes, batch_size, epochs, callbacks,
                                           open_name, retrain)

        yscore = cnn_std_model.predict(xtest2)
        # print(yscore)
        ypred = np.argmax(yscore, axis=1)
        # print(ypred)
        yscore = yscore[:, 1]
        # print(yscore)
        ytest = np.argmax(ytest2, axis=1)
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


        print('offt_model')
        open_name = 'encoded_offt_' + dataset + 'withoutTsai.pkl'
        # guideseq_name = 'encodedmismatchtype' + type + 'guideseqwithoutTsai.pkl'

        print('load data!')
        print('load data!')
        print(open_name)

        loaddata = pkl.load(
            open(flpath + open_name, 'rb'),
            encoding='latin1'
        )
        # guideseq_loaddata = pkl.load(
        #     open(flpath + guideseq_name, 'rb'),
        #     encoding='latin1'
        # )

        x_train, x_test, y_train, y_test = train_test_split(
            np.array(loaddata.images),
            loaddata.target,  # loaddata.target,
            stratify=pd.Series(loaddata.target),
            test_size=0.2,
            shuffle=True,
            random_state=42)

        x_train, x_val, y_train, y_val = train_test_split(
            x_train,
            y_train,  # loaddata.target,
            stratify=pd.Series(y_train),
            test_size=0.2,
            shuffle=True,
            random_state=42)

        neg = 0
        for i in y_train:
            if i == 0:
                neg += 1
        print(neg)

        xtrain, xtest3, ytrain, ytest3, xval, yval = newnetwork.offt_transformIO(x_train, x_test, y_train, y_test,
                                                                                 x_val, y_val, num_classes)

        pos_indices = y_train == 1
        # print(pos_indices)
        pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
        pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
        # print(pos_y)
        # print(1)
        # print(pos_y)
        print(len(pos_y))
        print(len(neg_y))

        pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
        # print(pos_ds)
        neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

        resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5], seed=seed)
        resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
        # print(resampled_ds)
        # for features, labels in resampled_ds:
        #     print(labels)
        resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
        print(resampled_steps_per_epoch)

        test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
        test_ds = test_ds.batch(batch_size)

        print('Training!!')

        offt_model = newnetwork.crispr_offt(test_ds, resampled_steps_per_epoch, resampled_ds, xtrain, ytrain,
                                            xtest3,
                                            ytest3, num_classes, batch_size, epochs, callbacks,
                                            open_name, retrain)

        yscore = offt_model.predict(xtest3)
        # print(yscore)
        ypred = np.argmax(yscore, axis=1)
        # print(ypred)
        yscore = yscore[:, 1]
        # print(yscore)
        ytest = np.argmax(ytest3, axis=1)
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

        models = [new_model, crispr_ip_model, cnn_std_model, offt_model, CRISPR_Net_model, CnnCrispr_model]

        labels = ['CrisprDNT','CRISPR_IP', 'CNN_std', 'CRISPR-OFFT', 'CRISPR_Net', 'CnnCrispr']

        xtests = [xtest4, xtest1, xtest2, xtest3,  xtest5, xtest6]

        ytests = [ytest4, ytest1, ytest2, ytest3, ytest5, ytest6]

        roc_name = 'roccurve_compare_' + dataset + '.pdf'
        pr_name = 'precisionrecallcurve_compare_' + dataset + '.pdf'

        newnetwork.plotRocCurve(models, labels, xtests, ytests, roc_name)

        newnetwork.plotPrecisionRecallCurve(models, labels, xtests, ytests, pr_name)

if flag == 2:
    # epochs = 60
    print('cnn_std')
    encoder_shape = (23,4)
    seg_len, coding_dim = encoder_shape
    for dataset in list_dataset:
        if dataset == 'hek293t':
            batch_size = 512
        else:
            batch_size = 512

        open_name = 'encoded4x23'+ dataset + 'withoutTsai.pkl'
        # guideseq_name = 'encodedmismatchtype' + type + 'guideseqwithoutTsai.pkl'

        print('load data!')
        print('load data!')
        print(open_name)

        loaddata = pkl.load(
            open(flpath + open_name, 'rb'),
            encoding='latin1'
        )
        # guideseq_loaddata = pkl.load(
        #     open(flpath + guideseq_name, 'rb'),
        #     encoding='latin1'
        # )

        x_train, x_test, y_train, y_test = train_test_split(
            np.array(loaddata.images),
            loaddata.target,  # loaddata.target,
            stratify=pd.Series(loaddata.target),
            test_size=0.2,
            shuffle=True,
            random_state=42)

        x_train, x_val, y_train, y_val = train_test_split(
            x_train,
            y_train,  # loaddata.target,
            stratify=pd.Series(y_train),
            test_size=0.2,
            shuffle=True,
            random_state=42)

        neg = 0
        for i in y_train:
            if i == 0:
                neg += 1
        print(neg)

        # x,y = np.array(loaddata.images),loaddata.target

        # guideseq_x, guideseq_y, inputshape = newnetwork.guideseq_transformIO(guideseq_loaddata.images,
        #                                                                      pd.Series(guideseq_loaddata.target),
        #                                                                      seg_len, coding_dim, num_classes)
        #
        # print(ytrain)
        # print(guideseq_y)

        xtrain,xtest,ytrain,ytest,xval,yval,input_shape = newnetwork.cnn_std_transformIO(x_train, x_test, y_train, y_test, x_val,y_val,seg_len,coding_dim, num_classes)

        pos_indices = y_train == 1
        # print(pos_indices)
        pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
        pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
        # print(pos_y)
        # print(1)
        # print(pos_y)
        print(len(pos_y))
        print(len(neg_y))

        pos_index = np.array(range(0, len(pos_y)))
        neg_index = np.array(range(len(pos_y), len(pos_y) + len(neg_y)))

        pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y, pos_index)).repeat()
        # print(pos_ds)
        neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y, neg_index)).repeat()

        resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5])
        resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
        # print(resampled_ds)
        # for features, labels in resampled_ds:
        #     print(labels)
        resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
        print(resampled_steps_per_epoch)

        test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
        test_ds = test_ds.batch(batch_size)

        # xtrain1, xtest1, ytrain1, ytest1, inputshape1 = newnetwork.transformIO(
        #     x_train1, x_test1, y_train1, y_test1, seg_len1, coding_dim1, num_classes)
        #
        # xtrain2, xtest2, ytrain2, ytest2, inputshape2 = newnetwork.transformIO(
        #     x_train2, x_test2, y_train2, y_test2, seg_len2, coding_dim2, num_classes)
        #
        # xtrain3, xtest3, ytrain3, ytest3, inputshape3 = newnetwork.transformIO(
        #     x_train3, x_test3, y_train3, y_test3, seg_len3, coding_dim3, num_classes)

        print('Training!!')

        # model = newnetwork.cnn_crispr(xtrain, ytrain, xtest, ytest, inputshape,num_classes, batch_size, epochs, callbacks,
        #                             'example_saved/example', retrain)

        # model = newnetwork.crispr_ip(xtrain, ytrain, xtest, ytest, inputshape,num_classes, batch_size, epochs, callbacks,
        #                             'example_saved/example', retrain)
        # pencil_model = newnetwork.pencil_crispr_ip(resampled_steps_per_epoch,resampled_ds,test_ds,xtrain, ytrain, xtest, ytest,inputshape, num_classes, batch_size, epochs, callbacks,
        #                         open_name,len(xtrain),retrain)
        # model = newnetwork.crispr_ip(resampled_steps_per_epoch, resampled_ds, test_ds, xtrain, ytrain, xtest, ytest,
        #                              inputshape, num_classes, batch_size, epochs, callbacks,
        #                              open_name, retrain)

        cnn_std_model = newnetwork.cnn_std(test_ds, resampled_steps_per_epoch, resampled_ds, xtrain, ytrain,
                                             xtest,
                                             ytest, num_classes, batch_size, epochs, callbacks,
                                             open_name, retrain)

        NCEandRCE_model = newnetwork.NCEandRCE_cnn_std(test_ds, resampled_steps_per_epoch, resampled_ds, xtrain,
                                                         ytrain,
                                                         xtest,
                                                         ytest,
                                                         num_classes, batch_size, epochs, callbacks,
                                                         open_name, retrain)

        NCEandMAE_model = newnetwork.NCEandMAE_cnn_std(test_ds, resampled_steps_per_epoch, resampled_ds, xtrain,
                                                         ytrain,
                                                         xtest,
                                                         ytest,
                                                         num_classes, batch_size, epochs, callbacks,
                                                         open_name, retrain)

        sce_model = newnetwork.sce_cnn_std(test_ds, resampled_steps_per_epoch, resampled_ds, xtrain, ytrain,
                                             xtest,
                                             ytest,
                                             num_classes, batch_size, epochs, callbacks,
                                             open_name, retrain)

        gce_model = newnetwork.gce_cnn_std(test_ds, resampled_steps_per_epoch, resampled_ds, xtrain, ytrain,
                                             xtest,
                                             ytest,
                                             num_classes, batch_size, epochs, callbacks,
                                             open_name, retrain)

        # pencil_model = newnetwork.pencil_crispr_ip(test_ds,resampled_steps_per_epoch, resampled_ds, xtrain, ytrain, xtest,
        #                                      ytest,
        #                                      inputshape, num_classes, batch_size, epochs, callbacks,
        #                                      open_name, len(xtrain),retrain)

        models = [cnn_std_model, NCEandRCE_model, NCEandMAE_model, sce_model, gce_model]

        labels = ['cnn_std_model', 'NCEandRCE_model', 'NCEandMAE_model', 'sce_model', 'gce_model']

        xtests = [xtest, xtest, xtest, xtest, xtest]

        ytests = [ytest, ytest, ytest, ytest, ytest]

        roc_name = 'roccurve_cnn_std_' + dataset + '.pdf'
        pr_name = 'precisionrecallcurve_cnn_std_' + dataset + '.pdf'

        newnetwork.plotRocCurve(models, labels, xtests, ytests, roc_name)

        newnetwork.plotPrecisionRecallCurve(models, labels, xtests, ytests, pr_name)

if flag == 1:
    # epochs = 60
    print('crispr-offt')
    for dataset in list_dataset:
        if dataset == 'hek293t':
            batch_size = 512
        else:
            batch_size = 512

        open_name = 'encoded_offt_'+ dataset + 'withoutTsai.pkl'
        # guideseq_name = 'encodedmismatchtype' + type + 'guideseqwithoutTsai.pkl'

        print('load data!')
        print('load data!')
        print(open_name)

        loaddata = pkl.load(
            open(flpath + open_name, 'rb'),
            encoding='latin1'
        )
        # guideseq_loaddata = pkl.load(
        #     open(flpath + guideseq_name, 'rb'),
        #     encoding='latin1'
        # )

        x_train, x_test, y_train, y_test = train_test_split(
            np.array(loaddata.images),
            loaddata.target,  # loaddata.target,
            stratify=pd.Series(loaddata.target),
            test_size=0.2,
            shuffle=True,
            random_state=42)

        x_train, x_val, y_train, y_val = train_test_split(
            x_train,
            y_train,  # loaddata.target,
            stratify=pd.Series(y_train),
            test_size=0.2,
            shuffle=True,
            random_state=42)

        neg = 0
        for i in y_train:
            if i == 0:
                neg += 1
        print(neg)

        # x,y = np.array(loaddata.images),loaddata.target

        # guideseq_x, guideseq_y, inputshape = newnetwork.guideseq_transformIO(guideseq_loaddata.images,
        #                                                                      pd.Series(guideseq_loaddata.target),
        #                                                                      seg_len, coding_dim, num_classes)
        #
        # print(ytrain)
        # print(guideseq_y)

        xtrain,xtest,xval,ytrain,ytest,yval = newnetwork.offt_transformIO(x_train,x_test,x_val,y_train,y_test,y_val,num_classes)

        pos_indices = y_train == 1
        # print(pos_indices)
        pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
        pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
        # print(pos_y)
        # print(1)
        # print(pos_y)
        print(len(pos_y))
        print(len(neg_y))

        pos_index = np.array(range(0, len(pos_y)))
        neg_index = np.array(range(len(pos_y), len(pos_y) + len(neg_y)))

        pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y, pos_index)).repeat()
        # print(pos_ds)
        neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y, neg_index)).repeat()

        resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5])
        resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
        # print(resampled_ds)
        # for features, labels in resampled_ds:
        #     print(labels)
        resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
        print(resampled_steps_per_epoch)

        test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
        test_ds = test_ds.batch(batch_size)

        # xtrain1, xtest1, ytrain1, ytest1, inputshape1 = newnetwork.transformIO(
        #     x_train1, x_test1, y_train1, y_test1, seg_len1, coding_dim1, num_classes)
        #
        # xtrain2, xtest2, ytrain2, ytest2, inputshape2 = newnetwork.transformIO(
        #     x_train2, x_test2, y_train2, y_test2, seg_len2, coding_dim2, num_classes)
        #
        # xtrain3, xtest3, ytrain3, ytest3, inputshape3 = newnetwork.transformIO(
        #     x_train3, x_test3, y_train3, y_test3, seg_len3, coding_dim3, num_classes)

        print('Training!!')

        # model = newnetwork.cnn_crispr(xtrain, ytrain, xtest, ytest, inputshape,num_classes, batch_size, epochs, callbacks,
        #                             'example_saved/example', retrain)

        # model = newnetwork.crispr_ip(xtrain, ytrain, xtest, ytest, inputshape,num_classes, batch_size, epochs, callbacks,
        #                             'example_saved/example', retrain)
        # pencil_model = newnetwork.pencil_crispr_ip(resampled_steps_per_epoch,resampled_ds,test_ds,xtrain, ytrain, xtest, ytest,inputshape, num_classes, batch_size, epochs, callbacks,
        #                         open_name,len(xtrain),retrain)
        # model = newnetwork.crispr_ip(resampled_steps_per_epoch, resampled_ds, test_ds, xtrain, ytrain, xtest, ytest,
        #                              inputshape, num_classes, batch_size, epochs, callbacks,
        #                              open_name, retrain)

        offt_model = newnetwork.crispr_offt(test_ds, resampled_steps_per_epoch, resampled_ds, xtrain, ytrain,
                                             xtest,
                                             ytest, num_classes, batch_size, epochs, callbacks,
                                             open_name, retrain)

        NCEandRCE_model = newnetwork.NCEandRCE_crispr_offt(test_ds, resampled_steps_per_epoch, resampled_ds, xtrain,
                                                         ytrain,
                                                         xtest,
                                                         ytest,
                                                         num_classes, batch_size, epochs, callbacks,
                                                         open_name, retrain)

        NCEandMAE_model = newnetwork.NCEandMAE_crispr_offt(test_ds, resampled_steps_per_epoch, resampled_ds, xtrain,
                                                         ytrain,
                                                         xtest,
                                                         ytest,
                                                         num_classes, batch_size, epochs, callbacks,
                                                         open_name, retrain)

        sce_model = newnetwork.sce_crispr_offt(test_ds, resampled_steps_per_epoch, resampled_ds, xtrain, ytrain,
                                             xtest,
                                             ytest,
                                             num_classes, batch_size, epochs, callbacks,
                                             open_name, retrain)

        gce_model = newnetwork.gce_crispr_offt(test_ds, resampled_steps_per_epoch, resampled_ds, xtrain, ytrain,
                                             xtest,
                                             ytest,
                                             num_classes, batch_size, epochs, callbacks,
                                             open_name, retrain)

        # pencil_model = newnetwork.pencil_crispr_ip(test_ds,resampled_steps_per_epoch, resampled_ds, xtrain, ytrain, xtest,
        #                                      ytest,
        #                                      inputshape, num_classes, batch_size, epochs, callbacks,
        #                                      open_name, len(xtrain),retrain)

        models = [offt_model, NCEandRCE_model, NCEandMAE_model, sce_model, gce_model]

        labels = ['offt_model', 'NCEandRCE_model', 'NCEandMAE_model', 'sce_model', 'gce_model']

        xtests = [xtest, xtest, xtest, xtest, xtest]

        ytests = [ytest, ytest, ytest, ytest, ytest]

        roc_name = 'roccurve_offt_' + dataset + '.pdf'
        pr_name = 'precisionrecallcurve_offt_' + dataset + '.pdf'

        newnetwork.plotRocCurve(models, labels, xtests, ytests, roc_name)

        newnetwork.plotPrecisionRecallCurve(models, labels, xtests, ytests, pr_name)

        # print(open_name+'predict guideseq:\n')
        # print('crispr_ip model:\n')
        # yscore = model.predict(guideseq_x)
        # # print(len(yscore))
        # ypred = np.argmax(yscore, axis=1)
        # # print(ypred)
        # # print(len(ypred))
        # yscore = yscore[:, 1]
        # # print(yscore)
        # guideseq_y = np.argmax(guideseq_y, axis=1)
        # # print(guideseq_y)
        # # print(len(guideseq_y))
        # eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score]
        # eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
        # eval_fun_types = [True, True, True, True, False, False]
        # for index_f, function in enumerate(eval_funs):
        #     if eval_fun_types[index_f]:
        #         score = np.round(function(guideseq_y, ypred), 4)
        #     else:
        #         score = np.round(function(guideseq_y, yscore), 4)
        #     print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

        # print('new_crispr_ip model:\n')
        # yscore = new_model.predict(guideseq_x)
        # # print(yscore)
        # ypred = np.argmax(yscore, axis=1)
        # # print(ypred)
        # yscore = yscore[:, 1]
        # # print(yscore)
        # # guideseq_y = np.argmax(guideseq_y, axis=1)
        # # print(ytest)
        # eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score]
        # eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
        # eval_fun_types = [True, True, True, True, False, False]
        # for index_f, function in enumerate(eval_funs):
        #     if eval_fun_types[index_f]:
        #         score = np.round(function(guideseq_y, ypred), 4)
        #     else:
        #         score = np.round(function(guideseq_y, yscore), 4)
        #     print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))
        # model = newnetwork.multi_cnn(xtrain1, ytrain1, xtest1, ytest1, inputshape1, xtrain2, ytrain2, xtest2, ytest2, inputshape2,xtrain3, ytrain3, xtest3, ytest3, inputshape3,num_classes, batch_size, epochs, callbacks,
        #                             'example_saved/example', retrain)
        # model = newnetwork.new_multi_cnn(xtrain1, ytrain1, xtest1, ytest1, inputshape1,xtrain2, ytrain2, xtest2, ytest2, inputshape2, xtrain3, ytrain3, xtest3, ytest3, inputshape3,num_classes, batch_size, epochs, callbacks,
        #                             'example_saved/example', retrain)
if flag == 0:
    retrain = False
    # epochs = 1
    for dataset in list_dataset:
        if dataset == 'hek293t':
            batch_size = 10000
        else:
            batch_size = 10000
        # if type == '8x23':
        #     open_name = 'encoded'+type+dataset+'withoutTsai.pkl'
        #     guideseq_name = 'encoded'+type+'guideseqwithoutTsai.pkl'
        #     encoder_shape = (23,8)
        #     seg_len, coding_dim = encoder_shape
        # else:
        #     open_name = 'encodedmismatchtype'+type+dataset+'withoutTsai.pkl'
        #     guideseq_name = 'encodedmismatchtype'+type+'guideseqwithoutTsai.pkl'
        #     encoder_shape = (23,14)
        #     seg_len, coding_dim = encoder_shape
        #
        # print('load data!')
        # print('load data!')
        # print(open_name)
        #
        # loaddata = pkl.load(
        #             open(flpath+open_name,'rb'),
        #             encoding='latin1'
        #         )
        # guideseq_loaddata = pkl.load(
        #     open(flpath + guideseq_name, 'rb'),
        #     encoding='latin1'
        # )
        #
        # x_train, x_test, y_train, y_test = train_test_split(
        # loaddata.images,
        # loaddata.target, #loaddata.target,
        # stratify=pd.Series(loaddata.target),
        # test_size=0.2,
        # shuffle=True,
        # random_state=42)
        #
        # x_train, x_val, y_train, y_val = train_test_split(
        # x_train,
        # y_train, #loaddata.target,
        # stratify=pd.Series(y_train),
        # test_size=0.2,
        # shuffle=True,
        # random_state=42)
        #
        # neg = 0
        # for i in y_train:
        #     if i == 0:
        #         neg += 1
        # print(neg)
        #
        # xtrain, xtest, ytrain, ytest,xval, yval, inputshape = newnetwork.transformIO(
        # x_train, x_test, y_train, y_test, x_val,y_val,seg_len, coding_dim, num_classes)
        #
        # # guideseq_x,guideseq_y,inputshape = newnetwork.guideseq_transformIO(guideseq_loaddata.images,pd.Series(guideseq_loaddata.target),seg_len, coding_dim, num_classes)
        # #
        # # print(ytrain)
        # # print(guideseq_y)
        #
        #
        # # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
        # # print(train_ds)
        # pos_indices = y_train == 1
        # # print(pos_indices)
        # pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
        # pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
        # # print(pos_y)
        # # print(1)
        # # print(pos_y)
        # print(len(pos_y))
        # print(len(neg_y))
        #
        #
        #
        # # pos_index = np.array(range(0,len(pos_y)))
        # # neg_index = np.array(range(len(pos_y),len(pos_y)+len(neg_y)))
        #
        # pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
        # # print(pos_ds)
        # neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()
        #
        # resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5])
        # resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
        # # print(resampled_ds)
        # # for features, labels in resampled_ds:
        # #     print(labels)
        # resampled_steps_per_epoch = np.ceil(2*neg/batch_size)
        # print(resampled_steps_per_epoch)
        #
        # test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
        # test_ds = test_ds.batch(batch_size)
        #
        #
        # print('Training!!')
        #
        #
        # new_model = newnetwork.new_crispr_ip(test_ds, resampled_steps_per_epoch, resampled_ds, xtrain, ytrain,
        #                                         xtest,
        #                                         ytest,
        #                                         inputshape, num_classes, batch_size, epochs, callbacks,
        #                                         open_name, retrain)
        print('new_model')
        # type = '14x23'
        open_name = 'encodedmismatchtype14x23' + dataset + 'withoutTsai.pkl'
        encoder_shape = (23, 14)
        seg_len, coding_dim = encoder_shape

        print('load data!')
        print('load data!')
        print(open_name)

        loaddata = pkl.load(
            open(flpath + open_name, 'rb'),
            encoding='latin1'
        )

        x_train, x_test, y_train, y_test = train_test_split(
            loaddata.images,
            loaddata.target,  # loaddata.target,
            stratify=pd.Series(loaddata.target),
            test_size=0.2,
            shuffle=True,
            random_state=42)

        x_train, x_val, y_train, y_val = train_test_split(
            x_train,
            y_train,  # loaddata.target,
            stratify=pd.Series(y_train),
            test_size=0.2,
            shuffle=True,
            random_state=42)

        neg = 0
        for i in y_train:
            if i == 0:
                neg += 1
        print(neg)

        xtrain, xtest, ytrain, ytest, xval, yval, inputshape = newnetwork.transformIO(
            x_train, x_test, y_train, y_test, x_val, y_val, seg_len, coding_dim, num_classes)

        #
        # print(ytrain)
        # print(guideseq_y)

        # train_ds = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
        # print(train_ds)
        pos_indices = y_train == 1
        # print(pos_indices)
        pos_x, neg_x = xtrain[pos_indices], xtrain[~pos_indices]
        pos_y, neg_y = ytrain[pos_indices], ytrain[~pos_indices]
        # print(pos_y)
        # print(1)
        # print(pos_y)
        print(len(pos_y))
        print(len(neg_y))

        pos_ds = tf.data.Dataset.from_tensor_slices((pos_x, pos_y)).repeat()
        # print(pos_ds)
        neg_ds = tf.data.Dataset.from_tensor_slices((neg_x, neg_y)).repeat()

        resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5],seed=seed)
        resampled_ds = resampled_ds.batch(batch_size).prefetch(2)
        # print(resampled_ds)
        # for features, labels in resampled_ds:
        #     print(labels)
        resampled_steps_per_epoch = np.ceil(2 * neg / batch_size)
        print(resampled_steps_per_epoch)

        test_ds = tf.data.Dataset.from_tensor_slices((xval, yval)).cache()
        test_ds = test_ds.batch(batch_size)

        print('Training!!')
        print(inputshape)

        retrain = False

        new_model = newnetwork.new_crispr_ip(test_ds, resampled_steps_per_epoch, resampled_ds, xtrain, ytrain,
                                             xtest,
                                             ytest,
                                             inputshape, num_classes, batch_size, epochs, callbacks,
                                             open_name, retrain)

        NCEandRCE_model = newnetwork.NCEandRCE_new_crispr_ip(test_ds, resampled_steps_per_epoch, resampled_ds, xtrain, ytrain,
                                             xtest,
                                             ytest,
                                             inputshape, num_classes, batch_size, epochs, callbacks,
                                             open_name, retrain)
        retrain = True

        NCEandMAE_model = newnetwork.NCEandMAE_new_crispr_ip(test_ds, resampled_steps_per_epoch, resampled_ds, xtrain, ytrain,
                                             xtest,
                                             ytest,
                                             inputshape, num_classes, batch_size, epochs, callbacks,
                                             open_name, retrain)

        # retrain = True

        sce_model = newnetwork.sce_new_crispr_ip(test_ds, resampled_steps_per_epoch, resampled_ds, xtrain, ytrain,
                                             xtest,
                                             ytest,
                                             inputshape, num_classes, batch_size, epochs, callbacks,
                                             open_name, retrain)

        # lsr_model = newnetwork.lsr_crispr_ip(test_ds, resampled_steps_per_epoch, resampled_ds, xtrain, ytrain,
        #                                      xtest,
        #                                      ytest,
        #                                      inputshape, num_classes, batch_size, epochs, callbacks,
        #                                      open_name, retrain)

        gce_model = newnetwork.gce_new_crispr_ip(test_ds, resampled_steps_per_epoch, resampled_ds, xtrain, ytrain,
                                             xtest,
                                             ytest,
                                             inputshape, num_classes, batch_size, epochs, callbacks,
                                             open_name, retrain)

        # pencil_model = newnetwork.pencil_crispr_ip(test_ds,resampled_steps_per_epoch, resampled_ds, xtrain, ytrain, xtest,
        #                                      ytest,
        #                                      inputshape, num_classes, batch_size, epochs, callbacks,
        #                                      open_name, len(xtrain),retrain)

        models = [new_model,NCEandRCE_model,NCEandMAE_model,sce_model,gce_model]

        labels = ['CrisprDNT','NCEandRCE_DNT','NCEandMAE_DNT','SCE_DNT','GCE_DNT']

        xtests = [xtest,xtest,xtest,xtest,xtest]

        ytests = [ytest,ytest,ytest,ytest,ytest]

        roc_name = 'roccurve_newcrisprip_' + dataset + '.pdf'
        pr_name = 'precisionrecallcurve_newcrisprip_' + dataset + '.pdf'

        newnetwork.plotRocCurve(models,labels,xtests,ytests,roc_name)

        newnetwork.plotPrecisionRecallCurve(models, labels, xtests, ytests, pr_name)

        # y_pred_list = []
        # for i in range(len(xtests)):
        #     yscore = models[i].predict(xtests[i])
        #     # print(yscore)
        #     ypred = np.argmax(yscore, axis=1)
        #     y_pred_list.append(ypred)
        #     # print(ypred)
        #     yscore = yscore[:, 1]
        #     # print(yscore)
        #     ytest = np.argmax(ytests[i], axis=1)
        #     # print(ytest)
        #     eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
        #                  average_precision_score]
        #     eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
        #     eval_fun_types = [True, True, True, True, False, False]
        #     for index_f, function in enumerate(eval_funs):
        #         if eval_fun_types[index_f]:
        #             score = np.round(function(ytest, ypred), 4)
        #         else:
        #             score = np.round(function(ytest, yscore), 4)
        #         print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))
        #
        # data_save = {'test': ytest, 'CrisprDNT': y_pred_list[0], 'NCEandRCE_DNT': y_pred_list[1], 'NCEandMAE_DNT': y_pred_list[2],
        #              'SCE_DNT': y_pred_list[3], 'GCE_DNT': y_pred_list[4]}
        #
        # data_save = pd.DataFrame(data_save)
        #
        # data_save.to_csv('roccurve_newcrisprip_' + dataset +'.csv', index=None)


        # new_model = newnetwork.smote_crispr_ip(xtrain, ytrain, xval,yval,xtest,
        #                                      ytest,
        #                                      inputshape, num_classes, batch_size, epochs, callbacks,
        #                                      open_name, retrain)

        # model = newnetwork.crispr_ip(resampled_steps_per_epoch, resampled_ds, test_ds, xtrain, ytrain, xtest, ytest,
        #                              inputshape, num_classes, batch_size, epochs, callbacks,
        #                              open_name, retrain)
        #
        # print(open_name+'predict guideseq:\n')
        # print('crispr_ip model:\n')
        # yscore = model.predict(guideseq_x)
        # # print(len(yscore))
        # ypred = np.argmax(yscore, axis=1)
        # # print(ypred)
        # # print(len(ypred))
        # yscore = yscore[:, 1]
        # # print(yscore)
        # guideseq_y = np.argmax(guideseq_y, axis=1)
        # # print(guideseq_y)
        # # print(len(guideseq_y))
        # eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score]
        # eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
        # eval_fun_types = [True, True, True, True, False, False]
        # for index_f, function in enumerate(eval_funs):
        #     if eval_fun_types[index_f]:
        #         score = np.round(function(guideseq_y, ypred), 4)
        #     else:
        #         score = np.round(function(guideseq_y, yscore), 4)
        #     print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

        # print('new_crispr_ip model:\n')
        # yscore = new_model.predict(guideseq_x)
        # # print(yscore)
        # ypred = np.argmax(yscore, axis=1)
        # # print(ypred)
        # yscore = yscore[:, 1]
        # # print(yscore)
        # guideseq_y = np.argmax(guideseq_y, axis=1)
        # # print(ytest)
        # eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score]
        # eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
        # eval_fun_types = [True, True, True, True, False, False]
        # for index_f, function in enumerate(eval_funs):
        #     if eval_fun_types[index_f]:
        #         score = np.round(function(guideseq_y, ypred), 4)
        #     else:
        #         score = np.round(function(guideseq_y, yscore), 4)
        #     print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))
        # model = newnetwork.multi_cnn(xtrain1, ytrain1, xtest1, ytest1, inputshape1, xtrain2, ytrain2, xtest2, ytest2, inputshape2,xtrain3, ytrain3, xtest3, ytest3, inputshape3,num_classes, batch_size, epochs, callbacks,
        #                             'example_saved/example', retrain)
        # model = newnetwork.new_multi_cnn(xtrain1, ytrain1, xtest1, ytest1, inputshape1,xtrain2, ytrain2, xtest2, ytest2, inputshape2, xtrain3, ytrain3, xtest3, ytest3, inputshape3,num_classes, batch_size, epochs, callbacks,
        #                             'example_saved/example', retrain)
print('End of the training!!')