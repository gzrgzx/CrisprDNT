# 导入gensim库
# from gensim.models import Word2Vec
import csv

import numpy as np
import pandas as pd
# from gensim.models import KeyedVectors
import pickle as pkl
from sklearn.utils import Bunch

# 词典构建及数据预处理
# from util import util as use
# from util import timingTool
import numpy as np
import xlrd
from mittens import GloVe
import numpy as np
import gc
import os

from tensorflow.keras import backend as K
K.clear_session()

os.environ['CUDA_VISIBLE_DEVICES'] = '1,0'

# tic = timingTool()
# 共现矩阵的计算
def countCOOC(cooccurrence, window, coreIndex):
    for index in range(len(window)):
        if index == coreIndex:
            continue
        else:
            cooccurrence[window[coreIndex]][window[index]] = cooccurrence[window[coreIndex]][window[index]] + 1
    return cooccurrence

flpath = '../data/'
loaddata = pkl.load(
				open(flpath+'encoded8x23CIRCLE-seqwithoutTsai.pkl','rb'),
				encoding='latin1'
			)

images = loaddata.images
target = loaddata.target

# data = pd.read_csv(flpath+'keras_GloVeVec_5_100_10000.csv')

# columnlist = list(data)
#
# data = data.values
#
# datavalue = []
# datavalue.append(columnlist)

# for i in range(len(data)):
#     datavalue.append(data[i])

# MATCH_ROW_NUMBER1 = {"AA": 0, "AC": 1, "AG": 2, "AT": 3, "CA": 4, "CC": 5, "CG": 6, "CT": 7, "GA": 8,
#                     "GC": 9, "GG": 10, "GT": 11, "TA": 12, "TC": 13, "TG": 14, "TT": 15}

MATCH_ROW_NUMBER1 = {"AA": 1, "AC": 2, "AG": 3, "AT": 4, "CA": 5, "CC": 6, "CG": 7, "CT": 8, "GA": 9,
                    "GC": 10, "GG": 11, "GT": 12, "TA": 13, "TC": 14, "TG": 15, "TT": 16}
crispor = []
alphabet = 'AGCT'
print(target[0])
for i in range(images.shape[0]):
    arr = []
    arr.append(target[i])
    for j in range(images.shape[2]):
        temp = ''
        indexlist = []
        templist = list(images[i,:,j])
        for index,num in enumerate(templist):
            if num != 0:
                indexlist.append(index)
        # print(indexlist)
        temp = temp + alphabet[indexlist[0]]
        temp = temp + alphabet[indexlist[1]-4]
        arr.append(MATCH_ROW_NUMBER1[temp]-1)
        if i==0:
            print(temp)
    crispor.append(arr)

print(crispor[0])

tableSize = 16
coWindow = 5
vecLength = 100  # The length of the matrix
max_iter = 10000  # Maximum number of iterations
display_progress = 1000
cooccurrence = np.zeros((tableSize, tableSize), "int64")
print("An empty table had been created.")
print(cooccurrence.shape)

# Start statistics
flag = 0
for item in crispor:
    itemInt = [int(x) for x in item]
    for core in range(1, len(item)):
        if core <= coWindow + 1:
            window = itemInt[1:core + coWindow + 1]
            coreIndex = core - 1
            cooccurrence = countCOOC(cooccurrence, window, coreIndex)

        elif core >= len(item) - 1 - coWindow:
            window = itemInt[core - coWindow:(len(item))]
            coreIndex = coWindow
            cooccurrence = countCOOC(cooccurrence, window, coreIndex)

        else:
            window = itemInt[core - coWindow:core + coWindow + 1]
            coreIndex = coWindow
            cooccurrence = countCOOC(cooccurrence, window, coreIndex)

    flag = flag + 1
    if flag % 20 == 0:
        print("%s pieces of data have been calculated" % flag)
# print("The calculation of co-occurrence matrix was completed, taking %s" % (tic.timmingGet()))

# del crispor, window
gc.collect()

# Display of statistical results
# nowTime = tic.getNow().strftime('%Y%m%d_%H%M%S')
coocPath = "../data/cooccurrence_CIRCLE-seq_%s.csv" % (coWindow)

f = open(coocPath,'w',newline='')

writer = csv.writer(f)
for item in cooccurrence:
    writer.writerow(item)
# print("The co-occurrence matrix is derived, taking %s" % (tic.timmingGet()))

# print(cooccurrence)

# GloVe
print("Start GloVe calculation")
coocMatric = np.array(cooccurrence, "float32")


glove_model = GloVe(n=vecLength, max_iter=max_iter,
                    display_progress=display_progress)
embeddings = glove_model.fit(coocMatric)

print(embeddings)
print(embeddings.shape)

del cooccurrence, coocMatric
gc.collect()

# Output calculation result
dicIndex = 0
# result=[]
# nowTime = tic.getNow().strftime('%Y%m%d_%H%M%S')
GlovePath = "../data/keras_GloVeVec_CIRCLE-seq_%s_%s_%s.csv" % (coWindow, vecLength,max_iter)

f = open(GlovePath,'w',newline='')

writer = csv.writer(f)

for embeddingsItem in embeddings:
    item = np.array([dicIndex])
    item = np.append(item, embeddingsItem)
    writer.writerow(item)
    dicIndex = dicIndex + 1
print(dicIndex)
print("Finished!")

f.close()


# def loadGlove(inputpath, outputpath=""):
#     data_list = []
#     wordEmb = {}
#     with open(inputpath) as f:
#         for line in f:
#             # 基本的数据整理
#             ll = line.strip().split(',')
#             ll[0] = str(int(float(ll[0])))
#             data_list.append(ll)
#
#
#             # 构建wordembeding的选项
#             ll_new = [float(i) for i in ll]
#             emb = np.array(ll_new[1:], dtype="float32")
#             wordEmb[str(int(ll_new[0]))] = emb
#     print(len(data_list))
#     if outputpath != "":
#         with open(outputpath) as f:
#             for data in data_list:
#                 f.writelines(' '.join(data))
#         # data_list = [float(i) for i in data_list]
#     return wordEmb
#
# from tensorflow.keras.layers import Embedding
# VERBOSE = 1
# VOCAB_SIZE = 16  # 4**3
# EMBED_SIZE = 100
# maxlen = 23  # [(L-kmer)/step] +1
#
# print("GloVe model loaded")
# glove_inputpath = "...\data\keras_GloVeVec_Kleinstiver_5gRNA_5_100_10000.csv"
# # load GloVe model
# model_glove = loadGlove(glove_inputpath)
#
#
# # print(model_glove)
#
# embedding_weights = np.zeros((VOCAB_SIZE, EMBED_SIZE))  # 词语的数量×嵌入的维度
# for i in range(VOCAB_SIZE):
#     embedding_weights[i, :] = model_glove[str(i)]
#
# crispor = np.array(crispor,dtype=np.float32)
data = []
for i in range(len(crispor)):
    data.append(crispor[i][1:])

print(data[0])

new_coding = Bunch(
    # target_names=loaddata.target_names,
    target=loaddata.target,
    images=data
)

iswritepkl = True
if iswritepkl is True:
    # we create the pkl file for later use
    pickle_out = open("../data/encoded_CnnCrispr_CIRCLE-seqwithoutTsai.pkl", "wb")
    pkl.dump(new_coding, pickle_out)
    pickle_out.close()


