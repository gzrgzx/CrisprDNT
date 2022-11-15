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


os.environ['CUDA_VISIBLE_DEVICES'] = '1,0'


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
for i in range(images.shape[0]):
    arr = []
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

new_coding = Bunch(
    # target_names=loaddata.target_names,
    target=loaddata.target,
    images=crispor
)

iswritepkl = True
if iswritepkl is True:
    # we create the pkl file for later use
    pickle_out = open("../data/encoded_offt_CIRCLE-seqwithoutTsai.pkl", "wb")
    pkl.dump(new_coding, pickle_out)
    pickle_out.close()


