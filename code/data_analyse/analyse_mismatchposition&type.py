import sys
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# dfGuideSeq = pd.read_pickle('../data/guideseq_data (dataset II-3).pkl')
# print(dfGuideSeq.columns)
# print(dfGuideSeq)
#
# target_dna_list = []
# target_rna_list = []
# target_label = []
# target_read = []
#
# for n in range(len(dfGuideSeq)):
#     target_dna = list(dfGuideSeq.loc[n, '30mer_mut'])
#     target_rna = list(dfGuideSeq.loc[n, '30mer'])
#     if target_rna[-3] == 'N':
#         if target_dna[-3] >= 'A' and target_dna[-3] <= 'Z':
#             target_rna[-3] = target_dna[-3]
#
#     for i in range(len(target_dna)):
#         if target_dna[i] >= 'a' and target_dna[i] <= 'z':
#             target_dna[i] = chr(ord(target_dna[i]) - ord('a') + ord('A'))
#         if target_dna[i] == 'N':
#             target_dna[i] = target_rna[i]
#
#     target_dna = ''.join(target_dna)
#     target_rna = ''.join(target_rna)
#     target_rna_list.append(target_rna)
#     target_dna_list.append(target_dna)
#     if dfGuideSeq.loc[n, 'GUIDE-SEQ Reads'] == 0:
#         target_label.append(0)
#     else:
#         target_label.append(1)
#     target_read.append(dfGuideSeq.loc[n, 'GUIDE-SEQ Reads'])
#
#
# target_data = {'sgrna':target_rna_list,'otdna':target_dna_list,'label':target_label,'reads':target_read}
#
# target_data = pd.DataFrame(target_data)
#
# target_data.to_csv('Tasi_reads_offTarget.csv')




fpath = 'Tasi_reads_offTarget.csv'
dfGuideSeq = pd.read_csv(fpath, sep=',')

target_dna_list = []
target_rna_list = []

for n in range(len(dfGuideSeq)):
    target_dna = list(dfGuideSeq.loc[n, 'otdna'])
    target_rna = list(dfGuideSeq.loc[n, 'sgrna'])

    if dfGuideSeq.loc[n, 'label'] == 1:

        for i in range(len(target_rna)):
            if target_rna[i] == 'N':
                target_rna[i] = target_dna[i]

        for i in range(len(target_dna)):
            if target_dna[i] >= 'a' and target_dna[i] <= 'z':
                target_dna[i] = chr(ord(target_dna[i]) - ord('a') + ord('A'))
            if target_dna[i] == 'N':
                target_dna[i] = target_rna[i]

        target_dna = ''.join(target_dna)
        target_rna = ''.join(target_rna)
        target_rna_list.append(target_rna)
        target_dna_list.append(target_dna)



MATCH_ROW_NUMBER1 = {"AC": 0, "AG": 1, "AT": 2, "CA": 3, "CG": 4, "CT": 5, "GA": 6,
                    "GC": 7, "GT": 8, "TA": 9, "TC": 10, "TG": 11}

column_list = range(1,24)

index_list = ["rA-dC", "rA-dG", "rA-dT", "rC-dA", "rC-dG", "rC-dT", "rG-dA",
                    "rG-dC", "rG-dT", "rT-dA", "rT-dC", "rT-dG"]

arr = np.zeros((len(index_list),23))

new_pd = pd.DataFrame(arr, index=index_list, columns=column_list)

zhuanhuan1 = 0
zhuanhuan2 = 0
dianhuan = 0

# zhuanhuan_list = [1,4,7,9]

for i in range(len(target_rna_list)):
    for j in range(len(target_rna_list[0])):
        if target_rna_list[i][j] != target_dna_list[i][j]:
            temp = target_rna_list[i][j] + target_dna_list[i][j]
            new_pd.iat[MATCH_ROW_NUMBER1[temp],j]+=dfGuideSeq.loc[i, 'reads']


    # new_pd.iat[i,20] = dfGuideSeq.loc[i, 'reads']


print(new_pd)
# print(new_pd.info)

# cor = new_pd.corr()
# print(cor[21])
#
# x = range(1,21)
#
#
# # l1=plt.plot(cor[21][0:-1],'b--')
# plt.plot(cor[21][0:-1],'o-')
# # plt.title('The Lasers in Three Conditions')
# plt.xlabel('Mismatch Position')
# plt.ylabel('Effect on GUIDE-seq reads')
# plt.legend()
#
# plt.xticks(range(1,21))
#
# plt.savefig('mismatch_position.jpg')
#
# plt.show()


# print(new_pd)

# 导入用到的包
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# mpl.rcParams['font.sans-serif'] = ['SimHei']   #设置简黑字体
mpl.rcParams['font.sans-serif'] = ['Songti SC']
mpl.rcParams['axes.unicode_minus'] = False

cm_light = mpl.colors.ListedColormap(['tab:green','yellow','orange','red'])
plt.figure(figsize=(15,7))
# mask标记过滤值
# ax = sns.heatmap(new_pd, square=True, annot=True, vmin=0, vmax=100, fmt='.0f', linewidths=.05
#                 , linecolor='gray', cmap=cm_light)

ax = sns.heatmap(new_pd,cmap="RdBu_r")

plt.ylim(0, len(new_pd)+.5)
plt.xlim(0, len(new_pd.columns)+.5)
plt.xlabel('Mismatch Position')
plt.ylabel('Mismatch Type')

plt.savefig('type.jpg')

plt.show()





