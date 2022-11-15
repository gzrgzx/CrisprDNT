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


arr = np.zeros((len(target_dna_list),4))

new_pd = pd.DataFrame(arr,columns=range(1,5))

MATCH_ROW_NUMBER1 = {"AC": 0, "AG": 1, "AT": 2, "CA": 3, "CG": 4, "CT": 5, "GA": 6,
                    "GC": 7, "GT": 8, "TA": 9, "TC": 10, "TG": 11}

# zhuanhuan_list = [1,4,7,9]

for i in range(len(target_rna_list)):
    for j in range(len(target_rna_list[0])-3):
        if target_rna_list[i][j] != target_dna_list[i][j]:
            temp = target_rna_list[i][j] + target_dna_list[i][j]
            if MATCH_ROW_NUMBER1[temp] == 1 or MATCH_ROW_NUMBER1[temp] == 6:
                new_pd.iat[i, 0] += 1
            elif MATCH_ROW_NUMBER1[temp] == 5 or MATCH_ROW_NUMBER1[temp] == 10:
                new_pd.iat[i, 1] += 1
            else:
                new_pd.iat[i, 2] += 1

    new_pd.iat[i,3] = dfGuideSeq.loc[i, 'reads']


print(new_pd)
# print(new_pd.info)

cor = new_pd.corr()
print(cor)
print(cor[4])


# 柱状图
x = ['Transition_1','Transition_2','Transversion']
plt.barh(x,abs(cor[4][0:-1]),color='SeaGreen')
# plt.errorbar(x,y=cor[4][0:-1])
# l1=plt.plot(x,cor[21][0:-1],'b--')
# plt.title('The Lasers in Three Conditions')
plt.ylabel('Mismatch Type')
plt.xlabel('Effect on GUIDE-seq reads')

plt.tight_layout()

# plt.figure(figsize=(20, 20.5))

# plt.yticks(x)

plt.savefig('mismatch_type_bar.jpg')

plt.show()


# 折线图
# x = ['Transition_1','Transition_2','Transversion']
# plt.plot(x,cor[4][0:-1],'go-')
# # plt.errorbar(x,y=cor[4][0:-1])
# # l1=plt.plot(x,cor[21][0:-1],'b--')
# # plt.title('The Lasers in Three Conditions')
# plt.xlabel('Mismatch Type')
# plt.ylabel('Effect on GUIDE-seq reads')
#
# # plt.xticks(x)
#
# plt.savefig('mismatch_type.jpg')
#
# plt.show()






