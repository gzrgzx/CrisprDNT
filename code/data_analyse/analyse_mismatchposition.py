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


arr = np.zeros((len(target_dna_list),21))

new_pd = pd.DataFrame(arr,columns=range(1,22))



# zhuanhuan_list = [1,4,7,9]
print(len(target_rna_list[0])-3)

for i in range(len(target_rna_list)):
    for j in range(len(target_rna_list[0])-3):
        if target_rna_list[i][j] != target_dna_list[i][j]:
            new_pd.iat[i,j] = 1

    new_pd.iat[i,20] = dfGuideSeq.loc[i, 'reads']


print(new_pd)
# print(new_pd.info)

cor = new_pd.corr()
print(cor)
print(list(cor[21]))

#柱状图
plt.bar(range(1,21), cor[21][0:-1],color='#73c0de')

plt.xlabel('Mismatch Position')
plt.ylabel('Effect on GUIDE-seq reads')
# plt.legend()

plt.xticks(range(1,21))

plt.savefig('mismatch_position_bar.jpg')

plt.show()



#折线图
# # l1=plt.plot(cor[21][0:-1],'b--')
# plt.plot(cor[21][0:-1],'o-')
# # plt.title('The Lasers in Three Conditions')
# plt.xlabel('Mismatch Position')
# plt.ylabel('Effect on GUIDE-seq reads')
# # plt.legend()
#
# plt.xticks(range(1,21))
#
# plt.savefig('mismatch_position.jpg')
#
# plt.show()






