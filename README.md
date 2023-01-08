# CrisprDNT
Transformer-based Anti-noise Model for Predicting CRISPR-Cas9 Off-Target Activities.
![image](https://github.com/gzrgzx/CrisprDNT/blob/main/model.png)

PREREQUISITE

CrisprDNT was conducted by TensorFlow version 2.3.2 and python version 3.6.

Following Python packages should be installed:

* numpy
- pandas
* scikit-learn
- TensorFlow
* Keras

Data Description:

* dataset1->Doench et al.(Protein knockout detection)
* dataset2->Haeussler et al.(PCR, Digenome-Seq and HTGTS)
* dataset3->Cameron et al.(SITE-Seq)
* dataset4->Tasi et al.(GUIDE-seq)
* dataset5->Kleinstiver et al(GUIDE-seq)
* dataset6->Listgarten et al.(GUIDE-seq)
* dataset7->Chuai et al.(GUIDE-Seq, Digenome-Seq,HTGTS,BLESS and IDLVs)
* dataset8->Chuai et al.(GUIDE-Seq, Digenome-Seq,HTGTS,BLESS and IDLVs)
* CIRCLE_seq_10gRNA_wholeDataset->mismatches and indels
* elevation_6gRNA_wholeDataset->miamatches and indels

Code Description
    * data_process(coding scheme)
        * create_coding_scheme.py->Create CrisprDNT, CRISPR_IP, CRISPR_Net and CNN_std encoding.
        * CnnCrispr_coding.py->Create CnnCrispr and CNN_std encoding.
        * CRISPR-OFFT_coding.py->Create CRISPR-OFFT encoding.
    * data_analyse
        * analyse_mismatchposition&type.py->Analysis of the effect of mismatch position&type based on GUIDE-seq dataset.
        * analyse_mismatchposition.py->Analysis of the effect of mismatch position based on GUIDE-seq dataset.
        * analyse_type.py->Analysis of the effect of mismatch type based on GUIDE-seq dataset.
    * data_analyse
        * model_network.py->CrisprDNT network and anti-noise loss function code.
    * train&test
        * experiment.py->code to reproduce the experiments with CrisprDNT, CRISPR_IP, CRISPR_Net, CnnCrispr, CRISPR-OFFT and CNN_std

saved_model Description:

