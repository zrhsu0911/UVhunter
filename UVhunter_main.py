# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import re
import argparse
import biovec
from Bio import SeqIO
import numpy as np
from keras.models import load_model

# 第一個引數為 fasta file，第二個引數 -n 為 cpu線程數目（非必要選項，默認20），第三個引數 -e 為 word2vec epoch次數（非必要選項，默認50）

my_parser = argparse.ArgumentParser()
my_parser.add_argument("fasta_fname", type=str)
my_parser.add_argument(
    "-n",
    help='number of cpu threads usage',
    type=int,
    dest='my_workers',
    default=20)
my_parser.add_argument(
    "-e",
    help='number of word2vec training epochs',
    type=int,
    dest='my_epochs',
    default=50)
my_args = my_parser.parse_args()
my_workers = my_args.my_workers
my_epochs = my_args.my_epochs
fasta_fname = my_args.fasta_fname
# randInteger = randint(0, 100)
# fasta_fname = sys.argv[1]
# corpus_outfname = fasta_fname + "corpus" + str(randInteger) + ".txt"
out_fname = fasta_fname + ".out"

# word2vec model loading
# Respiratory Adeno Parainflu
pv_res = biovec.load_protvec('models/respiratory_biovec_3gram.bin')
# EV and Rhino
pv_ev = biovec.load_protvec('models/ev_biovec_3gram.bin')
# Influ
pv_influ = biovec.load_protvec('models/influ_biovec_3gram.bin')


# find all chars except 'ATCGN' and trim continuous 'NN...' from head and tail
regex_findOtherChars = re.compile(r'[^ACTGN]', re.IGNORECASE)
regex_findNFromHead = re.compile(r'^[nN]*', re.IGNORECASE)
regex_findNFromTail = re.compile(r'[nN]*$', re.IGNORECASE)

# generate data for prediction
QIDSeq_dict = {}
for record in SeqIO.parse(fasta_fname, "fasta"):
    # UniProtID = (record.id).split('|')[1].strip()
    # QID = re.search('(.+?)\\.', (record.id)).group(1)
    QID = str(record.id)
    Seq = str(record.seq).upper()
    Seq_replaced = regex_findOtherChars.sub(r'N', Seq)
    Seq_trimHead = Seq_replaced.lstrip(
        regex_findNFromHead.search(Seq_replaced).group())
    Seq_trimed = Seq_trimHead.rstrip(
        regex_findNFromTail.search(Seq_trimHead).group())
    QIDSeq_dict[QID] = Seq_trimed

QIDNpArray_dict = {}  # Res Adeno Parainflu
ev_QIDNpArray_dict = {}  # EV Rhino
influ_QIDNpArray_dict = {}  # influ
for qid, Seq in QIDSeq_dict.items():
    biovec_array_list = pv_res.to_vecs(Seq)
    ev_biovec_array_list = pv_ev.to_vecs(Seq)
    influ_biovec_array_list = pv_influ.to_vecs(Seq)
    biovec_NpArray1 = np.array(biovec_array_list)
    biovec_NpArray = biovec_NpArray1.reshape((1, 300, 1))
    ev_biovec_NpArray1 = np.array(ev_biovec_array_list)
    ev_biovec_NpArray = ev_biovec_NpArray1.reshape((1, 300, 1))
    influ_biovec_NpArray1 = np.array(influ_biovec_array_list)
    influ_biovec_NpArray = influ_biovec_NpArray1.reshape((1, 300, 1))
    QIDNpArray_dict[qid] = biovec_NpArray
    ev_QIDNpArray_dict[qid] = ev_biovec_NpArray
    influ_QIDNpArray_dict[qid] = influ_biovec_NpArray

# load keras model
# Res 大類
re_model = load_model('models/re_3gram_v6-1.h5')
# ev and Rhino
ev_model = load_model('models/EV/ev71_only_v1.h5')
rhino_model = load_model('models/Rhino/rhino3_evbiovec_v1.h5')
# Adeno
adeno_model = load_model('models/Adeno/adeno_3gram_v3.h5')
# parainflu
parainflu_model = load_model('models/Parainflu/parainflu_v1.h5')
# influ
influ_lv2_model = load_model('models/Influ/influ_lv2.h5')
influ_lv3HA_model = load_model('models/Influ/influ_lv3-HA.h5')
influ_lv3NA_model = load_model('models/Influ/influ_lv3-NA.h5')
influ_lv4HA_model = load_model('models/Influ/influ_lv4-HA.h5')
influ_lv4NA_model = load_model('models/Influ/influ_lv4-NA.h5')

# build genotype to spec, and id to genotype relation
# Res 大類
res_IDtoName_dict = {}
with open('models/Labeltype.txt', 'r') as labelspecfile:
    for line in labelspecfile:
        labelID, labelName = line.rstrip().split()
        res_IDtoName_dict[labelID] = labelName

# EV and rhino
ev_IDtoName_dict = {}
with open('models/EV/Labeltype.txt', 'r') as labelspecfile:
    for line in labelspecfile:
        labelID, labelName = line.rstrip().split()
        ev_IDtoName_dict[labelID] = labelName

rhino_IDtoName_dict = {}
with open('models/Rhino/Labeltype.txt', 'r') as labelspecfile:
    for line in labelspecfile:
        labelID, labelName = line.rstrip().split()
        rhino_IDtoName_dict[labelID] = labelName

# Adeno
adeno_IDtoName_dict = {}
with open('models/Adeno/Labeltype.txt', 'r') as labelspecfile:
    for line in labelspecfile:
        labelID, labelName = line.rstrip().split()
        adeno_IDtoName_dict[labelID] = labelName

# parainflu
parainflu_IDtoName_dict = {}
with open('models/Parainflu/Labeltype.txt', 'r') as labelspecfile:
    for line in labelspecfile:
        labelID, labelName = line.rstrip().split()
        parainflu_IDtoName_dict[labelID] = labelName

# influ
influ_lv2_IDtoName_dict = {}
with open('models/Influ/Labeltype_lv2.txt', 'r') as labelspecfile:
    for line in labelspecfile:
        labelID, labelName = line.rstrip().split()
        influ_lv2_IDtoName_dict[labelID] = labelName

with open('models/Influ/thres_lv3-HA.txt', 'r') as labelspecfile:
    thres_lv3HA = float(labelspecfile.read().rstrip())

with open('models/Influ/thres_lv3-NA.txt', 'r') as labelspecfile:
    thres_lv3NA = float(labelspecfile.read().rstrip())

influ_lv4HA_IDtoName_dict = {}
with open('models/Influ/Labeltype_lv4-HA.txt', 'r') as labelspecfile:
    for line in labelspecfile:
        labelID, labelName = line.rstrip().split()
        influ_lv4HA_IDtoName_dict[labelID] = labelName

influ_lv4NA_IDtoName_dict = {}
with open('models/Influ/Labeltype_lv4-NA.txt', 'r') as labelspecfile:
    for line in labelspecfile:
        labelID, labelName = line.rstrip().split()
        influ_lv4NA_IDtoName_dict[labelID] = labelName


# Adeno genotyping
def Adeno_genotyping(qid):
    ndarray = QIDNpArray_dict[qid]
    y_pred = adeno_model.predict(ndarray)
    y_id = np.argmax(y_pred[0])
    genoName = adeno_IDtoName_dict[str(y_id)]
    return genoName


# parainflu genotyping
def Parainflu_genotyping(qid):
    ndarray = QIDNpArray_dict[qid]
    y_pred = parainflu_model.predict(ndarray)
    y_id = np.argmax(y_pred[0])
    genoName = parainflu_IDtoName_dict[str(y_id)]
    return genoName


# EV genotyping
def EV_genotyping(qid):
    ndarray = ev_QIDNpArray_dict[qid]
    y_pred = ev_model.predict(ndarray)
    y_id = np.argmax(y_pred[0])
    genoName = ev_IDtoName_dict[str(y_id)]
    return genoName


# Rhino genotyping
def Rhino_genotyping(qid):
    ndarray = ev_QIDNpArray_dict[qid]
    y_pred = rhino_model.predict(ndarray)
    y_id = np.argmax(y_pred[0])
    genoName = rhino_IDtoName_dict[str(y_id)]
    return genoName


# influ genotyping
def Influ_genotyping(qid):
    ndarray = influ_QIDNpArray_dict[qid]
    # lv2 8-type
    y_pred_lv2 = influ_lv2_model.predict(ndarray)
    y_id_lv2 = np.argmax(y_pred_lv2[0])
    genoName = 'N/A'
    if y_id_lv2 == 0:
        # go to lv3 HA
        genoName = 'Non-Human_Influenza_HA'
        y_pred_lv3HA = influ_lv3HA_model.predict(ndarray)
        y_pred_lv3HA_value = y_pred_lv3HA[0, 0]
        if y_pred_lv3HA_value >= thres_lv3HA:
            # go to lv4 HA
            y_pred_lv4HA = influ_lv4HA_model.predict(ndarray)
            y_id_lv4HA = np.argmax(y_pred_lv4HA[0])
            y_label_lv4HA = influ_lv4HA_IDtoName_dict[str(y_id_lv4HA)]
            if y_label_lv4HA == 'others':
                genoName = 'Human_Influenza_HA'
            else:
                genoName = 'Human_Influenza_' + str(y_label_lv4HA)
    if y_id_lv2 == 2:
        # go to lv3 NA
        genoName = 'Non-Human_Influenza_NA'
        y_pred_lv3NA = influ_lv3NA_model.predict(ndarray)
        y_pred_lv3NA_value = y_pred_lv3NA[0, 0]
        if y_pred_lv3NA_value >= thres_lv3NA:
            # go to lv4 NA
            y_pred_lv4NA = influ_lv4NA_model.predict(ndarray)
            y_id_lv4NA = np.argmax(y_pred_lv4NA[0])
            y_label_lv4NA = influ_lv4NA_IDtoName_dict[str(y_id_lv4NA)]
            if y_label_lv4NA == 'others':
                genoName = 'Human_Influenza_NA'
            else:
                genoName = 'Human_Influenza_' + str(y_label_lv4NA)
    return genoName


# perform prediction and write output
with open(out_fname, 'w') as outfile:
    for qid, nparray in QIDNpArray_dict.items():
        y_pred = re_model.predict(nparray)
        y_argsort = np.argsort(y_pred[0])
        # y_id = np.argmax(y_pred[0])
        # labelName = res_IDtoName_dict[str(y_id)]
        y_id1st = y_argsort[-1]
        y_id2nd = y_argsort[-2]
        eval_score_1st_hit = y_pred[0, y_id1st]  # round(float(y_pred[0, y_id1st]), 10)
        eval_score_2nd_hit = y_pred[0, y_id2nd]  # round(float(y_pred[0, y_id2nd]), 10)
        labelName_1st = res_IDtoName_dict[str(y_id1st)]
        labelName_2nd = res_IDtoName_dict[str(y_id2nd)]
        genoName = 'N/A'
        # Adeno subroutine
        if y_id1st == 0:
            genoName = Adeno_genotyping(qid)
        # EV subroutine
        if y_id1st == 2:
            genoName = EV_genotyping(qid)
        # Rhino subroutine
        if y_id1st == 10:
            genoName = Rhino_genotyping(qid)
        # Parainflu subroutine
        if y_id1st == 4:
            genoName = Parainflu_genotyping(qid)
        # Influ subroutine
        if y_id1st == 6:
            genoName = Influ_genotyping(qid)
    
        outstr = qid + '\t' + labelName_1st + '\t' + str(
            eval_score_1st_hit) + '\t' + labelName_2nd + '\t' + str(
                eval_score_2nd_hit) + '\t' + genoName + '\n'
        outfile.write(outstr)

# leave done message
with open(out_fname, 'a') as file1:
    outstr = '# done' + '\n'
    file1.write(outstr)
