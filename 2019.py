import os
GPU = True
if GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import re
import random
import collections
import numpy as np
from tqdm import tqdm
import tensorflow as tf

import nltk
from nltk.translate.bleu_score import SmoothingFunction
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import wordnet as wn

import keras
from keras import backend as K
from keras.layers import *
from keras.models import *
from keras.preprocessing import *
from keras.preprocessing.image import *
from keras.engine.topology import *
from keras.callbacks import *
from keras import optimizers
from keras import metrics

os.environ['PYTHONHASHSEED'] = '2019'
random.seed(2019)
np.random.seed(2019)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

# parameters  排序

IMG_DIM = 299
EPOCH = 100
EXP = '1'
BATCH = 1
MAXLEN = 12  # 14 12
# VOCAB = 1000  # 1000 900
# AUGMENT = False  # False True
DROP =0.0  # 0.5 0.2
DIM = 128  # 128 256
L2 = None  # None l2(1e-4)
# W2V = [1, 50]  # 50 100 200 300
# GLV = [0, 50]  # 50 100 200 300

# ---Data preparation---
# nltk.download('stopwords')
stop = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")
datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1, zoom_range=0.1)

def text_standard(text, remove=False):
    # punctuation & lower & stop & stem
    temp = text.strip('?').split(' ')
    temp_list = []
    for i in range(len(temp)):
        if temp[i] != '':
            temp[i] = re.sub("[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+ ", "", temp[i].lower())
            if remove == True:
                if temp[i] in stop and temp[i] != 'no':
                    temp[i] = ''
            temp[i] = stemmer.stem(temp[i])
            temp_list.append(temp[i].replace('-',' '))
    while '' in temp_list:
        temp_list.remove('')
    return temp_list

def traindata_prepare():
    name, ques, answ, labe = [], [], [], []
    f = open('/home/yzhou/VQA/VQAMed2018Train/VQAMed2018Train-QA.csv', 'r', encoding='utf-8')
    lines = f.readlines()
    for line in lines:
        temp_name = line.split('\t')[1]
        temp_name = '/home/yzhou/VQA/VQAMed2018Train/VQAMed2018Train-images/'+temp_name+'.jpg'
        name.append(temp_name)
        temp_ques = text_standard(line.split('\t')[2])
        ques.append(temp_ques)
        temp_answ = text_standard(line.split('\t')[3], True)
        answ.append(temp_answ)
    f.close()
    f = open('/home/yzhou/VQA/VQAMed2018Valid/VQAMed2018Valid-QA.csv', 'r', encoding='utf-8')
    lines = f.readlines()
    for line in lines:
        temp_name = line.split('\t')[1]
        temp_name = '/home/yzhou/VQA/VQAMed2018Valid/VQAMed2018Valid-images/'+temp_name+'.jpg'
        name.append(temp_name)
        temp_ques = text_standard(line.split('\t')[2])
        ques.append(temp_ques)
        temp_answ = text_standard(line.split('\t')[3], True)
        answ.append(temp_answ)
    f.close()
    return name, ques, answ

def testdata_prepare():
    name, imag, ques = [], [], []
    f = open('/home/yzhou/VQA/VQAMed2018Test/VQAMed2018Test-QA-Eval.csv', 'r', encoding='utf-8')
    lines = f.readlines()
    for line in lines:
        temp_name = line.split('\t')[1]
        name.append(temp_name)
        temp_name = '/home/yzhou/VQA/VQAMed2018Test/VQAMed2018Test-images/'+temp_name+'.jpg'
        temp_imag = image.load_img(temp_name, target_size=(IMG_DIM, IMG_DIM))
        temp_imag = image.img_to_array(temp_imag)
        temp_imag = np.expand_dims(temp_imag, axis=0)
        imag.append(temp_imag)
        temp_ques = text_standard(line.split('\t')[2])
        ques.append(temp_ques)
    f.close()
    imag = np.array(imag)
    imag = np.squeeze(imag)
    return name, imag, ques

train_imag, train_ques, train_answ = traindata_prepare()
test_name, test_imag, test_ques = testdata_prepare()

def rest_words(word_list, freq=2, cate='question'):
    freq_list = collections.Counter([val for sublist in word_list for val in sublist])
    rest_list = dict(filter(lambda x: x[1] > freq, freq_list.items()))
    print(cate + ' vocabulary: ', len(rest_list))
    return rest_list

rest_ques = rest_words(train_ques+test_ques)
rest_answ = rest_words(train_answ, 3, 'answer')

def test_questions(test_ques):
    ques = []
    for i in range(len(test_ques)):
        for j in range(len(test_ques[i])):
            if test_ques[i][j] not in rest_ques.keys():
                test_ques[i][j] = '_unk'
        temp_ques = ' '.join(test_ques[i])
        ques.append(temp_ques)
    return ques

def label_prepare(images, questions, answers):
    imag, ques, answ, labe = [], [], [], []
    for i in tqdm(range(len(questions))):
        for j in range(len(questions[i])):
            if questions[i][j] not in rest_ques.keys():
                questions[i][j] = '_unk'
        for j in range(len(answers[i])+1):
            if j != len(answers[i]):
                if answers[i][j] not in rest_answ.keys():
                    answers[i][j] = '_unk'
                labe.append(answers[i][j])
            else:
                labe.append('_end')
            temp_imag = image.load_img(images[i], target_size=(IMG_DIM, IMG_DIM))
            temp_imag = image.img_to_array(temp_imag)
            temp_imag = np.expand_dims(temp_imag, axis=0)
            for _temp_imag in datagen.flow(temp_imag, batch_size=1):
                break
            imag.append(_temp_imag)
            ques.append(' '.join(questions[i]))
            answ.append('_sta '+' '.join(answers[i][:j]))
    imag = np.array(imag)
    imag = np.squeeze(imag)
    return imag, ques, answ, labe

train_imag, train_ques, train_answ, train_labe = label_prepare(train_imag, train_ques, train_answ)
test_ques = test_questions(test_ques)

#Tokenizer
tokenizer_q = text.Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\]^`{|}~ ')
tokenizer_q.fit_on_texts(train_ques+test_ques)
train_ques = tokenizer_q.texts_to_sequences(train_ques)
test_ques = tokenizer_q.texts_to_sequences(test_ques)
train_ques = sequence.pad_sequences(train_ques, MAXLEN, truncating='post')
test_ques = sequence.pad_sequences(test_ques, MAXLEN, truncating='post')

tokenizer_a = text.Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\]^`{|}~ ', oov_token=1)
tokenizer_a.fit_on_texts(train_answ)
train_answ = tokenizer_a.texts_to_sequences(train_answ)
train_answ = sequence.pad_sequences(train_answ, MAXLEN)
train_labe = tokenizer_a.texts_to_sequences(train_labe)
train_labe = sequence.pad_sequences(train_labe, 1)

state = np.random.get_state()
np.random.shuffle(train_imag)
np.random.set_state(state)
np.random.shuffle(train_ques)
np.random.set_state(state)
np.random.shuffle(train_answ)
np.random.set_state(state)
np.random.shuffle(train_labe)

if GLV[0] != 0:
    w2v_ques = np.zeros((len(tokenizer_q.index_word)+1, GLV[1]))
    w2v_answ = np.zeros((len(tokenizer_a.index_word)+1, GLV[1]))
    f = open('/home/yzhou/VQA/glove/glove.6B.'+str(GLV[1])+'d.txt', 'r')
    lines = f.readlines()
    for line in tqdm(lines):
        word = ''.join(line.split(' ')[:-GLV[1]])
        for i in range(len(word)):
            cand = word[:len(word)-i]
            for j in range(len(tokenizer_q.index_word)):
                if tokenizer_q.index_word[j+1] == cand and w2v_ques[j+1][0] == 0:
                    w2v_ques[j+1] = np.array(line.split(' ')[-GLV[1]:])
            for j in range(len(tokenizer_a.index_word)):
                if tokenizer_a.index_word[j+1] == cand and w2v_answ[j+1][0] == 0:
                    w2v_answ[j+1] = np.array(line.split(' ')[-GLV[1]:])
    f.close()
    
# np.save('/home/yzhou/VQA/data/temp/train.imag',train_imag)
# np.save('/home/yzhou/VQA/data/temp/train.ques',train_ques)
# np.save('/home/yzhou/VQA/data/temp/train.answ',train_answ)
# np.save('/home/yzhou/VQA/data/temp/train.labe',train_labe)
# np.save('/home/yzhou/VQA/data/temp/w2v50.ques',w2v50_ques)
# np.save('/home/yzhou/VQA/data/temp/w2v50.ques',w2v50_answ)
# train_imag = np.load('/home/yzhou/VQA/data/temp/train.imag.npy')
# train_ques = np.load('/home/yzhou/VQA/data/temp/train.ques.npy')
# train_answ = np.load('/home/yzhou/VQA/data/temp/train.answ.npy')
# train_labe = np.load('/home/yzhou/VQA/data/temp/train.labe.npy')
# w2v50_ques = np.load('/home/yzhou/VQA/data/temp/w2v50.ques.npy')
# w2v50_answ = np.load('/home/yzhou/VQA/data/temp/w2v50.answ.npy')

def rec_unit(x, size, stage):
    x = Conv2D(size, 3, activation='relu', name='conv'+stage+'_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=L2)(x)
    x = BatchNormalization(name='norm'+stage+'_1')(x)
    x = Dropout(DROP, name='drop'+stage+'_1')(x)
    x1 = Conv2D(size, 3, activation='relu', name='conv'+stage+'_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=L2)(x)
    x1 = BatchNormalization(name='norm'+stage+'_2')(x1)
    x1 = Dropout(DROP, name='drop'+stage+'_2')(x1)
    x1 = add([x,x1], name='add'+stage+'_1')
    x1 = Conv2D(size, 3, activation='relu', name='conv'+stage+'_3', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=L2)(x1)
    x1 = BatchNormalization(name='norm'+stage+'_3')(x1)
    x1 = Dropout(DROP, name='drop'+stage+'_3')(x1)
    return x1

def att_unit(x, size, stage):
    a, b = x
    a = Conv2D(size, 1, activation='relu', name='conv'+stage+'_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=L2)(a)
    a = BatchNormalization(name='norm'+stage+'_1')(a)
    a = Dropout(DROP, name='drop'+stage+'_1')(a)
    b1 = Conv2D(size, 1, activation='relu', name='conv'+stage+'_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=L2)(b)
    b1 = BatchNormalization(name='norm'+stage+'_2')(b1)
    b1 = Dropout(DROP, name='drop'+stage+'_2')(b1)
    a = add([a,b1], name='add'+stage+'_1')
    a = Conv2D(size, 1, activation='relu', name='conv'+stage+'_3', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=L2)(a)
    a = BatchNormalization(name='norm'+stage+'_3')(a)
    a = Dropout(DROP, name='drop'+stage+'_3')(a)
    a = Activation('sigmoid', name='sigm'+stage+'_1')(a)
    a = multiply([a,b], name='mul'+stage+'_1')
    return a

def U_Net(x):
    size = [DIM//8,DIM//4,DIM//2,DIM]
    x1 = rec_unit(x, stage='x1', size=size[0])
    x2 = MaxPooling2D((2, 2), strides=2, name='pool1')(x1)
    x2 = rec_unit(x2, stage='x2', size=size[1])
    x3 = MaxPooling2D((2, 2), strides=2, name='pool2')(x2)
    x3 = rec_unit(x3, stage='x3', size=size[2])
    x4 = MaxPooling2D((2, 2), strides=2, name='pool3')(x3)
    x4 = rec_unit(x4, stage='x4', size=size[3])
    d3 = Conv2DTranspose(size[2], 2, strides=2, name='up3', padding='same')(x4)
    d3 = BatchNormalization(name='n3')(d3)
    x3 = att_unit([d3, x3], stage='a3', size=size[2])
    d3 = concatenate([x3, d3], name='cat3', axis=3)
    d3 = rec_unit(d3, stage='d3', size=size[2])
    d2 = Conv2DTranspose(size[1], 2, strides=2, name='up2', padding='same')(d3)
    d2 = BatchNormalization(name='n2')(d2)
    x2 = att_unit([d2, x2], stage='a2', size=size[1])
    d2 = concatenate([x2, d2], name='cat2', axis=3)
    d2 = rec_unit(d2, stage='d2', size=size[1])
    d1 = Conv2DTranspose(size[0], 2, strides=2, name='up1', padding='same')(d2)
    d1 = BatchNormalization(name='n1')(d1)
    x1 = att_unit([d1, x1], stage='a1', size=size[0])
    d1 = concatenate([x1, d1], name='cat1', axis=3)
    d1 = rec_unit(d1, stage='d1', size=size[0])
    u2 = MaxPooling2D((2, 2), strides=2, name='pool5')(d1)
    u2 = rec_unit(u2, stage='u2', size=size[1])
    d2 = att_unit([u2, d2], stage='c2', size=size[1])
    u2 = concatenate([d2, u2], name='con2', axis=3)
    u3 = MaxPooling2D((2, 2), strides=2, name='pool6')(u2)
    u3 = rec_unit(u3, stage='u3', size=size[2])
    d3 = att_unit([u3, d3], stage='c3', size=size[2])
    u3 = concatenate([d3, u3], name='con3', axis=3)
    u4 = MaxPooling2D((2, 2), strides=2, name='pool7')(u3)
    u4 = rec_unit(u4, stage='u4', size=size[3])
    x4 = att_unit([u4, x4], stage='c4', size=size[3])
    u4 = concatenate([x4, u4], name='con4', axis=3)
    u4 = GlobalAveragePooling2D(name='out')(u4)
    return u4

#问模型
in_qu = Input(shape=(MAXLEN,))
qu = Embedding(len(w2v50_ques),GLV[1],input_length=MAXLEN,weights=[w2v50_ques],trainable=True)(in_qu)
qu = Bidirectional(LSTM(DIM//2, return_sequences=True))(qu)

#答模型
in_an = Input(shape=(MAXLEN,))
an = Embedding(len(w2v50_answ),GLV[1],input_length=MAXLEN,weights=[w2v50_answ],trainable=True)(in_an)
an = LSTM(DIM, return_sequences=True)(an)

#图模型
in_im = Input(shape=(IMG_DIM, IMG_DIM, 3))
im = U_Net(in_im)
im = RepeatVector(MAXLEN)(im)
im = TimeDistributed(Dense(DIM, activation='relu'))(im)

#合并模型
me = concatenate([im, qu, an])
me = LSTM(DIM, return_sequences=False)(me)
me = Dense(len(tokenizer_a.word_index), activation='softmax')(me)
vqa_model = Model(inputs=[in_im,in_qu,in_an], outputs=me)
# vqa_model.summary()

EXP = '~unet~GLV'
vqa_model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(), metrics=[metrics.sparse_categorical_accuracy])

# callback
checkpoint = ModelCheckpoint('/home/yzhou/VQA/data/weight/' + EXP + '.hdf5', verbose=1, save_best_only=True, mode='min')
plateau = ReduceLROnPlateau(factor=0.2, patience=10, verbose=1, min_lr=0.000001)
stop = EarlyStopping(patience=7, verbose=1)
call_list = [checkpoint,plateau]

history = vqa_model.fit(x=[train_imag[:10], train_ques[:10], train_answ[:10]], y=train_labe[:10],
                        validation_data=([train_imag[:9], train_ques[:9], train_answ[:9]], train_labe[:9]),
                        batch_size=BATCH, epochs=EPOCH, callbacks=call_list)
print(np.argsort(vqa_model.predict([train_imag[:10], train_ques[:10], train_answ[:10]])))
print(train_labe[:10])
