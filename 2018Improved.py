#unfinished
#The part to be added: translate,word2vec,new model,beam search.
import sys
import keras
import re
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from googletrans import Translator
from stanfordcorenlp import StanfordCoreNLP
from gensim.models import word2vec
from pattern.en import lemma
from keras import *
from keras.layers import *
from keras.models import *
from keras.preprocessing import *
from keras.preprocessing.image import *
from keras.callbacks import *
from keras.applications.vgg16 import *
from keras.applications.vgg19 import *
from keras.applications.resnet50 import *
from keras.applications.inception_v3 import *
from keras.applications.inception_resnet_v2 import *

#常量
#path = '/home/yzhou/VQA'
path = 'C:/Users/zhou yangyang/VQA'
maxlen = 10
dic = 1000
dim = 128  #神经元基数
increase = 20  #图像变形数量
if os.path.exists(path + '/data/') == False:  #创建数据存放文件夹
    os.mkdir(path + '/data/')

#函数：处理文件
def data_processing(data):
    data_num  = open(path + '/data/' + data + '.num', 'w',encoding='utf-8')  # 序号
    data_name = open(path + '/data/' + data + '.name','w',encoding='utf-8')  # 文件名
    data_que  = open(path + '/data/' + data + '.q', 'w',encoding='utf-8')  # 问题
    if data != 'Test':
        data_ans  = open(path + '/data/' + data + '.a', 'w',encoding='utf-8')  # 答案
    with open(path + '/VQAMed2018' + data + '/VQAMed2018' + data + '-QA.csv', encoding='utf-8') as f:
        for line in f:
            data_num.write(line.split('	')[0]+'\n')
            data_name.write(line.split('	')[1]+'\n')
            if data != 'Test':
                data_que.write(line.split('	')[2]+'\n')
                data_ans.write(line.split('	')[3])
            else:
                data_que.write(line.split('	')[2])
    f.close()
    data_num.close()
    data_name.close()
    data_que.close()
    if data != 'Test':
        data_ans.close()
    
#处理文件
print('Generating data files...')
data_processing('Train')
data_processing('Valid')
#data_processing('Test')

#计算文件数量
num_train = len(open(path + '/data/Train.num', 'r').readlines())
num_valid = len(open(path + '/data/Valid.num', 'r').readlines())
#num_test = len(open(path + '/data/Test.num', 'r').readlines())

#函数：图像增强
def img_processing(data):
    f = open(path + '/data/' + data + '.name')  #读取图片名
    im = []
    i = 0
    z = np.zeros((1,1000))
    z = z.tolist()
    while 1:
        lines = f.readlines()
        if not lines:
            break
        for line in lines:
            line = line.strip()  #去掉图片名后面的'\n'
            if os.path.isfile(path + '/VQAMed2018' + data + '/VQAMed2018' + data + '-images/' + line + '.jpg') == True:
                img_path = path + '/VQAMed2018' + data + '/VQAMed2018' + data + '-images/' + line + '.jpg'
                img = image.load_img(img_path, target_size=(224, 224))  #读取文件
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x_o = preprocess_input(x)
                y = model_1.predict(x_o)  #提取特征量
                y = y.tolist()
                im.append(y)
                if data == 'Train':
                    for k in range(increase):  #图像增强
                        j = 0
                        for x_n in datagen.flow(x, batch_size=1):
                            j += 1
                            if j >= 1:
                                break
                        x_n = preprocess_input(x_n)
                        y = model_1.predict(x_n)
                        y = y.tolist()
                        im.append(y)
            else:
                im.append(z)  #找不到图片时
                print('Picture NO.', i+1, ' could not be found.')
            i += 1
            if data == 'Train':
                num_arrow = int(i * 50 / num_train) + 1  #计算显示多少个'>'
                percent = i * 100.0 / (num_train)  #计算完成进度，格式为xx.xx%
            elif data == 'Valid':
                num_arrow = int(i * 50 / num_valid) + 1
                percent = i * 100.0 / (num_valid)
            elif data == 'Test':
                num_arrow = int(i * 50 / num_test) + 1
                percent = i * 100.0 / (num_test)
            num_line = 50 - num_arrow  #计算显示多少个'-'
            process_bar = data +': [' + '>' * num_arrow + '-' * num_line + ']' + '%.2f' % percent + '%' + '\r'
            sys.stdout.write(process_bar) #这两句打印字符到终端
            sys.stdout.flush()
    im = np.array(im)
    im = np.squeeze(im)  #去掉为1的维度
    np.save(path + '/data/' + data + '_I3.npy', im)  #V16,V19,R50,I3,IR2
    print('\n')
    f.close()

#保存图像特征
print('Turning images into vectors...')
if os.path.isfile(path + '/data/Train_I3.npy') == False:
    model_1 = InceptionResNetV2(weights='imagenet', include_top=False, pooling='avg')  #加载模型
    datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1, zoom_range=0.1)  #图像增强器
    img_processing('Train')
    img_processing('Valid')
#    img_processing('Test')

#函数：问句增强
def que_processing(data):
    i = 0
    data_question  = open(path + '/data/' + data + '.qu', 'w',encoding='utf-8')
    with open(path + '/data/' + data + '.q', encoding='utf-8') as f:
        for lines in f:
            data_question.write(lines)
            lines = lines.strip()
            trans = Translator()
            for div in ['zh-CN','es','ar','pt','ru','ja','de','fr','hi','id','it','la','el','ko','pl','sv','th','tr','uk','nl']:
                re1 = trans.translate(lines,dest=div)
                re2 = trans.translate(re1.text)
                data_question.write(re2.text + '\n')
            i += 1
            num_arrow = int(i * 50 / num_train) + 1
            percent = i * 100.0 / (num_train)
            num_line = 50 - num_arrow  #计算显示多少个'-'
            process_bar = data +': [' + '>' * num_arrow + '-' * num_line + ']' + '%.2f' % percent + '%' + '\r'
            sys.stdout.write(process_bar) #这两句打印字符到终端
            sys.stdout.flush()
    f.close()
    data_question.close()

#生成增强问句
print('Translating questions...')
if os.path.isfile(path + '/data/Train.qu') == False:
    que_processing('Train')
#    que_processing('Valid')

#函数：问句预处理
def pretreat_q(data):
#    nlp = StanfordCoreNLP(r'/home/yzhou/stanford-corenlp-full-2018-02-27/')
    nlp = StanfordCoreNLP(r'C:/stanford-corenlp-full-2018-02-27/')
    stops = []
    fstop = open(path + '/stopwords.txt','r')
    lines = fstop.readlines()
    for line in lines:
        line = line.strip()
        stops.append(line)
    fstop.close()
    fw = open(path + '/data/' + data + '.que', 'w')
    if data == 'Train':
        f = open(path + '/data/' + data + '.qu',encoding='utf-8')
    else:
        f = open(path + '/data/' + data + '.q')
    lines = f.readlines()
    for line in lines:
        line = re.sub(r'[^A-Za-z0-9_ ]','',line)
        line = line.lower().split(' ')
        j = len(line)
        for i in range(j):
            for div in stops:  #停用词删除
                if line[i] == div:
                    line[i] = ''
            if re.match(r'[0-9]+', line[i]):  #数字替换
                line[i] = 'num'
            elif re.match(r'[a-z][0-9]+', line[i]):
                line[i] = 'pos'
            if line[i] != '':
                w = nlp.lemmas(line[i])
                line[i] = w[0][1]
            if i != j-1:
                if line[i] != '' and line[i+1] != '':
                    line[i] = line[i]+' '
            else:
                line[i] = line[i]+'\n'
            fw.write(line[i])
    f.close()
    fw.close()

#问句预处理
if os.path.isfile(path + '/data/Train.que') == False:
    pretreat_q('Train')
    pretreat_q('Valid')
    #pretreat_q('Test')

#函数：答句预处理
def pretreat_a(data):
#    nlp = StanfordCoreNLP(r'/home/yzhou/stanford-corenlp-full-2018-02-27/')
    nlp = StanfordCoreNLP(r'C:/stanford-corenlp-full-2018-02-27/')
    fw = open(path + '/data/' + data + '.ans', 'w')
    f = open(path + '/data/' + data + '.a')
    lines = f.readlines()
    for line in lines:
        line = re.sub(r'[^A-Za-z0-9_ ]','',line)
        line = line.lower().split(' ')
        j = len(line)
        for i in range(j):  
            for div in ['the','a','an']:
                if line[i] == div:
                    line[i] = ''
            if line[i] != '':
                w = nlp.lemmas(line[i])
                line[i] = w[0][1]
            if i != j-1:
                if line[i] != '' and line[i+1] != '':
                    line[i] = line[i]+' '
            else:
                line[i] = line[i]+'\n'
            fw.write(line[i])
    f.close()
    fw.close()

#答句预处理
if os.path.isfile(path + '/data/Train.ans') == False:
    pretreat_a('Train')
    pretreat_a('Valid')

#函数：问答向量化
print('Turning Q&A into vectors...')
global train_que, train_ans, valid_que, valid_ans, test_que, quelist, anslist
train_que, train_ans, valid_que, valid_ans, test_que, quelist, anslist = [], [], [], [], [], [], []
def qa_vec(data,qa):
    with open(path + '/data/' + data + '.' + qa) as f:
        for line in f:
            line = line.replace('magnetic resonance image','mri').replace('compute tomography','ct').replace('positron emission tomography','pet').replace('vena cava','vc').strip()
            if qa == 'que':
                quelist.append(line)
                if data == 'Train':
                    train_que.append(line)
                elif data == 'Valid':
                    valid_que.append(line)
                else:
                    test_que.append(line)
            else:
                anslist.append(line)
                if data == 'Train':
                    for j in range(increase + 1):
                        train_ans.append(line)
                else:
                    valid_ans.append(line)
    f.close()

#问答向量化
qa_vec('Train','que')
qa_vec('Train','ans')
qa_vec('Valid','que')
qa_vec('Valid','ans')
#qa_vec('Test','que')

#分词
tokenizer_q = text.Tokenizer(num_words=dic)
tokenizer_q.fit_on_texts(quelist)
trainque_feature = tokenizer_q.texts_to_sequences(train_que)  #文本向量化
validque_feature = tokenizer_q.texts_to_sequences(valid_que)
#testque_feature = tokenizer_q.texts_to_sequences(test_que)
word_index = tokenizer_q.word_index
trainque_feature = sequence.pad_sequences(trainque_feature, maxlen, padding='post', value=0, truncating='post')  #统一序列长度
validque_feature = sequence.pad_sequences(validque_feature, maxlen, padding='post', value=0, truncating='post')
#testque_feature = sequence.pad_sequences(testque_feature, maxlen, padding='post', value=0, truncating='post')
#trainall_feature = np.concatenate([trainque_feature, validque_feature], axis=0)

tokenizer_a = text.Tokenizer(num_words=dic)
tokenizer_a.fit_on_texts(anslist)
trainans_feature = tokenizer_a.texts_to_sequences(train_ans)
validans_feature = tokenizer_a.texts_to_sequences(valid_ans)
trainans_feature = sequence.pad_sequences(trainans_feature, maxlen, padding='post', value=0, truncating='post')
validans_feature = sequence.pad_sequences(validans_feature, maxlen, padding='post', value=0, truncating='post')
trainans_hot = keras.utils.to_categorical(trainans_feature, dic+1)  #one-hot
validans_hot = keras.utils.to_categorical(validans_feature, dic+1)
#trainall_hot = np.concatenate([trainans_hot, validans_hot], axis=0)

#读取图像特征
trainimg_feature = np.load(path + '/data/Train_I3.npy')
validimg_feature = np.load(path + '/data/Valid_I3.npy')
#testimg_feature = np.load(path + '/data/Test_I3.npy')
#trainall_img = np.concatenate([trainimg_feature, validimg_feature], axis=0)

#Word2Vec
ql = open(path + '/data/que.list', 'w',encoding='utf-8')
for i in range (len(quelist)):
    ql.write(quelist[i]+'\n')
ql.close()
sentences = word2vec.Text8Corpus(path + '/data/que.list')
model = word2vec.Word2Vec(sentences, size=50, min_count=1)
word2idx = {"_PAD": 0}
vocab_list = [(k, model.wv[k]) for k, v in model.wv.vocab.items()]
embeddingw2v_matrix = np.zeros((len(model.wv.vocab.items()) + 1, model.vector_size))
for i in range(len(vocab_list)):
    word = vocab_list[i][0]
    word2idx[word] = i + 1
    embeddingw2v_matrix[i + 1] = vocab_list[i][1]

#pos-enc
class Position_Embedding(Layer):
    def __init__(self, min_timescale=1.0, max_timescale=1.0e4, **kwargs):
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        super(Position_Embedding, self).__init__(**kwargs)
    def get_timing_signal_1d(self, length, channels):
        position = tf.to_float(tf.range(length))
        num_timescales = channels // 2
        log_timescale_increment = (math.log(float(self.max_timescale) / float(self.min_timescale)) / (tf.to_float(num_timescales) - 1))
        inv_timescales = self.min_timescale * tf.exp(tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
        signal = tf.reshape(signal, [1, length, channels])
        return signal
    def add_timing_signal_1d(self, x):
        length = tf.shape(x)[1]
        channels = tf.shape(x)[2]
        signal = self.get_timing_signal_1d(length, channels)
        return x + signal
    def call(self, x, mask=None):
        return self.add_timing_signal_1d(x)
    def compute_output_shape(self, input_shape):
        return input_shape

#LayerNorm
class LayerNormalization(Layer):
    def __init__(self, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
    def build(self, input_shape):
        self.scale = self.add_weight(name='layer_norm_scale',
                                    shape=(input_shape[-1]),
                                    initializer=tf.ones_initializer(),
                                    trainable=True)
        self.bias = self.add_weight(name='layer_norm_bias',
                                    shape=(input_shape[-1]),
                                    initializer=tf.zeros_initializer(),
                                    trainable=True)
        super(LayerNormalization, self).build(input_shape)
    def call(self, x, mask=None, training=None):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)
        norm_x = (x - mean) * tf.rsqrt(variance + K.epsilon())
        return norm_x * self.scale + self.bias
    def compute_output_shape(self, input_shape):
        return input_shape

#Attention
class Attention(Layer):
    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head*size_per_head
        super(Attention, self).__init__(**kwargs)
    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ', 
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK', 
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV', 
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)
    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:,0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape)-2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12    
    def call(self, x):
        #如果只传入Q_seq,K_seq,V_seq，那么就不做Mask
        #如果同时传入Q_seq,K_seq,V_seq,Q_len,V_len，那么对多余部分做Mask
        if len(x) == 3:
            Q_seq,K_seq,V_seq = x
            Q_len,V_len = None,None
        elif len(x) == 5:
            Q_seq,K_seq,V_seq,Q_len,V_len = x
        #对Q、K、V做线性变换
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0,2,1,3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0,2,1,3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0,2,1,3))
        #计算内积，然后mask，然后softmax
        A = K.batch_dot(Q_seq, K_seq, axes=[3,3]) / self.size_per_head**0.5
        A = K.permute_dimensions(A, (0,3,2,1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0,3,2,1))    
        A = K.softmax(A)
        #输出并mask
        O_seq = K.batch_dot(A, V_seq, axes=[3,2])
        O_seq = K.permute_dimensions(O_seq, (0,2,1,3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)  

#que&img attention
class context2query_attention(Layer):
    def __init__(self, output_dim, cont_limit, ques_limit, dropout, **kwargs):
        self.output_dim=output_dim
        self.cont_limit = cont_limit
        self.ques_limit = ques_limit
        self.dropout = dropout
        super(context2query_attention, self).__init__(**kwargs)
    def build(self, input_shape):
        # input_shape: [(None, 400, 128), (None, 50, 128)]
        self.W0 = self.add_weight(name='W0',
                                  shape=(input_shape[0][-1], 1),
                                  initializer='glorot_uniform',
                                  regularizer=regularizers.l2(3e-7),
                                  trainable=True)
        self.W1 = self.add_weight(name='W1',
                                  shape=(input_shape[1][-1], 1),
                                  initializer='glorot_uniform',
                                  regularizer=regularizers.l2(3e-7),
                                  trainable=True)
        self.W2 = self.add_weight(name='W2',
                                  shape=(1, 1, input_shape[0][-1]),
                                  initializer='glorot_uniform',
                                  regularizer=regularizers.l2(3e-7),
                                  trainable=True)
        self.bias = self.add_weight(name='linear_bias',
                                    shape=(input_shape[1][1],),
                                    initializer='zero',
                                    regularizer=regularizers.l2(3e-7),
                                    trainable=True)
        super(context2query_attention, self).build(input_shape)
    def Mask(self, inputs, seq_len, axis=1, time_dim=1, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            seq_len=K.cast(seq_len,tf.int32)
            mask = K.one_hot(seq_len[:, 0], K.shape(inputs)[time_dim])
            mask = 1 - K.cumsum(mask, 1)
            mask = K.expand_dims(mask, axis)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12
    def call(self, x, mask=None):
        x_cont, x_ques, cont_len, ques_len = x
        # get similarity matrix S
        subres0 = K.tile(K.dot(x_cont, self.W0), [1, 1, self.ques_limit])
        subres1 = K.tile(K.permute_dimensions(K.dot(x_ques, self.W1), pattern=(0, 2, 1)), [1, self.cont_limit, 1])
        subres2 = K.batch_dot(x_cont * self.W2, K.permute_dimensions(x_ques, pattern=(0, 2, 1)))
        S = subres0 + subres1 + subres2
        S += self.bias
        S_ = tf.nn.softmax(self.Mask(S, ques_len, axis=1, time_dim=2, mode='add'))
        S_T = K.permute_dimensions(tf.nn.softmax(self.Mask(S, cont_len, axis=2, time_dim=1, mode='add'), axis=1), (0, 2, 1))
        c2q = tf.matmul(S_, x_ques)
        q2c = tf.matmul(tf.matmul(S_, S_T), x_cont)
        result = K.concatenate([x_cont, c2q, x_cont * c2q, x_cont * q2c], axis=-1)
        return result
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)

#final
lab = 'final'
print('Building model...')
    
#问答模型
encoded_ques = Input(shape=(maxlen,))
mask_ques = Lambda(lambda x: tf.cast(x,tf.bool))(encoded_ques)
len_ques = Lambda(lambda x: tf.expand_dims(tf.reduce_sum(tf.cast(x,tf.int32),axis=1),axis=1))(mask_ques)
embed_ques = Embedding(len(embeddingw2v_matrix),50,input_length=maxlen,weights=[embeddingw2v_matrix],trainable=False)(encoded_ques)
drop_ques = Dropout(0.5)(embed_ques)
poenc_ques = Position_Embedding()(drop_ques)
norm1_ques = LayerNormalization()(poenc_ques)
depth1_ques = Bidirectional(LSTM(64,return_sequences=True,dropout=0.5))(norm1_ques)
norm2_ques = LayerNormalization()(depth1_ques)
att_ques = Attention(8,16)([norm2_ques,norm2_ques,norm2_ques])
norm3_ques = LayerNormalization()(att_ques)
dense_ques = Dense(128,activation='linear')(norm3_ques)

#图像模型
encoded_image = Input(shape=(2048,))
mask_image = Lambda(lambda x: tf.cast(x,tf.bool))(encoded_image)
len_image = Lambda(lambda x: tf.expand_dims(tf.reduce_sum(tf.cast(x,tf.int32),axis=1),axis=1))(mask_image)
repeat_image = RepeatVector(maxlen)(encoded_image)
time_image = TimeDistributed(Dense(dim,activation='relu'))(repeat_image)
poenc_image = Position_Embedding()(time_image)
norm1_image = LayerNormalization()(poenc_image)
depth1_image = Conv1D(128,(1,),padding='valid',activation='relu')(norm1_image)
depth1_image = Conv1D(128,(1,),dilation_rate=2,padding='valid',activation='relu')(depth1_image)
pool1_image = AveragePooling1D(1)(depth1_image)
norm2_image = LayerNormalization()(pool1_image)
depth2_image = Conv1D(128,(1,),padding='valid',activation='relu')(norm2_image)
depth2_image = Conv1D(128,(1,),dilation_rate=2,padding='valid',activation='relu')(depth2_image)
pool2_image = AveragePooling1D(1)(depth2_image)
norm3_image = LayerNormalization()(pool2_image)
depth3_image = Conv1D(128,(1,),padding='valid',activation='relu')(norm3_image)
depth3_image = Conv1D(128,(1,),dilation_rate=2,padding='valid',activation='relu')(depth3_image)
pool3_image = AveragePooling1D(1)(depth3_image)
norm4_image = LayerNormalization()(pool3_image)
att_image = Attention(8,16)([norm4_image,norm4_image,norm4_image])
norm5_image = LayerNormalization()(att_image)
dense_image = Dense(128,activation='linear')(norm5_image)

#合并模型
att1_model = context2query_attention(512,maxlen,maxlen,0.0)([dense_ques,dense_image,len_ques,len_image])
att1_model = Conv1D(128,1,activation='linear')(att1_model)
poenc_model = Position_Embedding()(att1_model)
norm1_model = LayerNormalization()(poenc_model)
depth1_model = Conv1D(128,(1,),padding='valid',activation='relu')(norm1_model)
depth1_model = Conv1D(128,(1,),dilation_rate=2,padding='valid',activation='relu')(depth1_model)
pool1_model = AveragePooling1D(1)(depth1_model)
norm2_model = LayerNormalization()(pool1_model)
att2_model = Attention(8,16)([norm2_model,norm2_model,norm2_model])
norm3_model = LayerNormalization()(att2_model)
dense_model = Dense(128,activation='linear')(norm3_model)
output_model = TimeDistributed(Dense(dic+1,activation='softmax'))(dense_model)
vqa_model = Model(inputs=[encoded_ques,encoded_image],outputs=output_model)
vqa_model.summary()

#编译模型
rms = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
vqa_model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=[metrics.categorical_accuracy])
if os.path.exists(path + '/data/' + lab + '/') == False:
    os.mkdir(path + '/data/' + lab + '/')
filepath = path + '/data/' + lab + '/model.hdf5'
checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True, mode='min')
plateau = ReduceLROnPlateau(factor=0.1, patience=3, verbose=1, min_delta=1e-7, min_lr=0.00002)
stop = EarlyStopping(patience=7, verbose=1)
callbacks_list = [checkpoint,plateau,stop]

#训练模型
if os.path.isfile(filepath) == False:
    print('Training model...')
    history = vqa_model.fit([trainque_feature,trainimg_feature],trainans_hot,epochs=100,batch_size=256,
                                 validation_data=([validque_feature,validimg_feature],validans_hot),callbacks=callbacks_list,verbose=1)
    fig = plt.figure()
    fig.set_dpi(300)    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()
    fig.savefig(path + '/data/' + lab + '/loss.png')
else:
    print('Loading model...')  #载入模型
    vqa_model.load_weights(filepath, by_name=True)  
    json_string = vqa_model.to_json()  
    vqa_model = model_from_json(json_string)

#测试结果
print('Answer generating...')
dic_q = tokenizer_q.word_index  #字典-索引
ind_q ={value:key for key, value in dic_q.items()}  #索引-字典
dic_a = tokenizer_a.word_index
ind_a ={value:key for key, value in dic_a.items()}

#修正输出词
#for i in range(dic):
    #if ind_a[i+1] == 'po':
        #ind_a[i+1] = 't2'
    #if ind_a[i+1] == 'num':
        #ind_a[i+1] = '10'

def kbeam(ans,maxlen,numbering,k=3):   
    k_beam = [(0, [0]*(maxlen+1))]
    for l in range(maxlen):
        all_k_beams = []
        for prob, sent_predict in k_beam:
            predicted = ans[numbering]
            possible_k = predicted[l].argsort()[-k:][::-1]
            all_k_beams += [
                (
                    sum(np.log(predicted[i][sent_predict[i+1]]) for i in range(l)) + np.log(predicted[l][next_wid]),
                    list(sent_predict[:l+1])+[next_wid]+[0]*(maxlen-l-1)
                )
                for next_wid in possible_k
            ]
        k_beam = sorted(all_k_beams)[-k:]
    return k_beam

#beam search
def beam_ans(data, num):
    if data == 'Train':
        ans = vqa_model.predict([trainque_feature, trainimg_feature])
        feature = trainque_feature
    elif data == 'Valid':
        ans = vqa_model.predict([validque_feature, validimg_feature])
        feature = validque_feature
    elif data == 'Test':
        ans = vqa_model.predict([testque_feature, testimg_feature])
        feature = testque_feature
    fp = open(path + '/data/' + data + '.beam', 'w')
    for h in range(num):
        if data == 'Train':
            i = h*(increase+1)  #训练集还原
        else:
            i = h
        if kbeam(ans,maxlen,i)[-1][-1][1] == 0:
            fp.write('abnormality\n')  #低频词用“异常”替换
        elif (feature[i][0] == dic_q['do'] or feature[i][0] == dic_q['be']) and (np.argmax(ans[i][0],axis=0) != dic_a['yes'] and np.argmax(ans[i][0],axis=0) != dic_a['no']):
            fp.write('no\n')    #是非题如果答案不是“yes”或“no”的替换
        else:
            for j in range(maxlen):
                an = kbeam(ans,maxlen,i)[-1][-1][j+1]
                if j != maxlen-1:
                    anext = kbeam(ans,maxlen,i)[-1][-1][j+2]
                    if an != 0 and anext != 0:  #前后均有词
                        if an == anext:
                            fp.write('')  #删除重复词
                        else:
                            fp.write(ind_a[an] + ' ')
                    elif an != 0 and anext == 0:  #前有词后无词
                        fp.write(ind_a[an])
                    elif an == 0 and anext != 0:  #前无词后有词
                        fp.write(' ')
                    else:  #前后均无词
                        fp.write('')
                else:
                    if an != 0:
                        fp.write(ind_a[an] + '\n')
                    else:
                        fp.write('\n')
    fp.close()

#greedy algorithm
def greedy_ans(data, num):
    if data == 'Train':
        ans = vqa_model.predict([trainque_feature, trainimg_feature])
        feature = trainque_feature
    elif data == 'Valid':
        ans = vqa_model.predict([validque_feature, validimg_feature])
        feature = validque_feature
    elif data == 'Test':
        ans = vqa_model.predict([testque_feature, testimg_feature])
        feature = testque_feature
    fp = open(path + '/data/' + data + '.greedy', 'w')
    for h in range(num):
        if data == 'Train':
            i = h*(increase+1)  #训练集还原
        else:
            i = h
        if np.argmax(ans[i][0],axis=0) == 0:
            fp.write('abnormality\n')  #低频词用“异常”替换
        elif (feature[i][0] == dic_q['do'] or feature[i][0] == dic_q['be']) and (np.argmax(ans[i][0],axis=0) != dic_a['yes'] and np.argmax(ans[i][0],axis=0) != dic_a['no']):
            fp.write('no\n')    #是非题如果答案不是“yes”或“no”的替换
        else:
            for j in range(maxlen):
                an = np.argmax(ans[i][j],axis=0)
                if j != maxlen-1:
                    anext = np.argmax(ans[i][j+1],axis=0)
                    if an != 0 and anext != 0:  #前后均有词
                        if an == anext:
                            fp.write('')  #删除重复词
                        else:
                            fp.write(ind_a[an] + ' ')
                    elif an != 0 and anext == 0:  #前有词后无词
                        fp.write(ind_a[an])
                    elif an == 0 and anext != 0:  #前无词后有词
                        fp.write(' ')
                    else:  #前后均无词
                        fp.write('')
                else:
                    if an != 0:
                        fp.write(ind_a[an] + '\n')
                    else:
                        fp.write('\n')
    fp.close()

if os.path.isfile(path + '/data/Valid.greedy') == False:
    beam_ans('Train', num_train)
    greedy_ans('Train', num_train)
    beam_ans('Valid', num_valid)
    greedy_ans('Valid', num_valid)
    #beam_ans('Test', num_test)
    #greedy_ans('Test', num_test)

#删除多余文件
def delete_file(data):
    for name in [path + '/data/' + data + '.num', path + '/data/' + data + '.name', path + '/data/' + data + '.q']:  #保留对比
        os.remove(name)
#     if data != 'Test':
#         for name in [path + '/data/' + data + '.ans']:
#             os.remove(name)
if os.path.isfile(path + '/data/Valid.greedy') == True:
    try:
        delete_file('Train')
        delete_file('Valid')
#        delete_file('Test')
    except:
        print('fail to delete some data files...')
print('Finished.')
