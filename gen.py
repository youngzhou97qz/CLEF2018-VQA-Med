import sys, keras, re, os
import numpy as np
import matplotlib.pyplot as plt
from pattern.en import lemma
from keras import optimizers, metrics
from keras.layers import normalization, Input, LSTM, Dense, RepeatVector, Bidirectional, TimeDistributed, Embedding, concatenate
from keras.models import Model, model_from_json
from keras.preprocessing import image, text, sequence
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import ModelCheckpoint
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input

#常量
path = 'C:/Users/zhou yangyang/VQA'
quelen = 9  #序列长度
anslen = 6
que_dic = 1000  #频数4以上
ans_dic = 1000
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
                data_que.write(line.split('	')[2].replace('?', '').replace('(', '').replace(')', '').replace('  ', ' ').replace('/', ' ').replace(',', '').replace('-', ' ').lower().replace('computed tomography', 'ct').replace('magnetic resonance image', 'mri').replace('magnetic resonance imaging', 'mri').replace('vena cava', 'vc').replace('cava vena', 'vc').replace('iv contrast', 'intravenous contrast').replace('medial collateral ligament', 'mcl').replace('positron emission tomography', 'pet').replace('t1w1', 't1 weighted').replace('t1wi', 't1 weighted').replace('t1w', 't1 weighted').replace('t2w1', 't2 weighted').replace('t2wi', 't2 weighted').replace('t2w', 't2 weighted')+'\n')
                data_ans.write(line.split('	')[3].replace('?', '').replace('(', '').replace(')', '').replace('  ', ' ').replace('/', ' ').replace(',', '').replace('-', ' ').lower().replace('computed tomography', 'ct').replace('magnetic resonance image', 'mri').replace('magnetic resonance imaging', 'mri').replace('vena cava', 'vc').replace('cava vena', 'vc').replace('iv contrast', 'intravenous contrast').replace('medial collateral ligament', 'mcl').replace('positron emission tomography', 'pet').replace('t1w1', 't1 weighted').replace('t1wi', 't1 weighted').replace('t1w', 't1 weighted').replace('t2w1', 't2 weighted').replace('t2wi', 't2 weighted').replace('t2w', 't2 weighted'))
            else:
                data_que.write(line.split('	')[2].replace('?', '').replace('(', '').replace(')', '').replace('  ', ' ').replace('/', ' ').replace(',', '').replace('-', ' ').lower().replace('computed tomography', 'ct').replace('magnetic resonance image', 'mri').replace('magnetic resonance imaging', 'mri').replace('vena cava', 'vc').replace('cava vena', 'vc').replace('iv contrast', 'intravenous contrast').replace('medial collateral ligament', 'mcl').replace('positron emission tomography', 'pet').replace('t1w1', 't1 weighted').replace('t1wi', 't1 weighted').replace('t1w', 't1 weighted').replace('t2w1', 't2 weighted').replace('t2wi', 't2 weighted').replace('t2w', 't2 weighted'))
    data_num.close()
    data_name.close()
    data_que.close()
    if data != 'Test':
        data_ans.close()

#处理文件
print('Generating data files...')
if os.path.isfile(path + '/data/Test.num') == False:
    data_processing('Train')
    data_processing('Valid')
    data_processing('Test')

#计算文件数量
num_train = len(open(path + '/data/Train.num', 'r').readlines())
num_valid = len(open(path + '/data/Valid.num', 'r').readlines())
num_test = len(open(path + '/data/Test.num', 'r').readlines())

#函数：处理图像
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
                if data != 'Test':
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
            process_bar = data +': [' + '>' * num_arrow + '-' * num_line + ']'\
                      + '%.2f' % percent + '%' + '\r'
            sys.stdout.write(process_bar) #这两句打印字符到终端
            sys.stdout.flush()
    im = np.array(im)
    im = np.squeeze(im)  #去掉为1的维度
    np.save(path + '/data/' + data + '_im.npy', im)
    print('\n')
    f.close()

#保存图像特征
print('Turning images into vectors...')
if os.path.isfile(path + '/data/Test_im.npy') == False:
    model_1 = InceptionResNetV2(weights='imagenet')  #加载InceptionResNetV2模型
    datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1, zoom_range=0.1)  #图像增强器
    img_processing('Train')
    img_processing('Valid')
    img_processing('Test')

#函数：词形还原，数字替换，停用词删除
def pretreat_q(data):
    global train_que, valid_que, test_que, quelist
    fw = open(path + '/data/' + data + '.que', 'w')
    f = open(path + '/data/' + data + '.q')
    lines = f.readlines()
    for line in lines:
        line = line.replace('\n', '').split(' ')
        j = len(line)
        for i in range(j):
            for div in ['the','of','in','and','a','with','to','an','at','on','from','after','into']:  #停用词删除'the','of','in','and','a','with','to','an','at','on','from','after','into'
                if line[i] == div:
                    line[i] = ''
            if re.match(r'[0-9]+', line[i]):  #数字替换
                line[i] = 'num'
            elif re.match(r'[a-z][0-9]+', line[i]):
                line[i] = 'pos'
            line[i] = lemma(line[i])  #词形还原
            if line[i] == 'have':
                line[i] = ''
            if i != j-1:
                if line[i] != '' and line[i+1] != '':
                    line[i] = line[i]+' '
            else:
                line[i] = line[i]+'\n'
            fw.write(line[i])
    f.close()
    fw.close()

#问句预处理
if os.path.isfile(path + '/data/Test.que') == False:
    pretreat_q('Train')
    pretreat_q('Valid')
    pretreat_q('Test')

#函数：答句
def pretreat_a(data):
    global train_ans, valid_ans, anslist
    fw = open(path + '/data/' + data + '.ans', 'w')
    f = open(path + '/data/' + data + '.a')
    lines = f.readlines()
    for line in lines:
        line = line.replace('\n', '').split(' ')  #手动分词
        j = len(line)
        for i in range(j):  #'the','and','a','an','with'
            for div in ['the','and','a','an','with']:
                if line[i] == div:
                    line[i] = ''
            #if re.match(r'[0-9]+', line[i]):
                #line[i] = 'num'
            #elif re.match(r'[a-z][0-9]+', line[i]):
                #line[i] = 'pos'
            line[i] = lemma(line[i])
            if line[i] == 'have':
                line[i] = ''
            if i != j-1:
                if line[i] != '' and line[i+1] != '':
                    line[i] = line[i]+' '
            else:
                line[i] = line[i]+'\n'
            fw.write(line[i])
    f.close()
    fw.close()

#答句预处理
if os.path.isfile(path + '/data/Valid.ans') == False:
    pretreat_a('Train')
    pretreat_a('Valid')

#加载问答
train_que, train_ans, valid_que, valid_ans, test_que, quelist, anslist = [], [], [], [], [], [], []
f = open(path + '/data/Train.que')
lines = f.readlines()
for line in lines:
    line = line.strip()
    quelist.append(line)
    train_que.append(line)
    for div in [' show',' scan',' image',' demonstrate',' axial',' reveal',' contrast',' enhance',' large',' enhancement', ' area',' section',' thi',' well',' soft',' side', ' view',' small',' complete',' postoperative']:
        train_que.append(line.replace(div, ''))  #问句增强
f.close()
f = open(path + '/data/Valid.que')
lines = f.readlines()
for line in lines:
    line = line.strip()
    quelist.append(line)
    valid_que.append(line)
    for div in [' show',' scan',' image',' demonstrate',' axial',' reveal',' contrast',' enhance',' large',' enhancement', ' area',' section',' thi',' well',' soft',' side', ' view',' small',' complete',' postoperative']:
        valid_que.append(line.replace(div, ''))
f.close()
f = open(path + '/data/Test.que')
lines = f.readlines()
for line in lines:
    line = line.strip()
    test_que.append(line)
    quelist.append(line)
f.close()
f = open(path + '/data/Train.ans')
lines = f.readlines()
for line in lines:
    line = line.strip()
    for j in range(increase+1):
        train_ans.append(line)
    anslist.append(line)
f.close()
f = open(path + '/data/Valid.ans')
lines = f.readlines()
for line in lines:
    line = line.strip()
    for j in range(increase+1):
        valid_ans.append(line)
    anslist.append(line)
f.close()

#分词
print('Turning Q&A into vectors...')
tokenizer_q = text.Tokenizer(num_words=que_dic)
tokenizer_q.fit_on_texts(quelist)
trainque_feature = tokenizer_q.texts_to_sequences(train_que)  #文本向量化
validque_feature = tokenizer_q.texts_to_sequences(valid_que)
testque_feature = tokenizer_q.texts_to_sequences(test_que)
trainque_feature = sequence.pad_sequences(trainque_feature, quelen, padding='post', value=0, truncating='post')  #统一序列长度
validque_feature = sequence.pad_sequences(validque_feature, quelen, padding='post', value=0, truncating='post')
testque_feature = sequence.pad_sequences(testque_feature, quelen, padding='post', value=0, truncating='post')
trainall_feature = np.concatenate([trainque_feature, validque_feature], axis=0)

tokenizer_a = text.Tokenizer(num_words=ans_dic)
tokenizer_a.fit_on_texts(anslist)
trainans_feature = tokenizer_a.texts_to_sequences(train_ans)
validans_feature = tokenizer_a.texts_to_sequences(valid_ans)
trainans_feature = sequence.pad_sequences(trainans_feature, anslen, padding='post', value=0, truncating='post')
validans_feature = sequence.pad_sequences(validans_feature, anslen, padding='post', value=0, truncating='post')
trainans_hot = keras.utils.to_categorical(trainans_feature, ans_dic+1)  #one-hot
validans_hot = keras.utils.to_categorical(validans_feature, ans_dic+1)
trainall_hot = np.concatenate([trainans_hot, validans_hot], axis=0)

#读取图像特征数据
trainimg_feature = np.load(path + '/data/Train_im.npy')
validimg_feature = np.load(path + '/data/Valid_im.npy')
testimg_feature = np.load(path + '/data/Test_im.npy')
trainall_img = np.concatenate([trainimg_feature, validimg_feature], axis=0)

#注意力模型
print('Building model...')

#图像模型
encoded_image = Input(shape=(1000,))
dense_image = Dense(dim*2)(encoded_image)
batch_image = normalization.BatchNormalization()(dense_image)
repeat_image = RepeatVector(maxlen)(batch_image)

#问答模型
encode_question = Input(shape=(maxlen,))
embed_question = Embedding(input_dim=dic, output_dim=dim, input_length=maxlen)(encode_question)
lstm_question = Bidirectional(LSTM(dim, return_sequences=True,dropout=0.5))(embed_question)
batch_question = normalization.BatchNormalization()(lstm_question)

#注意力机制
merge_attention = add([repeat_image, batch_question])
act1_attention = Activation('tanh')(merge_attention)
dense1_attention = TimeDistributed(Dense(1))(act1_attention)
flat_attention = Flatten()(dense1_attention)
act2_attention = Activation('softmax')(flat_attention)
repeat_attention = RepeatVector(dim*2)(act2_attention)
permute_attention = Permute((2, 1))(repeat_attention)

#合并模型
merge_model = concatenate([batch_question, permute_attention])
batch_model = normalization.BatchNormalization()(merge_model)
output_model = TimeDistributed(Dense(dic+1, activation='softmax'))(batch_model)
vqa_model = Model(inputs=[encoded_image, encode_question], outputs=output_model)
vqa_model.summary()

#编译模型
nadam = optimizers.Nadam()
vqa_model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=[metrics.categorical_accuracy])

# 保存模型
filepath = path + '/data/final注意96序列4词频有介词.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

#训练模型
if os.path.isfile(path + '/data/final注意96序列4词频有介词.hdf5') == False:
    print('Training model...')
    history = vqa_model.fit([trainall_img, trainall_feature], trainall_hot, epochs=300, batch_size=256, validation_data=([validimg_feature, validque_feature], validans_hot), callbacks=callbacks_list, verbose=1)
    
    #可视化训练过程
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()
else:
    print('Loading model...')  #载入模型
    vqa_model.load_weights(path + '/data/final注意96序列4词频有介词.hdf5', by_name=True)
    json_string = vqa_model.to_json()
    vqa_model = model_from_json(json_string)
	
#测试结果
print('Answer generating...')
dic_q = tokenizer_q.word_index  #字典-索引
ind_q ={value:key for key, value in dic_q.items()}  #索引-字典
dic_a = tokenizer_a.word_index
ind_a ={value:key for key, value in dic_a.items()}

#修正输出词
def dic_fix_s(word):
    for i in range(ans_dic):
        if ind_a[i+1] == word:
            ind_a[i+1] = ind_a[i+1]+'s'  #不完整词还原
for div in ['thi','sinu','pelvi','pancrea','thrombosi','necrosi','corpu','stenosi','metastasi','ye','venou','thrombu','uteru','thalamu','pon','esophagu','cavernou','sac','meniscu','los','conu','rib','nucleu','fibrosi','ramu','rectu','agenesi','hi','bulbu','osseou','osteomyeliti','edematou','tuberculosi','plexu','clivu','pneumocephalu','atelectasi','vermi','globu','sclerosi','iliopsoa','psoa','supraspinatu','hydronephrosi']:
    dic_fix_s(div)
for i in range(ans_dic):
    if ind_a[i+1] == 'vertebra':
        ind_a[i+1] = ind_a[i+1]+'e'
    if ind_a[i+1] == 'iv':
        ind_a[i+1] = ind_a[i+1]+'c'
    if ind_a[i+1] == 'axi':
        ind_a[i+1] = ind_a[i+1]+'al'
    if ind_a[i+1] == 'pleura':
        ind_a[i+1] = ind_a[i+1]+'l'
    if ind_a[i+1] == 'axilla':
        ind_a[i+1] = ind_a[i+1]+'ry'
    #if ind_a[i+1] == 'po':
        #ind_a[i+1] = 't2'
    #if ind_a[i+1] == 'num':
        #ind_a[i+1] = '10'
        
#生成答句文件
def final_ans(data, num):
    if data == 'Train':
        ans = vqa_model.predict([trainimg_feature, trainque_feature])
        feature = trainque_feature
    elif data == 'Valid':
        ans = vqa_model.predict([validimg_feature, validque_feature])
        feature = validque_feature
    elif data == 'Test':
        ans = vqa_model.predict([testimg_feature, testque_feature])
        feature = testque_feature
    fp = open(path + '/data/' + data + '.fn', 'w')
    for h in range(num):
        if data != 'Test':
            i = h*(increase+1)  #训练集还原
        else:
            i = h
        fp.write(str(h+1))  #序号
        fp.write('	')
        fe = open(path + '/data/' + data + '.name')
        for k in fe.readlines()[h]:
            k = k.strip()
            fp.write(k)  #图片名
        fp.write('	')
        fe.close()
        if np.argmax(ans[i][0],axis=0) == 0:
            fp.write('abnormality\n')  #低频词用“异常”替换
        elif (feature[i][0] == dic_q['do'] or feature[i][0] == dic_q['be']) and (np.argmax(ans[i][0],axis=0) != dic_a['ye'] and np.argmax(ans[i][0],axis=0) != dic_a['no']):
            fp.write('no\n')    #是非题如果答案不是“yes”或“no”的替换
        else:
            for j in range(anslen):
                an = np.argmax(ans[i][j],axis=0)
                if j != anslen-1:
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
    
if os.path.isfile(path + '/data/Test.fn') == False:
    final_ans('Train', num_train)
    final_ans('Valid', num_valid)
    final_ans('Test', num_test)

#删除多余文件
def delete_file(data):
    for name in [path + '/data/' + data + '.num', path + '/data/' + data + '.name', path + '/data/' + data + '.que', path + '/data/' + data + '.q']:  #保留对比
        os.remove(name)
    if data != 'Test':
        for name in [path + '/data/' + data + '.a']: #, path + '/data/' + data + '.ans'保留对比
            os.remove(name)
if os.path.isfile(path + '/data/Valid.ans') == True:
    try:
        delete_file('Train')
        delete_file('Valid')
        delete_file('Test')
    except:
        print('fail to delete some data files...')
#结束
print('Finished.')
