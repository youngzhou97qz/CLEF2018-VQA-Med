import sys, os, keras, re
import numpy as np
import matplotlib.pyplot as plt
from pattern.en import lemma
from keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed, Embedding
from keras.models import Model, model_from_json
from keras.preprocessing import image, text, sequence
from keras.callbacks import ModelCheckpoint
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input

#常量
path = 'C:/Users/zhou yangyang/VQA'
maxlen = 13  #序列长度
dic_num = 1000

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
                data_que.write(line.split('	')[2].replace('?', '').replace('(', '').replace(')', '').replace('  ', ' ').replace('/', ' ').replace(',', '').replace('-', ' ').lower().replace('computed tomography', 'ct').replace('magnetic resonance imaging', 'mri').replace('inferior vena cava', 'ivc').replace('iv contrast', 'intravenous contrast').replace('medial collateral ligament', 'mcl').replace('right ventricle', 'rv').replace('t1w1', 't1 weighted').replace('t1wi', 't1 weighted').replace('t1w', 't1 weighted').replace('t2w1', 't2 weighted').replace('t2wi', 't2 weighted').replace('t2w', 't2 weighted')+'\n')
                data_ans.write(line.split('	')[3].replace('?', '').replace('(', '').replace(')', '').replace('  ', ' ').replace('/', ' ').replace(',', '').replace('-', ' ').lower().replace('computed tomography', 'ct').replace('magnetic resonance imaging', 'mri').replace('inferior vena cava', 'ivc').replace('iv contrast', 'intravenous contrast').replace('medial collateral ligament', 'mcl').replace('right ventricle', 'rv').replace('t1w1', 't1 weighted').replace('t1wi', 't1 weighted').replace('t1w', 't1 weighted').replace('t2w1', 't2 weighted').replace('t2wi', 't2 weighted').replace('t2w', 't2 weighted'))
            else:
                data_que.write(line.split('	')[2].replace('?', '').replace('(', '').replace(')', '').replace('  ', ' ').replace('/', ' ').replace(',', '').replace('-', ' ').lower().replace('computed tomography', 'ct').replace('magnetic resonance imaging', 'mri').replace('inferior vena cava', 'ivc').replace('iv contrast', 'intravenous contrast').replace('medial collateral ligament', 'mcl').replace('right ventricle', 'rv').replace('t1w1', 't1 weighted').replace('t1wi', 't1 weighted').replace('t1w', 't1 weighted').replace('t2w1', 't2 weighted').replace('t2wi', 't2 weighted').replace('t2w', 't2 weighted'))
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
                x = preprocess_input(x)
                y = model_1.predict(x)  #提取特征量
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
            for div in ['the','of','in','and','a','with','to','an','at','on','have']:
                if line[i] == div:
                    line[i] = ''
            if re.match(r'[0-9]+', line[i]):
                line[i] = 'num'
            elif re.match(r'[a-z][0-9]+', line[i]):
                line[i] = 'pos'
            line[i] = lemma(line[i])
            if line[i] == 'have':
                line[i] = ''
            if i == (j-1):
                line[i] = line[i]+'\n'
            else:
                line[i] = line[i]+' '
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
        line = line.replace('\n', '').split(' ')
        j = len(line)
        for i in range(j):
            for div in ['the','of','in','and','a','with','to','an','at','on','have']:
                if line[i] == div:
                    line[i] = ''
            if re.match(r'[0-9]+', line[i]):
                line[i] = 'num'
            elif re.match(r'[a-z][0-9]+', line[i]):
                line[i] = 'pos'
            line[i] = lemma(line[i])
            if line[i] == 'have':
                line[i] = ''
            if i == (j-1):
                line[i] = line[i]+'\n'
            else:
                line[i] = line[i]+' '
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
    train_que.append(line)
    quelist.append(line)
f.close()
f = open(path + '/data/Valid.que')
lines = f.readlines()
for line in lines:
    line = line.strip()
    valid_que.append(line)
    quelist.append(line)
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
    train_ans.append(line)
    anslist.append(line)
f.close()
f = open(path + '/data/Valid.ans')
lines = f.readlines()
for line in lines:
    line = line.strip()
    valid_ans.append(line)
    anslist.append(line)
f.close()

#分词
print('Turning Q&A into vectors...')
tokenizer_q = text.Tokenizer(num_words=dic_num)
tokenizer_q.fit_on_texts(quelist)
trainque_feature = tokenizer_q.texts_to_sequences(train_que)  #文本向量化
validque_feature = tokenizer_q.texts_to_sequences(valid_que)
testque_feature = tokenizer_q.texts_to_sequences(test_que)
trainque_feature = sequence.pad_sequences(trainque_feature, maxlen, padding='post', value=0, truncating='post')  #统一序列长度
validque_feature = sequence.pad_sequences(validque_feature, maxlen, padding='post', value=0, truncating='post')
testque_feature = sequence.pad_sequences(testque_feature, maxlen, padding='post', value=0, truncating='post')

tokenizer_a = text.Tokenizer(num_words=dic_num)
tokenizer_a.fit_on_texts(anslist)
trainans_feature = tokenizer_a.texts_to_sequences(train_ans)
validans_feature = tokenizer_a.texts_to_sequences(valid_ans)
trainans_feature = sequence.pad_sequences(trainans_feature, maxlen, padding='post', value=0, truncating='post')
validans_feature = sequence.pad_sequences(validans_feature, maxlen, padding='post', value=0, truncating='post')
trainans_feature = keras.utils.to_categorical(trainans_feature, dic_num+1)  #one-hot
validans_feature = keras.utils.to_categorical(validans_feature, dic_num+1)

#读取图像特征数据
trainimg_feature = np.load(path + '/data/Train_im.npy')
validimg_feature = np.load(path + '/data/Valid_im.npy')
testimg_feature = np.load(path + '/data/Test_im.npy')

#建立模型
print('Building model...')

#图像模型
encoded_image = Input(shape=(1000,))

#问答模型
question_input = Input(shape=(maxlen,))
embedded_question = Embedding(input_dim=dic_num, output_dim=512, input_length=maxlen)(question_input)  #嵌入层
encoded_question = LSTM(512)(embedded_question)  #LSTM层

#合并模型
merged = keras.layers.concatenate([encoded_question, encoded_image])  #融合层
repeated = RepeatVector(maxlen)(merged)  #复制层
sequenced = LSTM(return_sequences=True, units=512)(repeated)  #LSTM层
output = TimeDistributed(Dense(dic_num+1, activation='softmax'))(sequenced)  #封装层

#模型统计
vqa_model = Model(inputs=[encoded_image, question_input], outputs=output)
vqa_model.summary()

#编译模型
vqa_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# 保存模型
filepath = path + '/data/best.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

#训练模型
if os.path.isfile(path + '/data/best.hdf5') == False:
    print('Training model...')
    history = vqa_model.fit([trainimg_feature, trainque_feature], trainans_feature, epochs=500, batch_size=64, validation_data=([validimg_feature, validque_feature], validans_feature), callbacks=callbacks_list, verbose=1)
    
    #可视化训练过程
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
else:
    print('Loading model...')  #载入模型
    vqa_model.load_weights(path + '/data/best.hdf5', by_name=True)  
    json_string = vqa_model.to_json()  
    vqa_model = model_from_json(json_string)

#测试结果
print('Answer generating...')
ans = vqa_model.predict([trainimg_feature, trainque_feature])
dic = tokenizer_a.word_index  #字典-索引
ind ={value:key for key, value in dic.items()}  #索引-字典

#修正输出词
def dic_fix_s(word):
    for i in range(dic_num):
        if ind[i+1] == word:
            ind[i+1] = ind[i+1]+'s'
for div in ['po','thi','sinu','pelvi','pancrea','thrombosi','necrosi','corpu','stenosi','metastasi','ye','venou','thrombu','uteru','thalamu','pon','esophagu','cavernou','sac','meniscu','los','conu','rib','nucleu','fibrosi','ramu','rectu','agenesi','hi','bulbu','osseou','osteomyeliti','edematou','tuberculosi','plexu','clivu','pneumocephalu','atelectasi','vermi','globu','sclerosi','iliopsoa','psoa','supraspinatu','hydronephrosi']:
    dic_fix_s(div)
for i in range(dic_num):
    if ind[i+1] == 'vertebra':
        ind[i+1] = ind[i+1]+'e'
    if ind[i+1] == 'iv':
        ind[i+1] = ind[i+1]+'c'
    if ind[i+1] == 'axi':
        ind[i+1] = ind[i+1]+'al'
    if ind[i+1] == 'pleura':
        ind[i+1] = ind[i+1]+'l'
    if ind[i+1] == 'axilla':
        ind[i+1] = ind[i+1]+'ry'

#生成答句文件
def final_ans(data, num):
    fp = open(path + '/data/' + data + '.fn', 'w')
    for i in range(num):
        fp.write(str(i+1))
        fp.write('	')
        fe = open(path + '/data/' + data + '.name')
        for k in fe.readlines()[i]:
            k = k.strip()
            fp.write(k)
        fp.write('	')
        for j in range(maxlen):
            an = np.argmax(ans[i][j],axis=0)
            if j != maxlen-1:
                anext = np.argmax(ans[i][j+1],axis=0)
            if an != 0 and j != maxlen-1:
                if an != anext:
                    fp.write(ind[an])
                    if anext != 0:
                        fp.write(' ')
            elif an != 0 and j == maxlen-1:
                fp.write(ind[an])
                fp.write('\n')
            elif an == 0 and j == maxlen-1:
                fp.write('\n')
    fp.close()
    
if os.path.isfile(path + '/data/Test.fn') == False:
    final_ans('Train', num_train)
    final_ans('Valid', num_valid)
    final_ans('Test', num_test)
print('Finished.')