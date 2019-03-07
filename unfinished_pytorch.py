import os
import re
import csv
import math
import time
import codecs
import string
import random
import warnings
import collections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from PIL import Image
from tqdm import tqdm
from scipy import spatial

import nltk
from nltk.translate.bleu_score import SmoothingFunction
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import wordnet as wn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
from torch.nn.utils.weight_norm import weight_norm
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torchvision
from torchvision import transforms, models
from pytorch_pretrained_bert import BertTokenizer, BertModel

# parameters
BATCH = 32
IM_K = 19
DIM = 1024
DROP = 0.2
EPOCH = 100
MAXLEN = 30
NUM = "1"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ['PYTHONHASHSEED'] = '2019'
random.seed(2019)
np.random.seed(2019)
torch.manual_seed(2019)
torch.cuda.manual_seed(2019)
torch.backends.cudnn.deterministic = True

# ---Data preparation---
stop = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
transform1 = transforms.Compose([transforms.Resize((224,224)), transforms.RandomResizedCrop(224,scale=(0.9,1.0),ratio=(0.9,1.1)),
                                 transforms.RandomRotation(10), transforms.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1,hue=0.1),
                                 transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])
transform2 = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
                                 transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])])

def imag2vect(file,stan_type=1):
    f = Image.open(file)
    if stan_type == 1:
        imag = transform1(f)
    else:
        imag = transform2(f)
    f.close()
    if len(imag) == 1:
        imag = torch.cat([imag,imag,imag],dim=0)
    return imag

def text2token(text):
    text = tokenizer.tokenize(text)
    return tokenizer.convert_tokens_to_ids(text)

def ques_standard(text):
    # punctuation & lower
    temp = text.strip('?').split(' ')
    temp_list = []
    for i in range(len(temp)):
        if temp[i] != '':
            temp[i] = re.sub("[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+ ", "", temp[i].lower())
            temp_list.append(temp[i].replace('-',' '))
    return ' '.join(temp_list)

def answ_standard(text):
    # punctuation & lower & stop & stem
    temp = text.strip('?').split(' ')
    temp_list = []
    for i in range(len(temp)):
        if temp[i] != '':
            temp[i] = re.sub("[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+ ", "", temp[i].lower())
            if temp[i] in stop and temp[i] != 'no':
                temp[i] = ''
            temp[i] = stemmer.stem(temp[i])
            temp_list.append(temp[i].replace('-',' '))
    while '' in temp_list:
        temp_list.remove('')
    return ' '.join(temp_list)

def train_prepare(category='Train'):
    imag, ques, answ = [], [], []
    f = open('/home/yzhou/VQA/VQAMed2018'+category+'/VQAMed2018'+category+'-QA.csv', 'r', encoding='utf-8')
    lines = f.readlines()
    for line in lines:
        temp_name = line.split('\t')[1]
        temp_imag = '/home/yzhou/VQA/VQAMed2018'+category+'/VQAMed2018'+category+'-images/'+temp_name+'.jpg'
        temp_ques = ques_standard(line.split('\t')[2])
        toke_ques = text2token(temp_ques)
        temp_answ = answ_standard(line.split('\t')[3])
        toke_answ = text2token(temp_answ)
        if (temp_ques + temp_answ).find('ct') != -1:
            imag.append(temp_imag)
            ques.append(toke_ques)
            answ.append([1]+toke_answ+[2])
        elif (temp_ques + temp_answ).find('mri') != -1:
            for _ in range(2):
                imag.append(temp_imag)
                ques.append(toke_ques)
                answ.append([1]+toke_answ+[2])
        else:
            for _ in range(3):
                imag.append(temp_imag)
                ques.append(toke_ques)
                answ.append([1]+toke_answ+[2])
    f.close()
    return imag, ques, answ

def test_prepare(category='Test'):
    name, imag, ques = [], [], []
    f = open('/home/yzhou/VQA/VQAMed2018'+category+'/VQAMed2018'+category+'-QA-Eval.csv', 'r', encoding='utf-8')
    lines = f.readlines()
    for line in lines:
        temp_name = line.split('\t')[1]
        name.append(temp_name)
        temp_imag = '/home/yzhou/VQA/VQAMed2018'+category+'/VQAMed2018'+category+'-images/'+temp_name+'.jpg'
        imag.append(imag2vect(temp_imag, 2))
        temp_ques = ques_standard(line.split('\t')[2])
        toke_ques = text2token(temp_ques)
        ques.append(toke_ques)
    f.close()
    return name, imag, ques

train_imag, train_ques, train_answ = train_prepare()
valid_imag, valid_ques, valid_answ = train_prepare('Valid')
test_name, test_imag, test_ques = test_prepare()

# answer_dictionary
token_answers = train_answ + valid_answ
freq = collections.Counter([val for sublist in (token_answers) for val in sublist])
for i in range(len(token_answers)):
    for j, num in enumerate(token_answers[i]):
        if freq[num] <= 7:
            token_answers[i][j] = 100
word2index = {'[PAD]': 0, '[STA]': 1, '[END]': 2, '[UNK]': 3}
count = 4
for i in range(len(token_answers)):
    for num in token_answers[i]:
        if num != 0 and num != 1 and num != 2 and num != 100:
            if tokenizer.convert_ids_to_tokens([num])[0] not in word2index:
                word2index[tokenizer.convert_ids_to_tokens([num])[0]] = count
                count += 1
index2word ={value:key for key, value in word2index.items()}

# get_sequences
images, questions, answers, labels= [], [], [], []
for i in tqdm(range(len(token_answers))):
    for j in range(len(token_answers[i])-1):
        images.append((train_imag + valid_imag)[i])
        questions.append((train_ques + valid_ques)[i])
        answers.append(token_answers[i][:j+1])
        if token_answers[i][j+1] == 2:
            labels.append(2)
        else:
            labels.append(word2index[tokenizer.convert_ids_to_tokens([token_answers[i][j+1]])[0]])

data_imag, data_ques, data_answ, data_labe = [], [], [], []
frequency = collections.Counter(labels)
for i in range(len(labels)):
    if frequency[labels[i]] > 400:
        if random.random() < 400.0 / float(frequency[labels[i]]):
            data_imag.append(images[i])
            data_ques.append(questions[i])
            data_answ.append(answers[i])
            data_labe.append(labels[i])
    else:
        data_imag.append(images[i])
        data_ques.append(questions[i])
        data_answ.append(answers[i])
        data_labe.append(labels[i])

# data_loading
print('dict_length：',len(index2word))
print('trainingset_length: ',len(data_labe))
class Load_data(torch.utils.data.Dataset):
    def __init__(self, images, questions, answers, labels):
        self.imag = images
        self.ques = questions
        self.answ = answers
        self.labe = labels
    def __getitem__(self, index):
        return imag2vect(self.imag[index], 1), self.ques[index], self.answ[index], self.labe[index]
    def __len__(self):
        return len(self.labe)

def collate_fn1(data):
    return zip(*data)

state = np.random.get_state()
np.random.shuffle(data_imag)
np.random.set_state(state)
np.random.shuffle(data_ques)
np.random.set_state(state)
np.random.shuffle(data_answ)
np.random.set_state(state)
np.random.shuffle(data_labe)

train_loader1 = torch.utils.data.DataLoader(Load_data(data_imag[:16384], data_ques[:16384], data_answ[:16384], data_labe[:16384]),
                                            batch_size=BATCH,shuffle=True,collate_fn=collate_fn1)
train_loader2 = torch.utils.data.DataLoader(Load_data(data_imag[16384:32768], data_ques[16384:32768], data_answ[16384:32768], data_labe[16384:32768]),
                                           batch_size=BATCH,shuffle=True,collate_fn=collate_fn1)
train_loader3 = torch.utils.data.DataLoader(Load_data(data_imag[32768:49152], data_ques[32768:49152], data_answ[32768:49152], data_labe[32768:49152]),
                                           batch_size=BATCH,shuffle=True,collate_fn=collate_fn1)
valid_loader1 = torch.utils.data.DataLoader(Load_data(data_imag[:2048],data_ques[:2048],data_answ[:2048], data_labe[:2048])
                                            ,batch_size=BATCH,shuffle=True,collate_fn=collate_fn1)
valid_loader2 = torch.utils.data.DataLoader(Load_data(data_imag[16384:18432], data_ques[16384:18432],
                                                      data_answ[16384:18432], data_labe[16384:18432])
                                            ,batch_size=BATCH,shuffle=True,collate_fn=collate_fn1)
valid_loader3 = torch.utils.data.DataLoader(Load_data(data_imag[32768:34816],data_ques[32768:34816],
                                                      data_answ[32768:34816], data_labe[32768:34816])
                                            ,batch_size=BATCH,shuffle=True,collate_fn=collate_fn1)

# ---Model preparation---
# (-1, 3. 224, 224) → (-1, IM_K, 784)
class Transfer(nn.Module):
    def __init__(self, model1=models.resnet152(pretrained=True), model2=models.densenet161(pretrained=True)):
        super(Transfer, self).__init__()
        self.model1 = model1.to(device)
        self.model2 = model2.to(device)
        for p in self.parameters():
            p.requires_grad=False
        self.upsamp = nn.PixelShuffle(4).to(device)
        self.conv = nn.Conv2d(266, IM_K, kernel_size=(1, 1), stride=(1, 1), bias=False).to(device)
        self.batch = nn.BatchNorm2d(IM_K, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True).to(device)
        self.relu = nn.ReLU().to(device)
    def forward(self, v):
        for i in range(len(v)):
            if i == 0:
                temp = torch.unsqueeze(v[i], 0).view(-1,3,224,224)
            else:
                temp = torch.cat((temp,torch.unsqueeze(v[i], 0).view(-1,3,224,224)),0)
        v = temp.to(device)
        modules1 = list(self.model1.children())[:-2]
        fix1 = nn.Sequential(*modules1).to(device)
        modules2 = list(self.model2.children())[:-1]
        fix2 = nn.Sequential(*modules2).to(device)
        v = torch.cat((fix1(v), fix2(v)), 1)
        v = self.upsamp(v)
        v = self.conv(v)
        v = self.batch(v)
        v = self.relu(v)
        return v.view(len(v),IM_K,-1)

# (-1, seq_len) → (-1, seq_len, 1024)
class Bert(nn.Module):
    def __init__(self, model=BertModel.from_pretrained('bert-large-uncased')):
        super(Bert, self).__init__()
        self.model = model.to(device)
        for p in self.parameters():
            p.requires_grad=False
    def forward(self, q, a):
        q_len = max(len(temp) for temp in q)
        a_len = max(len(temp) for temp in a)
        qa = np.concatenate((np.array([temp + [0] * (q_len - len(temp)) for temp in q]),
                             np.array([temp + [0] * (a_len - len(temp)) for temp in a])),axis=1)
        segm = np.repeat(np.expand_dims(np.concatenate((np.zeros(q_len, dtype=int),np.ones(a_len, dtype=int))), axis=0), len(q), axis=0)
        mask = np.concatenate((np.array([[1 if word != 0 else 0 for _, word in enumerate(temp)] for temp in 
                                         np.array([temp + [0] * (q_len - len(temp)) for temp in q])]),
                               np.array([[1 if word != 0 else 0 for _, word in enumerate(temp)] for temp in 
                                         np.array([temp + [0] * (a_len - len(temp)) for temp in a])])),axis=1)
        self.model.eval()
        qa = torch.tensor(qa).to(device)
        segm = torch.tensor(segm).to(device)
        mask = torch.tensor(mask).to(device)
        qa_mask = 1 - torch.mul(mask[:, q_len:].unsqueeze(-1), mask[:, :q_len].unsqueeze(-2))
        out, _ = self.model(input_ids=qa, token_type_ids=segm, attention_mask=mask)
        return out[-2][:, :q_len, :], out[-2][:, q_len:, :], qa_mask

# (-1, IM_K, 784) + (-1, seq_len, 1024) → （-1, 1024）
class Top_down(nn.Module):
    def __init__(self):
        super(Top_down, self).__init__()
        self.norm1 = nn.Linear(784, DIM).to(device)
        self.norm2 = nn.Linear(DIM, DIM).to(device)
        self.norm3 = weight_norm(nn.Linear(DIM//2, 1), dim=None).to(device)
        self.norm4 = nn.Linear(DIM//2, DIM*2).to(device)
        self.drop1 = nn.Dropout(DROP).to(device)
        self.drop2 = nn.Dropout(DROP).to(device)
    def forward(self, v, t):
        v = F.glu(self.norm1(v))
        t = F.glu(self.norm2(t))
        t = torch.sum(t, dim=1)
        att = v * t.unsqueeze(1).repeat(1, IM_K, 1)
        att = F.softmax(self.norm3(self.drop1(att)), 1)
        v = (att * v).sum(1)
        tv = self.drop2(t * v)
        tv = F.glu(self.norm4(tv))
        return tv, att

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.query_scale = (DIM//num_heads)**-0.5
        self.query_linear = nn.Linear(DIM, DIM//2).to(device)
        self.key_linear = nn.Linear(DIM, DIM//2).to(device)
        self.value_linear = nn.Linear(DIM, DIM//2).to(device)
        self.out_linear = nn.Linear(DIM//2, DIM).to(device)
        self.dropout = nn.Dropout(DROP).to(device)
    def _split_heads(self, x):
        shape = x.shape
        return x.view(shape[0], shape[1], self.num_heads, shape[2]//self.num_heads).permute(0, 2, 1, 3)
    def _merge_heads(self, x):
        shape = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(shape[0], shape[2], shape[3]*self.num_heads)
    def forward(self, q, k, v, kv_mask=None):
        q = self._split_heads(self.query_linear(q))
        k = self._split_heads(self.key_linear(k))
        v = self._split_heads(self.value_linear(v))
        att = torch.matmul(q*self.query_scale, k.permute(0, 1, 3, 2))
        if kv_mask is not None:
            att = att.masked_fill_(kv_mask.type(torch.cuda.ByteTensor).unsqueeze(1).repeat(1, self.num_heads, 1, 1), -1e9)
        att = self.dropout(F.softmax(att, dim=-1))
        out = torch.matmul(att, v)
        out = self._merge_heads(out)
        out = self.out_linear(out)
        return out, att
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(DIM, DIM//2).to(device)
        self.w_2 = nn.Linear(DIM//4, DIM).to(device)
        self.dropout = nn.Dropout(DROP).to(device)
    def forward(self, x):
        return self.w_2(self.dropout(F.glu(self.w_1(x))))

# (-1, ans_len, 1024) + (-1, que_len, 1024) + (-1, ans_len, que_len) → (-1, 1024)
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.multi_head_attention = MultiHeadAttention()
        self.positionwise_feed_forward = PositionwiseFeedForward()
        self.drop1 = nn.Dropout(DROP).to(device)
        self.drop2 = nn.Dropout(DROP).to(device)
        self.layernorm1 = nn.LayerNorm(DIM, eps=1e-5).to(device)
        self.layernorm2 = nn.LayerNorm(DIM, eps=1e-5).to(device)
        self.layernorm3 = nn.LayerNorm(DIM, eps=1e-5).to(device)
    def forward(self, origin, ques, mask):
        answ = self.layernorm1(origin)
        ques = self.layernorm2(ques)
        out, att = self.multi_head_attention(answ, ques, ques, mask)
        origin = origin + self.drop1(out)
        out = self.layernorm3(origin)
        out = self.positionwise_feed_forward(out)
        out = (origin + self.drop2(out)).sum(1)
        return out, att

# (-1, 1024) + (-1, 1024) + (-1, 1024) → (-1, dic_len)
class Final_model(nn.Module):
    def __init__(self, dic_len=len(index2word)):
        super(Final_model, self).__init__()
        self.transfer = Transfer()
        self.bert = Bert()
        self.topdown1 = Top_down()
        self.topdown2 = Top_down()
        self.tranformer = Transformer()
        self.norm1 = nn.Linear(3*DIM, DIM//2).to(device)
        self.norm2 = nn.Linear(DIM//4, dic_len).to(device)
        self.drop = nn.Dropout(DROP).to(device)
        self.out = nn.LogSoftmax(dim=-1).to(device)
    def forward(self, imag, ques, answ):
        imag = self.transfer(imag)
        ques, answ, mask = self.bert(ques, answ)
        feat1, vq_att = self.topdown1(imag, ques)
        feat2, va_att = self.topdown2(imag, answ)
        feat3, qa_att = self.tranformer(answ, ques, mask)
        out = torch.cat((feat1, feat2, feat3), -1)
        out = F.glu(self.norm1(out))
        out = self.norm2(self.drop(out))
        out = self.out(out)
        return out, vq_att, va_att, qa_att

# ---Training---
class labelsmoothing(nn.Module):
    def __init__(self, smoothing=0.1, dic_len=len(index2word), bias=1):
        super(labelsmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum').to(device)
        self.dic_len = dic_len
        self.smoothing = smoothing
        self.bias = bias
    def forward(self, pred, label):
        gt = torch.zeros(len(label), self.dic_len).to(device)
        for i in range(len(label)):
            if label[i] == 2 or label[i] == 3:
                for j in range(self.dic_len):
                    if j == 0 or j == 1:
                        pass
                    elif j == label[i]:
                        gt[i][j] = 1.0 - self.bias * self.smoothing
                    else:
                        gt[i][j] = self.bias * self.smoothing / (self.dic_len - 3)
            else:
                for j in range(self.dic_len):
                    if j == 0 or j == 1:
                        pass
                    elif j == label[i]:
                        gt[i][j] = 1.0 - self.smoothing
                    else:
                        gt[i][j] = self.smoothing / (self.dic_len - 3)
        correct = 0
        for i in range(len(label)):
            if int(torch.argmax(pred, dim=1)[i]) == label[i]:
                correct += 1
        return self.criterion(pred, gt), correct

def train_epoch(model, training_data, optimizer):
    model.train()
    total_loss, n_word_total, n_word_correct = 0, 0, 0
    questions, answers, attentions = [], [], []
    for batch in tqdm(training_data, desc='  - (Training)   ', leave=False):
        imag, ques, answ, label = map(lambda x: x, batch)
        optimizer.zero_grad()
        pred, vq_att, va_att, qa_att = model(imag, ques, answ)
        for i in range(len(answ)):
            if int(answ[i][-1]) == 2:
                questions.append(ques[i])
                answers.append(answ[i])
                attentions.append(qa_att[i])
        loss, n_correct = labelsmoothing()(pred, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_word_total += len(answ)
        n_word_correct += n_correct
    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy, questions, answers, attentions

def eval_epoch(model, validation_data):
    model.eval()
    total_loss, n_word_total, n_word_correct = 0, 0, 0
    questions, answers, attentions = [], [], []
    with torch.no_grad():
        for batch in tqdm(validation_data, desc='  - (Validation) ', leave=False):
            imag, ques, answ, label = map(lambda x: x, batch)
            pred, vq_att, va_att, qa_att = model(imag, ques, answ)
            for i in range(len(answ)):
                if int(label[i]) == 2:
                    questions.append(ques[i])
                    answers.append(answ[i])
                    attentions.append(qa_att[i])
            loss, n_correct = labelsmoothing()(pred, label)
            total_loss += loss.item()
            n_word_total += len(answ)
            n_word_correct += n_correct
    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy, questions, answers, attentions

def train(model, training_data, validation_data, optimizer, num='1'):
    log_train_file = '/home/yzhou/VQA/data/weight/train' + NUM + num + '.log'
    log_valid_file = '/home/yzhou/VQA/data/weight/valid' + NUM + num + '.log'
    with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
        log_tf.write('epoch, loss, ppl, accuracy\n')
        log_vf.write('epoch, loss, ppl, accuracy\n')

    valid_losses = []
    patient = 0
    scheduler = ReduceLROnPlateau(optimizer, factor=0.2, patience=4, verbose=True)
    for epoch_i in range(EPOCH):
        print('[ Epoch', epoch_i, ']')
        start = time.time()
        train_loss, train_accu, train_ques, train_answ, train_att = train_epoch(model, training_data, optimizer)
        print('  - (Training)   loss: {loss: 3.3f}, ppl: {ppl: 3.3f}, accuracy: {accu:3.3f} %, elapse: {elapse:3.3f} min'
              .format(loss=train_loss, ppl=math.exp(min(train_loss,100)), accu=100*train_accu, elapse=(time.time()-start)/60))

        start = time.time()
        valid_loss, valid_accu, valid_ques, valid_answ, valid_att = eval_epoch(model, validation_data)
        print('  - (Validation) loss: {loss: 3.3f}, ppl: {ppl: 3.3f}, accuracy: {accu:3.3f} %, elapse: {elapse:3.3f} min'
              .format(loss=valid_loss, ppl=math.exp(min(valid_loss,100)), accu=100*valid_accu, elapse=(time.time()-start)/60))
        scheduler.step(valid_loss)
        
        valid_losses += [valid_loss]
        if valid_loss <= min(valid_losses):
            torch.save({'model': model.state_dict()}, '/home/yzhou/VQA/data/weight/model' + NUM + num + '.chkpt')
            print('    - [Info] The checkpoint file has been updated.')
            patient = 0
        else:
            patient += 1
            if patient > 6:
                break
                
        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu: 3.5f}\n'.format(
                    epoch=epoch_i, loss=train_loss, ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu: 3.5f}\n'.format(
                    epoch=epoch_i, loss=valid_loss, ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu))
                
# training
model = Final_model()
for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
model_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
train(model, train_loader1, valid_loader1, model_opt, '1')

for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
model_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
train(model, train_loader2, valid_loader2, model_opt, '2')

for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
model_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
train(model, train_loader3, valid_loader3, model_opt, '3')

# ---Visualization---
def show_attention(question,answer,attention):
    y_lab = []
    for i in range(len(answer)):
        if answer[i] != 2:
            y_lab.append(index2word[answer[i]])
        else:
            y_lab.append(index2word[answer[i]])
            break
    fig = plt.figure()
    fig.set_dpi(300)
    ax = fig.add_subplot(111)
    cax = ax.matshow(attention[:len(y_lab)], cmap='bone')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + tokenizer.convert_ids_to_tokens(question), rotation=90)
    ax.set_yticklabels([''] + y_lab)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()

def show_loss(folder='/home/yzhou/VQA/data/weight/', name='11'):
    loss_train, loss_valid, accu_train, accu_valid = [], [], [], []
    f = open(folder + 'train' + name + '.log')
    next(f)
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        loss_train.append(float(line.split(', ')[1]))
        accu_train.append(float(line.split(', ')[3]))
    f.close()
    f = open(folder + 'valid' + name + '.log')
    next(f)
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        loss_valid.append(float(line.split(', ')[1]))
        accu_valid.append(float(line.split(', ')[3]))
    f.close()
    fig = plt.figure()
    fig.set_dpi(300)
    plt.plot(loss_train)
    plt.plot(loss_valid)
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper right')
    plt.show()
    plt.savefig(folder + 'loss' + name + '.jpg')
    fig = plt.figure()
    fig.set_dpi(300)
    plt.plot(accu_train)
    plt.plot(accu_valid)
    plt.title('Accuracy')
    plt.ylabel('accu')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='lower right')
    plt.show()
    plt.savefig(folder + 'accuracy' + name + '.jpg')

# show_loss(name='11')

# ---Evaluation---
# loading weights
model_a, model_b, model_c = Final_model(), Final_model(), Final_model()
model_a.load_state_dict(torch.load('/home/yzhou/VQA/data/weight/model' + NUM + '1.chkpt')['model'])
model_b.load_state_dict(torch.load('/home/yzhou/VQA/data/weight/model' + NUM + '2.chkpt')['model'])
model_c.load_state_dict(torch.load('/home/yzhou/VQA/data/weight/model' + NUM + '3.chkpt')['model'])

def beam_search(images, questions, k=2, bias=-999999., att=False):
    images = images.unsqueeze(0).repeat(k,1,1,1)
    questions = tuple([questions for _ in range(k)])
    attention = []
    for i in range(MAXLEN):
        model_a.eval()
        model_b.eval()
        model_c.eval()
        if i == 0:
            answers = tuple([[1] for _ in range(k)])
            scores = [0] * k
            last_choices = [1] * k
            with torch.no_grad():
                pred_a, _1, _2, at_1 = model_a(images, questions, answers)
                pred_b, _4, _5, at_2 = model_b(images, questions, answers)
                pred_c, _7, _8, at_3 = model_c(images, questions, answers)
            pred = pred_a + pred_b + pred_c
            attention.append((at_1.mean(1)[0][-1].detach().cpu().numpy()+at_2.mean(1)[0][-1].detach().cpu().numpy()+
                              at_3.mean(1)[0][-1].detach().cpu().numpy())/3)
            choices = np.argsort(pred.cpu())[:,-k:]
            for j in range(k):
                temp_choice = int(choices[0][-1-j])
                answers[j].append(temp_choice)
                if temp_choice in last_choices or temp_choice in [2, 3]:
                    scores[j] += bias
                scores[j] += pred[0][temp_choice].cpu()
                last_choices[j] = temp_choice
        else:
            with torch.no_grad():
                pred_a, _1, _2, at_1 = model_a(images, questions, answers)
                pred_b, _4, _5, at_2 = model_b(images, questions, answers)
                pred_c, _7, _8, at_3 = model_c(images, questions, answers)
            pred = pred_a + pred_b + pred_c
            attention.append((at_1.mean(1)[0][-1].detach().cpu().numpy()+at_2.mean(1)[0][-1].detach().cpu().numpy()+
                              at_3.mean(1)[0][-1].detach().cpu().numpy())/3)
            choices = np.argsort(pred.cpu())[:,-k:]
            answers = list(answers)
            temp_answers, temp_scores = [], []
            for j in range(k):
                for m in range(k):
                    temp_choice = int(choices[j][-1-m])
                    temp_answers.append(answers[j] + [temp_choice])
                    if temp_choice in last_choices or temp_choice in [2, 3]:
                        scores[j] += bias
                        temp_scores.append(scores[j] + pred[j][temp_choice].cpu() + torch.tensor(bias))
                    else:
                        temp_scores.append(scores[j] + pred[j][temp_choice].cpu())
            choosing = np.argsort(temp_scores)
            for j in range(k):
                answers[j] = temp_answers[choosing[-1-j]]
                scores[j] = temp_scores[choosing[-1-j]]
                last_choices[j] = temp_answers[choosing[-1-j]][-1]
            answers = tuple(answers)
    if att == True:
        show_attention(questions[0], answers[0], np.asarray(attention))
    for i in range(len(answers[0])):
        if answers[0][i] in [3]:
            answers[0][i] = ''
        elif answers[0][i] in [0,2]:
            break
        else:
            answers[0][i] = index2word[answers[0][i]]
    count = 0
    while '' in answers[0]:
        answers[0].remove('')
        count += 1
    if questions[0][0] in [2003,2024,2079,2515] and answers[0][1] not in ['yes','no']:
        return 'no'
    else:
        return ' '.join(answers[0][1:i-count]).replace(' ##', '')

f = open('/home/yzhou/VQA/data/result/' + NUM + '1.csv', 'w')
for i in tqdm(range(len(test_name))):
    f.write(str(i+1) + '	' + test_name[i] + '	')
    f.write(beam_search(test_imag[i], test_ques[i], bias=-999999.) + '\n')
f.close()

f = open('/home/yzhou/VQA/data/result/' + NUM + '2.csv', 'w')
for i in tqdm(range(len(test_name))):
    f.write(str(i+1) + '	' + test_name[i] + '	')
    f.write(beam_search(test_imag[i], test_ques[i], bias=-99999.) + '\n')
f.close()

f = open('/home/yzhou/VQA/data/result/' + NUM + '3.csv', 'w')
for i in tqdm(range(len(test_name))):
    f.write(str(i+1) + '	' + test_name[i] + '	')
    f.write(beam_search(test_imag[i], test_ques[i], bias=-9999.) + '\n')
f.close()

class VqaMedEvaluator:
    remove_stopwords = True
    stemming = True
    case_sensitive = False
    def __init__(self, answer_file_path):
        self.answer_file_path = answer_file_path
        self.gt = self.load_gt()
        self.word_pair_dict = {}
    def _evaluate(self, submission_file_path):
        predictions = self.load_predictions(submission_file_path)
        wbss = self.compute_wbss(predictions)
        bleu = self.compute_bleu(predictions)
        _result_object = {"wbss": wbss, "bleu" : bleu}
        return _result_object
    def load_gt(self):
        results = []
        for line in codecs.open(self.answer_file_path,'r','utf-8'):
            QID = line.split('\t')[0]
            ImageID = line.split('\t')[1]
            ans = line.split('\t')[2].strip()
            results.append((QID, ImageID, ans))
        return results
    def load_predictions(self, submission_file_path):
        qa_ids_testset = [tup[0] for tup in self.gt]
        image_ids_testset = [tup[1] for tup in self.gt]
        predictions = []
        occured_qaid_imageid_pairs = []
        with open(submission_file_path) as csvfile:
            reader = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
            lineCnt = 0
            occured_images = []
            for row in reader:
                lineCnt += 1
                if(len(row) != 3 and len(row) != 2):
                    raise Exception("Wrong format: Each line must consist of an QA-ID followed by a tab, an Image ID, a tab and an answer ({}), where the answer can be empty {}"
                        .format("<QA-ID><TAB><Image-ID><TAB><Answer>", self.line_nbr_string(lineCnt)))
                qa_id = row[0]
                image_id = row[1]
                try:
                    i = qa_ids_testset.index(qa_id)
                    expected_image_id = image_ids_testset[i]
                    if image_id != expected_image_id:
                        raise Exception
                except :
                    raise Exception("QA-ID '{}' with Image-ID '{}' does not represent a valid QA-ID - IMAGE ID pair in the testset {}"
                        .format(qa_id, image_id, self.line_nbr_string(lineCnt)))
                if (qa_id, image_id) in occured_qaid_imageid_pairs:
                    raise Exception("The QA-ID '{}' with Image-ID '{}' pair appeared more than once in the submission file {}"
                        .format(qa_id, image_id, self.line_nbr_string(lineCnt)))
                answer = row[2] if (len(row) == 3) else ""
                predictions.append((qa_id, image_id, answer))
                occured_qaid_imageid_pairs.append((qa_id, image_id))
            if len(predictions) != len(self.gt):
                raise Exception("Number of QA-ID - Image-ID pairs in submission file does not correspond with number of QA-ID - Image-ID pairs in testset")
        return predictions
    def compute_wbss(self, predictions):
        nltk.download('wordnet')
        count = 0
        totalscore_wbss = 0.0
        for tuple1, tuple2 in zip(self.gt, predictions):
            QID1 = tuple1[0]
            QID2 = tuple2[0]
            imageID1 = tuple1[1]
            imageID2 = tuple2[1]
            ans1 = tuple1[2]
            ans2 = tuple2[2]
            assert (QID1 == QID2)
            assert (imageID1 == imageID2)
            count+=1
            QID = QID1
            if ans1==ans2:
                score_wbss = 1.0
            elif ans2.strip() == "":
                score_wbss = 0
            else:
                score_wbss = self.calculateWBSS(ans1,ans2)
            totalscore_wbss+=score_wbss
        return totalscore_wbss/float(count)
    def calculateWBSS(self,S1, S2):
        if S1 is None or S2 is None:
            return 0.0
        dictionary = self.constructDict(S1.split(), S2.split())
        vector1 = self.getVector_wordnet(S1, dictionary)
        vector2 = self.getVector_wordnet(S2, dictionary)
        cos_similarity = self.calculateCosineSimilarity(vector1, vector2)
        return cos_similarity
    def getVector_wordnet(self,S, dictionary):
        vector = [0.0]*len(dictionary)
        for index, word in enumerate(dictionary):
            for wordinS in S.split():
                if wordinS == word:
                    score = 1.0
                else:
                    score = self.wups_score(word,wordinS)
                if score > vector[index]:
                    vector[index]=score
        return vector
    def constructDict(self, list1, list2):
        return list(set(list1+list2))
    def wups_score(self,word1, word2):
        score = 0.0
        score = self.wup_measure(word1, word2)
        return score
    def wup_measure(self,a, b, similarity_threshold = 0.925, debug = False):
        if debug: print('Original', a, b)
        if a+','+b in self.word_pair_dict.keys():
            return  self.word_pair_dict[a+','+b]
        def get_semantic_field(a):
            return wn.synsets(a, pos=wn.NOUN)
        if a == b: return 1.0
        interp_a = get_semantic_field(a)
        interp_b = get_semantic_field(b)
        if debug: print(interp_a)
        if interp_a == [] or interp_b == []:
            return 0.0
        if debug: print('Stem', a, b)
        global_max=0.0
        for x in interp_a:
            for y in interp_b:
                local_score=x.wup_similarity(y)
                if debug: print('Local', local_score)
                if local_score > global_max:
                    global_max=local_score
        if debug: print('Global', global_max)
        if global_max < similarity_threshold:
            interp_weight = 0.1
        else:
            interp_weight = 1.0
        final_score = global_max * interp_weight
        self.word_pair_dict[a+','+b] = final_score
        return final_score
    def calculateCosineSimilarity(self, vector1, vector2):
        return 1-spatial.distance.cosine(vector1, vector2)
    def compute_bleu(self, predictions):
        warnings.filterwarnings('ignore')
        nltk.download('punkt')
        nltk.download('stopwords')
        stops = set(stopwords.words("english"))
        stemmer = SnowballStemmer("english")
        translator = str.maketrans('', '', string.punctuation)
        candidate_pairs = self.readresult(predictions)
        gt_pairs = self.readresult(self.gt)
        max_score = len(gt_pairs)
        current_score = 0
        i = 0
        for image_key in candidate_pairs:
            candidate_caption = candidate_pairs[image_key]
            gt_caption = gt_pairs[image_key]
            if not VqaMedEvaluator.case_sensitive:
                candidate_caption = candidate_caption.lower()
                gt_caption = gt_caption.lower()
            candidate_words = nltk.tokenize.word_tokenize(candidate_caption.translate(translator))
            gt_words = nltk.tokenize.word_tokenize(gt_caption.translate(translator))
            if VqaMedEvaluator.remove_stopwords:
                candidate_words = [word for word in candidate_words if word.lower() not in stops]
                gt_words = [word for word in gt_words if word.lower() not in stops]
            if VqaMedEvaluator.stemming:
                candidate_words = [stemmer.stem(word) for word in candidate_words]
                gt_words = [stemmer.stem(word) for word in gt_words]
            try:
                if len(gt_words) == 0 and len(candidate_words) == 0:
                    bleu_score = 1
                else:
                    bleu_score = nltk.translate.bleu_score.sentence_bleu([gt_words], candidate_words, smoothing_function=SmoothingFunction().method0)
            except ZeroDivisionError:
                pass
            current_score += bleu_score
        return current_score / max_score
    def readresult(self,tuples):
        pairs = {}
        for row in tuples:
             pairs[row[0]]=row[2]
        return pairs
    def line_nbr_string(self, line_nbr):
        return "(Line nbr {})".format(line_nbr)
    
gt_file_path = "/home/yzhou/VQA/data/result/gt_file.csv"
evaluator = VqaMedEvaluator(gt_file_path)
submission_file_path = '/home/yzhou/VQA/data/result/' + NUM + '1.csv'
result = evaluator._evaluate(submission_file_path)
print('1: ', result)
submission_file_path = '/home/yzhou/VQA/data/result/' + NUM + '2.csv'
result = evaluator._evaluate(submission_file_path)
print('2: ', result)
submission_file_path = '/home/yzhou/VQA/data/result/' + NUM + '3.csv'
result = evaluator._evaluate(submission_file_path)
print('3: ', result)
