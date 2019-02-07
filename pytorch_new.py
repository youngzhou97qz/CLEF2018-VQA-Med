import math
import time
import collections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from PIL import Image
from tqdm import tqdm

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
BATCH = 8
IM_K = 19
DIM = 768
DROP = 0.2
EPOCH = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
torch.cuda.manual_seed(0)

# preprocessing
# image 3*224*224
def readname(name):
    list_name = []
    f = open('/home/yzhou/VQA/data/process/' + name + '.txt','r',encoding='utf-8')
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        list_name.append(line)
    f.close()
    return list_name

train_name = readname('train_name')
valid_name = readname('valid_name')
test_name = readname('test_name')

transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def img2vec(name,list_name):
    temp_imag = []
    for i in tqdm(range(len(list_name))):
        if name == 'train':
            data = Image.open('/home/yzhou/VQA/VQAMed2018Train/VQAMed2018Train-images/' + list_name[i] + '.jpg')
        elif name == 'valid':
            data = Image.open('/home/yzhou/VQA/VQAMed2018Valid/VQAMed2018Valid-images/' + list_name[i] + '.jpg')
        elif name == 'test':
            data = Image.open('/home/yzhou/VQA/VQAMed2018Test/VQAMed2018Test-images/' + list_name[i] + '.jpg')
        temp = transform(data)
        if len(temp) == 1:
            temp = temp.repeat(3,1,1)
        temp_imag.append(temp)
    return temp_imag

train_imag = img2vec('train', train_name)
valid_imag = img2vec('valid', valid_name)
test_imag = img2vec('test', test_name)

# Q&A
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def readques(ques):
    index = []
    f = open('/home/yzhou/VQA/data/process/' + ques + '.txt','r',encoding='utf-8')
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        tokenized_text = tokenizer.tokenize(line)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        index.append(indexed_tokens)
    f.close()
    return index

train_ques = readques('train_ques')
valid_ques = readques('valid_ques')
test_ques = readques('test_ques')

def readansw(answ):
    index, temp_1, temp_2 = [], [1], [2]
    f = open('/home/yzhou/VQA/data/process/' + answ + '.txt','r',encoding='utf-8')
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        tokenized_text = tokenizer.tokenize(line)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        index.append(temp_1 + indexed_tokens + temp_2)
    f.close()
    return index

train_answ = readansw('train_answ')
valid_answ = readansw('valid_answ')
test_answ = readansw('test_answ')

# answer_dictionary
answer = train_answ + valid_answ
freq = collections.Counter([val for sublist in (answer) for val in sublist])

def answ_label(answ,low):
    for i in tqdm(range(len(answ))):
        for j, num in enumerate(answ[i]):
            if freq[num] <= low:
                answ[i][j] = 100
    return answ

train_answ = answ_label(train_answ,5)
valid_answ = answ_label(valid_answ,5)
test_answ = answ_label(test_answ,5)

answer = train_answ + valid_answ
word2index = {'[PAD]': 0, '[STA]': 1, '[END]': 2, '[UNK]': 3}
count = 4
for i in range(len(answer)):
    for num in answer[i]:
        if num != 0 and num != 1 and num != 2 and num != 100:
            if tokenizer.convert_ids_to_tokens([num])[0] not in word2index:
                word2index[tokenizer.convert_ids_to_tokens([num])[0]] = count
                count += 1
index2word ={value:key for key, value in word2index.items()}

# pack_data
def prepare(imag,ques,answ):
    images, questions, answers, labels= [], [], [], []
    for i in tqdm(range(len(ques))):
        if len(imag[i]) == 1:
            imag[i] = torch.cat([imag[i],imag[i],imag[i]],dim=1)
        for j in range(len(answ[i])-1):
            images.append(imag[i])
            questions.append(ques[i])
            answers.append(answ[i][:j+1])
            if answ[i][j+1] == 2:
                labels.append(2)
            else:
                labels.append(word2index[tokenizer.convert_ids_to_tokens([answ[i][j+1]])[0]])
    return images, questions, answers, labels

train_imag, train_ques, train_answ, train_label = prepare(train_imag, train_ques, train_answ)
valid_imag, valid_ques, valid_answ, valid_label = prepare(valid_imag, valid_ques, valid_answ)

print('dict_length：',len(index2word))
print('train_quantity：',len(train_label))
print('valid_quantity：',len(valid_label))

# data_loading
class Train_data(torch.utils.data.Dataset):
    def __init__(self,train_imag,train_ques,train_answ,train_label):
        self.imag = train_imag
        self.ques = train_ques
        self.answ = train_answ
        self.labe = train_label
    def __getitem__(self, index):
        return self.imag[index], self.ques[index], self.answ[index], self.labe[index]
    def __len__(self):
        return len(self.labe)

class Valid_data(torch.utils.data.Dataset):
    def __init__(self,valid_imag,valid_ques,valid_answ,valid_label):
        self.imag = valid_imag
        self.ques = valid_ques
        self.answ = valid_answ
        self.labe = valid_label
    def __getitem__(self, index):
        return self.imag[index], self.ques[index], self.answ[index], self.labe[index]
    def __len__(self):
        return len(self.labe)

def collate_fn1(data):
    return zip(*data)

train_loader = torch.utils.data.DataLoader(Train_data(train_imag[:10],train_ques[:10],train_answ[:10],train_label[:10]),batch_size=BATCH,
                                           shuffle=True,num_workers=2,collate_fn=collate_fn1)
valid_loader = torch.utils.data.DataLoader(Valid_data(valid_imag[:10],valid_ques[:10],valid_answ[:10],valid_label[:10]),batch_size=BATCH,
                                           shuffle=True,num_workers=2,collate_fn=collate_fn1)

# model_loading
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
        self.batch = nn.BatchNorm2d(IM_K, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).to(device)
        self.relu = nn.ReLU().to(device)
    def forward(self, v):
        for i in range(len(v)):
            if i == 0:
                temp = torch.unsqueeze(v[i], 0).view(-1,3,224,224)
            else:
                temp = torch.cat((temp,torch.unsqueeze(v[i], 0).view(-1,3,224,224)),0)
        v = temp.to(device)
        if len(v) == 1:
            v = v[0].clone().detach().requires_grad_(True)
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

# (-1, seq_len) → (-1, seq_len, 768)
class Bert(nn.Module):
    def __init__(self, model=BertModel.from_pretrained('bert-base-uncased')):
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

# (-1, IM_K, 784) + (-1, seq_len, 768) → （-1, 768）
class Top_down(nn.Module):
    def __init__(self):
        super(Top_down, self).__init__()
        self.norm1 = nn.Linear(784, DIM).to(device)
        self.norm2 = nn.Linear(DIM, DIM).to(device)
        self.norm3 = weight_norm(nn.Linear(DIM, 1), dim=None).to(device)
        self.norm4 = nn.Linear(DIM, DIM).to(device)
        self.drop1 = nn.Dropout(DROP).to(device)
        self.drop2 = nn.Dropout(DROP).to(device)
    def forward(self, v, t):
        v = F.relu(self.norm1(v))
        t = F.relu(self.norm2(t))
        t = torch.mean(t, dim=1)
        att = v * t.unsqueeze(1).repeat(1, IM_K, 1)
        att = F.softmax(self.norm3(self.drop1(att)), 1)
        v = (att * v).sum(1)
        tv = self.drop2(t * v)
        tv = F.relu(self.norm4(tv))
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
        self.w_1 = nn.Linear(DIM, DIM//4).to(device)
        self.w_2 = nn.Linear(DIM//4, DIM).to(device)
        self.dropout = nn.Dropout(DROP).to(device)
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

# (-1, ans_len, 1024) + (-1, que_len, 1024) + (-1, ans_len, que_len) → (-1, 512)
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.multi_head_attention = MultiHeadAttention()
        self.positionwise_feed_forward = PositionwiseFeedForward()
        self.drop1 = nn.Dropout(DROP).to(device)
        self.drop2 = nn.Dropout(DROP).to(device)
        self.layernorm1 = nn.LayerNorm(DIM, eps=1e-6).to(device)
        self.layernorm2 = nn.LayerNorm(DIM, eps=1e-6).to(device)
        self.layernorm3 = nn.LayerNorm(DIM, eps=1e-6).to(device)
    def forward(self, origin, ques, mask):
        answ = self.layernorm1(origin)
        ques = self.layernorm2(ques)
        out, att = self.multi_head_attention(answ, ques, ques, mask)
        origin = origin + self.drop1(out)
        out = self.layernorm3(origin)
        out = self.positionwise_feed_forward(out)
        out = (origin + self.drop2(out)).mean(1)
        return out, att

# output:(-1, dic_len)
class Final_model(nn.Module):
    def __init__(self, dic_len=len(index2word)):
        super(Final_model, self).__init__()
        self.transfer = Transfer()
        self.bert = Bert()
        self.topdown1 = Top_down()
        self.topdown2 = Top_down()
        self.tranformer = Transformer()
        self.norm1 = nn.Linear(3*DIM, DIM//4).to(device)
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
        out = F.relu(self.norm1(out))
        out = self.norm2(self.drop(out))
        out = self.out(out)
        return out, vq_att, va_att, qa_att

# from torchsummary import summary
# summary(Top_down(), [(IM_K, 784),(22, DIM)])
# summary(MultiHeadAttention(), [(4, DIM),(18, DIM),(18, DIM),(4,18)])
# summary(PositionwiseFeedForward(), (4, DIM))

# training_loading
class labelsmoothing(nn.Module):
    def __init__(self, smoothing=0.1, dic_len=len(index2word), bias=3):
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

def show_att(question,answer,attention):
    fig = plt.figure()
    fig.set_dpi(300)
    ax = fig.add_subplot(111)
    cax = ax.matshow(attention.detach().cpu().numpy(), cmap='bone')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + tokenizer.convert_ids_to_tokens(question), rotation=90)
    ax.set_yticklabels([''] + tokenizer.convert_ids_to_tokens(answer[1:]))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()
   
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

def train(model, training_data, validation_data, optimizer):
    log_train_file = '/home/yzhou/VQA/data/weight/train.log'
    log_valid_file = '/home/yzhou/VQA/data/weight/valid.log'
    with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
        log_tf.write('epoch, loss, ppl, accuracy\n')
        log_vf.write('epoch, loss, ppl, accuracy\n')

    valid_losses = []
    patient = 0
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True)
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
            torch.save(model, '/home/yzhou/VQA/data/weight/model.pkl')
            print('    - [Info] The checkpoint file has been updated.')
            if epoch_i > 6:
                for i in range(len(valid_ques)):
                    show_att(valid_ques[i],valid_answ[i],valid_att[i][0])
        else:
            patient += 1
            if patient > 20:
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
train(model, train_loader, valid_loader, model_opt)

# Visualization
loss_train, loss_valid, accu_train, accu_valid = [], [], [], []
f = open('/home/yzhou/VQA/data/weight/train.log')
next(f)
lines = f.readlines()
for line in lines:
    line = line.strip()
    loss_train.append(float(line.split(', ')[1]))
    accu_train.append(float(line.split(', ')[3]))
f.close()
f = open('/home/yzhou/VQA/data/weight/valid.log')
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
plt.savefig("/home/yzhou/VQA/data/weight/loss.jpg")

fig = plt.figure()
fig.set_dpi(300)
plt.plot(accu_train)
plt.plot(accu_valid)
plt.title('Accuracy')
plt.ylabel('accu')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='lower right')
plt.show()
plt.savefig("/home/yzhou/VQA/data/weight/accuracy.jpg")

# evaluation
