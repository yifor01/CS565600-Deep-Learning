import numpy as np
import pandas as pd

NUM_PROGRAM = 8
# 載入 programs
programs = []
for i in range(1, NUM_PROGRAM+1):
    program = pd.read_csv('Program0%d.csv' % (i))
    programs.append(program)
# 載入 questions
questions = pd.read_csv('Question.csv')
for i in range(6):
    print(questions.loc[:2]['Option%d' % (i)])
    print()

import jieba
jieba.set_dictionary('big5_dict.txt')

def jieba_lines(lines):
    cut_lines = []
    
    for line in lines:
        cut_line = jieba.lcut(line)
        cut_lines.append(cut_line)
    
    return cut_lines
cut_programs = []

for program in programs:
    episodes = len(program)
    cut_program = []
    
    for episode in range(episodes):
        lines = program.loc[episode]['Content'].split('\n')
        cut_program.append(jieba_lines(lines))
    
    cut_programs.append(cut_program)

cut_questions = []
n = len(questions)

for i in range(n):
    cut_question = []
    lines = questions.loc[i]['Question'].split('\n')
    cut_question.append(jieba_lines(lines))
    
    for j in range(6):
        line = questions.loc[i]['Option%d' % (j)]
        cut_question.append(jieba.lcut(line))
    
    cut_questions.append(cut_question)

# 存檔
np.save('cut_Programs.npy', cut_programs)
np.save('cut_Questions.npy', cut_questions)
##########################################################################################################################
##########################################################################################################################
import numpy as np
import pandas as pd
NUM_PROGRAM = 8
import jieba
import operator
import random
from gensim.models import Word2Vec
import scipy

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")


jieba.set_dictionary('big5_dict.txt')
cut_programs = np.load('cut_Programs.npy')
cut_Question = np.load('cut_Questions.npy')

# 建立字典
def add_word_dict(w):
    if not w in word_dict:
        word_dict[w] = 1
    else:
        word_dict[w] += 1


word_dict = dict()

for program in cut_programs:
    for lines in program:
        for line in lines:
            for w in line:
                add_word_dict(w)

for question in cut_Question:
    lines = question[0]
    for line in lines:
        for w in line:
            add_word_dict(w)
    
    for i in range(1, 7):
        line = question[i]
        for w in line:
            add_word_dict(w)

word_dict = sorted(word_dict.items(), key=operator.itemgetter(1), reverse=True)

# 抽取一定數量的詞頻字典
VOC_SIZE = 130000
VOC_START = 20

voc_dict = word_dict[VOC_START:VOC_START+VOC_SIZE]
# 設定 辭典列表
voc_dict2 = []
for i in range(len(voc_dict)):
    voc_dict2.append(voc_dict[i][0] )


NUM_PROGRAM = 8
NUM_TRAIN = 30000
TRAIN_VALID_RATE = 0.7


def generate_training_data():
    Xs, Ys = [], []
    
    for i in range(NUM_TRAIN): # 設定樣本數為10000的training set
        pos_or_neg = random.randint(0, 1) # 隨機選擇0或1
        
        if pos_or_neg==1: # 1->是上下文，選擇文句方式為選擇前後接連的兩句
            program_id = random.randint(0, NUM_PROGRAM-1)                            # 八個節目任選一個節目
            episode_id = random.randint(0, len(cut_programs[program_id])-1)          # 選出的節目中的所有集數中任選一集
            line_id = random.randint(0, len(cut_programs[program_id][episode_id])-2) # 從選出的那一集中任選一句話(-2以避免選到最後一句話)
            Xs.append([cut_programs[program_id][episode_id][line_id], 
                       cut_programs[program_id][episode_id][line_id+1]])
            Ys.append(1)
            
        else:             # 0->不是上下文，選擇文句方式為任意選擇兩句
            first_program_id = random.randint(0, NUM_PROGRAM-1)                           # 八個節目任選一個節目
            first_episode_id = random.randint(0, len(cut_programs[first_program_id])-1)   # 選出的節目中的所有集數中任選一集
            first_line_id = random.randint(0, len(cut_programs[first_program_id][first_episode_id])-1) # 從選出的那一集中任選一句話
            second_program_id = random.randint(0, NUM_PROGRAM-1)                          # 八個節目任選一個節目
            second_episode_id = random.randint(0, len(cut_programs[second_program_id])-1) # 選出的節目中的所有集數中任選一集
            second_line_id = random.randint(0, len(cut_programs[second_program_id][second_episode_id])-1) # 從選出的那一集中任選一句話
            Xs.append([cut_programs[first_program_id][first_episode_id][first_line_id], 
                       cut_programs[second_program_id][second_episode_id][second_line_id]])
            Ys.append(0)
    
    return Xs, Ys



Xs, Ys = generate_training_data()

## 定義:一句話中取只有在字典裡的字
def choosevoc(dat):
    example_doc2 = []
    for w in dat:
        choose = ''
        if w in voc_dict2:
            choose = w
            example_doc2.append(choose)
        else:
            choose = ''
    return example_doc2

## 取 Xs 中有在字典裡的字為 new_Xs
new_Xs = Xs.copy()
for i in range(NUM_TRAIN):
    for j in range(2):
        new_Xs[i][j] = choosevoc(Xs[i][j])
    if i % 1000 ==0:
        print("目前進度",i+1,"/",NUM_TRAIN)


# 定義:把每句對話串在一起
def abc(dat):
    doc = []
    for li in dat:
        for w in li:
            doc.append(w)
    return doc 


# 定義:把每句字詞串在一起
def abcd(dat):
    doc = []
    for li in dat:
        for w in li:
            for qq in w:
                doc.append(qq)
    return doc


## 問題與六個選項合併
qaaaaaa=[]
for i in range(len(cut_Question)):
    qaaaaaa.append(abc(cut_Question[i][0]))
    qaaaaaa.append(cut_Question[i][1])
    qaaaaaa.append(cut_Question[i][2])
    qaaaaaa.append(cut_Question[i][3])
    qaaaaaa.append(cut_Question[i][4])
    qaaaaaa.append(cut_Question[i][5])
    qaaaaaa.append(cut_Question[i][6])
print(qaaaaaa[:5])


# 問題與六個選項合併(只選出在字典裡的字)
new_qa = qaaaaaa.copy()
for i in range(3500):
    new_qa[i] = choosevoc(qaaaaaa[i])
    if i % 500 ==0:
        print("目前進度",i+1,"/",3500)


# 將所有字 ensemble 維度為200維
model = Word2Vec(abcd(cut_programs) + qaaaaaa, size=200, window=5, min_count=1, workers=4)



model.wv['音樂','錢包']

print(model.wv.similarity('老師', '學生'))
print(model.wv.similarity('認真', '豬'))
print(model.wv.similarity('認真', '笨'))
model.wv[new_Xs[0][1]].sum(axis=0)



# 兩句話皆轉成詞向量相加的資料集
data_combined = []
for i in range(len(Xs)):
    if new_Xs[i][0]==[]: 
        first = np.zeros(200,)
    else: 
        first = model.wv[new_Xs[i][0]].sum(axis=0)
    if new_Xs[i][1]==[]: 
        second = np.zeros(200,)
    else: 
        second = model.wv[new_Xs[i][1]].sum(axis=0)
    if new_Xs[i][0]==[] or new_Xs[i][1]==[]: 
            dist = 0
    else:
        dist = scipy.spatial.distance.correlation(first, second)
    sec_combined = np.hstack((first, second, dist))
    data_combined.append(sec_combined)
print(data_combined[:5])  


## 題目跟每個問題合併
qa1 = []; qa2 = []; qa3 = []; qa4 = []; qa5 = []; qa6 = []
for i in range(len(cut_Question)):
    c = []
    c.append(abc(cut_Question[i][0]))
    c.append(cut_Question[i][1])
    qa1.append(c)
for i in range(len(cut_Question)):
    c = []
    c.append(abc(cut_Question[i][0]))
    c.append(cut_Question[i][2])
    qa2.append(c)
for i in range(len(cut_Question)):
    c = []
    c.append(abc(cut_Question[i][0]))
    c.append(cut_Question[i][3])
    qa3.append(c)
for i in range(len(cut_Question)):
    c = []
    c.append(abc(cut_Question[i][0]))
    c.append(cut_Question[i][4])
    qa4.append(c)
for i in range(len(cut_Question)):
    c = []
    c.append(abc(cut_Question[i][0]))
    c.append(cut_Question[i][5])
    qa5.append(c)
for i in range(len(cut_Question)):
    c = []
    c.append(abc(cut_Question[i][0]))
    c.append(cut_Question[i][6])
    qa6.append(c)
print(qa5[:5])


## 取 qaX 中有在字典裡的字為 new_qaX
## 定義:取 qaX 中有在字典裡的字為 new_qaX
def indic(data):
    new_data = data
    for i in range(len(data)):
        for j in range(2):
            new_data[i][j] = choosevoc(data[i][j])
    return(new_data)

new_qa1 = indic(qa1)
new_qa2 = indic(qa2)
new_qa3 = indic(qa3)
new_qa4 = indic(qa4)
new_qa5 = indic(qa5)
new_qa6 = indic(qa6)


# 定義:兩句話皆轉成詞向量相加/取平均的資料集
def qc(data):
    data_combined = []
    for i in range(len(data)):
        if data[i][0]==[]: 
            first = np.zeros(200,)
        else: 
            first = model.wv[data[i][0]].sum(axis=0)
        if data[i][1]==[]: 
            second = np.zeros(200,)
        else: 
            second = model.wv[data[i][1]].sum(axis=0)
        if data[i][0]==[] or data[i][1]==[]: 
            dist = 0
        else:
            dist = scipy.spatial.distance.correlation(first, second)
        sec_combined = np.hstack((first, second, dist))
        data_combined.append(sec_combined)
    return(data_combined)

## 將合併的問題與選項轉成向量
qa1_combined = qc(new_qa1)
qa2_combined = qc(new_qa2)
qa3_combined = qc(new_qa3)
qa4_combined = qc(new_qa4)
qa5_combined = qc(new_qa5)
qa6_combined = qc(new_qa6)



######################################################################
# model 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data_combined2 = pd.DataFrame(data_combined)
data_combined2['Y'] = Ys
data_combined2[:5]

X = data_combined2.drop('Y', 1)
y = data_combined2['Y']


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=1)


forest = RandomForestClassifier(criterion='entropy',
                                n_estimators=200, 
                                random_state=1,
                                n_jobs=2)
forest.fit(X_train, y_train)
print('[Train Accuracy (forest)]: %.3f' % accuracy_score(y_train, forest.predict(X_train)))
print('[Valid Accuracy (forest)]: %.3f' % accuracy_score(y_test, forest.predict(X_test)))



forest.fit(X, y)

## 預測
qa1_pred_prob = forest.predict_proba(qa1_combined)
qa2_pred_prob = forest.predict_proba(qa2_combined)
qa3_pred_prob = forest.predict_proba(qa3_combined)
qa4_pred_prob = forest.predict_proba(qa4_combined)
qa5_pred_prob = forest.predict_proba(qa5_combined)
qa6_pred_prob = forest.predict_proba(qa6_combined)


pred_prob = np.column_stack((qa1_pred_prob[:,1],qa2_pred_prob[:,1],qa3_pred_prob[:,1],qa4_pred_prob[:,1],qa5_pred_prob[:,1],qa6_pred_prob[:,1]))
print(pred_prob[:5])
pred_label = pred_prob.argmax(axis=1)
print(pred_label)

#  0.36800
output = {'Id':range(0,500),'Answer':pred_label}
output = pd.DataFrame(output,columns=['Id','Answer'])
print(output[:5])
output.to_csv('output_rf.csv',header=True,index=False)
