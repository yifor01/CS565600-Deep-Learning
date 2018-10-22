import pandas as pd

NUM_PROGRAM = 8
programs = []
for i in range(1, NUM_PROGRAM+1):
    program = pd.read_csv('Program0%d.csv' % (i))
    
    print('Program %d' % (i))
    print('Episodes: %d' % (len(program)))
    print(program.columns)
    print()
    
    print(program.loc[:1]['Content'])
    print()
    
    programs.append(program)


questions = pd.read_csv('Question.csv')

# Question view
for j in range(89,110):
    print("Q",j+1,":",questions.loc[j]['Question'])
    for i in range(6):
            print("A",i+1,":",questions.loc[j]['Option%d' % (i)])
    print('--'*30) 

#########################################################################
# 拆句子
import jieba
jieba.set_dictionary('big5_dict.txt')

NUM_PROGRAM = 8

# 停用詞
with open('jieba_extra/stop_words.txt', 'r', encoding='utf8') as f:  
    stops = f.read().split('\n') 

# 自建辭典
jieba.load_userdict("jieba_extra/my.dict.txt")


# 停用詞
with open('jieba_extra/stop_words.txt', 'r', encoding='utf8') as f:  
    stops = f.read().split('\n') 


def jieba_lines(lines):
    cut_lines = []
    
    for line in lines:
        cut_line = jieba.lcut(line)
        cut_lines.append(cut_line)
    
    return cut_lines

# 拆成8篇句子群   cut_programs[節目][文章][句子]
cut_programs = []

for program in programs:
    episodes = len(program)
    cut_program = []
    
    for episode in range(episodes):
        lines = program.loc[episode]['Content'].split('\n')
        cut_program.append(jieba_lines(lines))
    
    cut_programs.append(cut_program)

print("%d programs" % len(cut_programs))
print("%d episodes in program 0" % len(cut_programs[0]))
print("%d lines in first episode of program 0" % len(cut_programs[0][0]))

print()
print("first 3 lines in 1st episode of program 0: ")
print(cut_programs[0][0][:3])

#########################################################################
# 拆問題
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
print("%d questions" % len(cut_questions))
print(len(cut_questions[0]))

# 1 question
print(cut_questions[0][0])

# 6 optional reponses
for i in range(1, 7):
    print(cut_questions[0][i])
    
import numpy as np
# 存檔
np.save('cut_Programs.npy', cut_programs)
np.save('cut_Questions.npy', cut_questions)




#########################################################################
#########################################################################
#########################################################################

# 下次重開spyder只要跑底下的
import numpy as np
import pandas as pd
import jieba
import operator
jieba.set_dictionary('big5_dict.txt')

NUM_PROGRAM = 8

# 停用詞
with open('jieba_extra/stop_words.txt', 'r', encoding='utf8') as f:  
    stops = f.read().split('\n') 

# 自建辭典
jieba.load_userdict("jieba_extra/my.dict.txt")


cut_programs = np.load('cut_Programs.npy')
cut_questions = np.load('cut_Questions.npy')


# 針對train data 建立詞頻字典
word_dict = dict()
def add_word_dict(w):
    if not w in word_dict:
        word_dict[w] = 1
    else:
        word_dict[w] += 1

# 蒐集字 建立字頻表
for program in cut_programs:
    for lines in program:
        for line in lines:
            for w in line:
                add_word_dict(w)
for question in cut_questions:
    lines = question[0]
    for line in lines:
        for w in line:
            add_word_dict(w)
    
    for i in range(1, 7):
        line = question[i]
        for w in line:
            add_word_dict(w)


word_dict = sorted(word_dict.items(), key=operator.itemgetter(1), reverse=True)
print("Total %d words in word_dict" % len(word_dict))

# 選定詞頻範圍(38~81632)
VOC_SIZE = 130000 
VOC_START = 20
voc_dict = word_dict[VOC_START:VOC_START+VOC_SIZE]

# 設定 辭典列表
voc_dict2 = []
for i in range(len(voc_dict)):
    voc_dict2.append(voc_dict[i][0] )


# 建立 data
import random
NUM_TRAIN = 20000
TRAIN_VALID_RATE = 0.7
def generate_training_data():
    Xs, Ys = [], []
    
    for i in range(NUM_TRAIN):
        pos_or_neg = random.randint(0, 1)
        if pos_or_neg==1:
            program_id = random.randint(0, NUM_PROGRAM-1)
            episode_id = random.randint(0, len(cut_programs[program_id])-1)
            line_id = random.randint(0, len(cut_programs[program_id][episode_id])-2)
           
            Xs.append([[t for t in cut_programs[program_id][episode_id][line_id] if t not in stops],
                       [t for t in cut_programs[program_id][episode_id][line_id+1] if t not in stops]] )
            Ys.append(1)
        else:
            first_program_id = random.randint(0, NUM_PROGRAM-1)
            first_episode_id = random.randint(0, len(cut_programs[first_program_id])-1)
            first_line_id = random.randint(0, len(cut_programs[first_program_id][first_episode_id])-1)
            second_program_id = random.randint(0, NUM_PROGRAM-1)
            second_episode_id = random.randint(0, len(cut_programs[second_program_id])-1)
            second_line_id = random.randint(0, len(cut_programs[second_program_id][second_episode_id])-1)
            Xs.append([[t for t in cut_programs[first_program_id][first_episode_id][first_line_id] if t not in stops],
                       [t for t in cut_programs[second_program_id][second_episode_id][second_line_id] if t not in stops]] )
    
            Ys.append(0)
    return Xs, Ys


Xs, Ys = generate_training_data()
#####################################################
# 定義函數:選字典中的詞
def choosevoc(dat):
    kkkk = []
    for w in dat:
        choose = ''
        if w in voc_dict2 and w not in stops:
            choose = w
            kkkk.append(choose)
    return kkkk


## 取 Xs 中有在字典裡的字為 new_Xs
new_Xs = Xs.copy()
for i in range(NUM_TRAIN):
    for j in range(1):
        new_Xs[i][j] = choosevoc(Xs[i][j])
    if i % 200 ==0 :
        print("目前執行到",i+1,"/",NUM_TRAIN)

print(new_Xs[:10])




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

# 處理 Question
qaaaaaa=[]
for i in range(len(cut_questions)):
    qaaaaaa.append(abc(cut_questions[i][0]))
    qaaaaaa.append(cut_questions[i][1])
    qaaaaaa.append(cut_questions[i][2])
    qaaaaaa.append(cut_questions[i][3])
    qaaaaaa.append(cut_questions[i][4])
    qaaaaaa.append(cut_questions[i][5])
    qaaaaaa.append(cut_questions[i][6])
print(qaaaaaa[:5])


# 問題與六個選項合併(只選出在字典裡的字)
new_qa = qaaaaaa.copy()
for i in range(3500):
    new_qa[i] = choosevoc(qaaaaaa[i])
    if i %100 == 0:
        print("目前執行到",i+1,"/",3500)
print(new_qa[:10])

####################################################################################

# 建立 word2vec 
import warnings
warnings.filterwarnings('ignore')
from gensim.models import Word2Vec

# model = Word2Vec(abc(new_Xs), min_count=3) #把new_Xs中的所有單字放入當模型的單字庫
# iter(訓練次數) 必須調整
size_ = 200
model = Word2Vec(abc(new_Xs)+new_qa, size=size_, window=5, min_count=1, workers=4,iter=5)


model.wv['音樂','錢包']
model.wv['媽給']

print(model.wv.similarity('老師', '學生'))
print(model.wv.similarity('認真', '豬'))
print(model.wv.similarity('認真', '笨'))
model.wv[new_Xs[0][0]].sum(axis=0)



import scipy

data_combined = []
for i in range(len(new_Xs)):
    if new_Xs[i][0]==[]: 
        first = np.zeros(size_,)
    else: 
        first = model.wv[new_Xs[i][0]].sum(axis=0)
    if new_Xs[i][1]==[]: 
        second = np.zeros(size_,)
    else: 
        second = model.wv[new_Xs[i][1]].sum(axis=0)
    if new_Xs[i][0]==[] or new_Xs[i][1]==[]: 
            dist = 0
    else:
        dist = scipy.spatial.distance.correlation(first, second) # cos值
    sec_combined = np.hstack((first, second, dist))
    data_combined.append(sec_combined)
print(data_combined[:5])  # 兩句話皆轉成詞向量相加的資料集

data_combined2 = pd.DataFrame(data_combined)
data_combined2['Y'] = Ys
data_combined2[:5]




# model 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


X = data_combined2.drop('Y', 1)
y = data_combined2['Y']

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=1)

# RF 
forest = RandomForestClassifier(criterion='entropy',
                                n_estimators=300, 
                                random_state=1,
                                n_jobs=3)
forest.fit(X_train, y_train)

print('Train accuracy (RF): %.3f' % accuracy_score(y_train, forest.predict(X_train)))
print('Valid accuracy (RF): %.3f' % accuracy_score(y_test, forest.predict(X_test)))
