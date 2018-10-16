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

print('Question')
print('Episodes: %d' % (len(questions)))
print(questions.columns)
print()

print(questions.loc[:2]['Question'])
print()

for i in range(6):
    print(questions.loc[:2]['Option%d' % (i)])
    print()

###############################################################################
import jieba

jieba.set_dictionary('big5_dict.txt')
example_str = '我討厭吃蘋果'
cut_example_str = jieba.lcut(example_str)
print(cut_example_str)


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

print("%d programs" % len(cut_programs))
print("%d episodes in program 0" % len(cut_programs[0]))
print("%d lines in first episode of program 0" % len(cut_programs[0][0]))

print()
print("first 3 lines in 1st episode of program 0: ")
print(cut_programs[0][0][:3])


###############################################################################

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
    
###############################################################################


import numpy as np

np.save('cut_Programs.npy', cut_programs)
np.save('cut_Questions.npy', cut_questions)

cut_programs = np.load('cut_Programs.npy')
cut_Question = np.load('cut_Questions.npy')




word_dict = dict()
def add_word_dict(w):
    if not w in word_dict:
        word_dict[w] = 1
    else:
        word_dict[w] += 1
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
import operator

word_dict = sorted(word_dict.items(), key=operator.itemgetter(1), reverse=True)
print("Total %d words in word_dict" % len(word_dict))

VOC_SIZE = 15000
VOC_START = 20

voc_dict = word_dict[VOC_START:VOC_START+VOC_SIZE]
print(voc_dict[:10])
print()
print("Total %d words in voc_dict" % len(voc_dict))
np.save('voc_dict.npy', voc_dict)
voc_dict = np.load('voc_dict.npy')



import random

NUM_TRAIN = 10000
TRAIN_VALID_RATE = 0.7
def generate_training_data():
    Xs, Ys = [], []
    
    for i in range(NUM_TRAIN):
        pos_or_neg = random.randint(0, 1)
        
        if pos_or_neg==1:
            program_id = random.randint(0, NUM_PROGRAM-1)
            episode_id = random.randint(0, len(cut_programs[program_id])-1)
            line_id = random.randint(0, len(cut_programs[program_id][episode_id])-2)
            
            Xs.append([cut_programs[program_id][episode_id][line_id], 
                       cut_programs[program_id][episode_id][line_id+1]])
            Ys.append(1)
            
        else:
            first_program_id = random.randint(0, NUM_PROGRAM-1)
            first_episode_id = random.randint(0, len(cut_programs[first_program_id])-1)
            first_line_id = random.randint(0, len(cut_programs[first_program_id][first_episode_id])-1)
            
            second_program_id = random.randint(0, NUM_PROGRAM-1)
            second_episode_id = random.randint(0, len(cut_programs[second_program_id])-1)
            second_line_id = random.randint(0, len(cut_programs[second_program_id][second_episode_id])-1)
            
            Xs.append([cut_programs[first_program_id][first_episode_id][first_line_id], 
                       cut_programs[second_program_id][second_episode_id][second_line_id]])
            Ys.append(0)
    
    return Xs, Ys
Xs, Ys = generate_training_data()

x_train, y_train = Xs[:int(NUM_TRAIN*TRAIN_VALID_RATE)], Ys[:int(NUM_TRAIN*TRAIN_VALID_RATE)]
x_valid, y_valid = Xs[int(NUM_TRAIN*TRAIN_VALID_RATE):], Ys[int(NUM_TRAIN*TRAIN_VALID_RATE):]


example_doc = []

# lines in 1st episode in program 0 
for line in cut_programs[0][0]:
    example_line = ''
    for w in line:
        if w in voc_dict:
            example_line += w+' '
        
    example_doc.append(example_line)

print(example_doc[:10])





import scipy as sp
from sklearn.feature_extraction.text import CountVectorizer

# ngram_range=(min, max), default: 1-gram => (1, 1)
count = CountVectorizer(ngram_range=(1, 2))

count.fit(example_doc)
BoW = count.vocabulary_
print('[vocabulary]\n')
for key in list(BoW.keys())[:10]:
    print('%s %d' % (key, BoW[key]))



# get matrix (line_id, vocabulary_id) --> (this vocabulary occurs how many times in this line)
doc_bag = count.transform(example_doc)
print(' (lid, vid)     tf')
print(doc_bag[:10])

print('\nIs document-term matrix a scipy.sparse matrix? {}'.format(sp.sparse.issparse(doc_bag)))




doc_bag = doc_bag.toarray()
print(doc_bag[:10])

print('\nAfter calling .toarray(), is it a scipy.sparse matrix? {}'.format(sp.sparse.issparse(doc_bag)))





doc_bag = count.fit_transform(example_doc).toarray()

print("[most frequent vocabularies]")

# conpute how many times each word occurs in doc_bag
bag_cnts = np.sum(doc_bag, axis=0)

# get words occur in doc_bag
words_num = bag_cnts.shape[0]
ones_vector = np.ones(words_num)
words = count.inverse_transform(ones_vector)[0]

# sort bag_cnts and get its indices and values
most_freq_word_index = bag_cnts.argsort()
most_freq_word_times = np.sort(bag_cnts)

top = 10
# [::-1] reverses a list since sort is in ascending order
for tok, v in zip(words[most_freq_word_index[::-1][:top]], 
                        most_freq_word_times[::-1][:top]):
    print('%s: %d' % (tok, v))




from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(ngram_range=(1,1))
tfidf.fit(example_doc)

top = 10
# get idf score of vocabularies
idf = tfidf.idf_
print('[vocabularies with smallest idf scores]')
sorted_idx = idf.argsort()
for i in range(top):
    print('%s: %.2f' % (tfidf.get_feature_names()[sorted_idx[i]], idf[sorted_idx[i]]))

doc_tfidf = tfidf.transform(example_doc).toarray()
tfidf_sum = np.sum(doc_tfidf, axis=0)
print("\n[vocabularies with highest tf-idf scores]")
for tok, v in zip(tfidf.inverse_transform(np.ones(tfidf_sum.shape[0]))[0][tfidf_sum.argsort()[::-1]][:top], 
                  np.sort(tfidf_sum)[::-1][:top]):
    print('%s: %.2f' % (tok, v))


from sklearn.feature_extraction.text import HashingVectorizer

hashvec = HashingVectorizer(n_features=2**6)

doc_hash = hashvec.transform(example_doc)
print(doc_hash.shape)

###########################################################################
