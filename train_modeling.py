file_name = 'test.txt'
epochs = 1000
learning_rate = 0.01
batch_size = 20
embedding_size = 3

import operator
import pickle
import tensorflow as tf
import numpy as np

# n_gram vocab 생성

vocab = {}
def make_vocab(sen):
    for i in sen:
        if(i in vocab):
            vocab[i] += 1
        else:
            vocab[i] = 1

def sen2ngram(sen, line_len, n):
    leng = line_len - n
    ngrams = [sen[i:i+n] for i in range(0, leng+1)]
    return ngrams

line_vocab = {}

def check_line_vocab(ngrams,line_num):
    for ngram in ngrams:
        line_vocab.setdefault(ngram, []).append(line_num)    

with open(file_name, 'r') as f:
    lines = f.readlines()

line_num = 0
for line in lines:
    line_num +=1
    line = line.replace('\n','')
    sentence = tuple(line.split(' '))
    make_vocab(sentence)

    ngrams = sen2ngram(sentence, len(sentence), 2)
    if ngrams == [] : continue
    
    check_line_vocab(ngrams, line_num)

# todo : 가중치 주기 위한 보캡
sorted_vocab = sorted(vocab.items(), key=operator.itemgetter(1), reverse=True)
# print(sorted_vocab)
word_list = []

for word in sorted_vocab:
    word_list.append(word[0])

# print(word_list)
# 워드에 인덱스를 더한 보캡 생성
word_dict = {w: i for i,w in enumerate(word_list)}
word_dict_reverse = {i: w for i,w in enumerate(word_list)}
# print(word_dict)

# 빠르게 사용할 수 있는 abcd 태그 { 2 ngram 단어 , 라인 넘버 } 저장한 최종 보캡  
ngram_vocab = {}
for i in line_vocab:
    ngram_vocab.setdefault(i[0][0], []).append([i, line_vocab[i]])

# print(ngram_vocab)

# 파일 저장
with open(r'word_dict.pickle', 'wb') as f:
    pickle.dump(word_dict, f)

with open(r'word_dict_reverse.pickle', 'wb') as f:
    pickle.dump(word_dict_reverse, f)

with open(r'ngram_vocab.pickle', 'wb') as f:
    pickle.dump(ngram_vocab, f)

# word2vec training
# 2 skip gram 방식으로 보캡 생성
skipgram_vocab = []
for line in lines:
    line = line.replace('\n','')
    sentence = tuple(line.split(' '))
    for i in range(1, len(sentence) -1):
        skipgram_vocab.append([word_dict[sentence[i]] , word_dict[sentence[i-1]]])
        skipgram_vocab.append([word_dict[sentence[i]] , word_dict[sentence[i+1]]])

voc_size = len(word_list)

inputs = tf.placeholder(tf.int32, shape = [batch_size])
labels = tf.placeholder(tf.int32, shape = [batch_size, 1])

embeddings = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0,  1.0))
selected_embeddings = tf.nn.embedding_lookup(embeddings, inputs)

nce_weights = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
nce_biases = tf.Variable(tf.zeros([voc_size]))

loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                    biases=nce_biases, 
                                    labels=labels, 
                                    inputs=selected_embeddings, 
                                    num_sampled = min(64, line_num), 
                                    num_classes=voc_size))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

def make_batch(data, size):
    input_batchs = []
    label_batchs = []
    
    indexs = np.random.choice(range(len(data)), size, replace=False)

    for i in indexs:
        input_batchs.append(data[i][0])
        label_batchs.append([data[i][1]])

    return input_batchs , label_batchs

for step in range(1, epochs+1):
    batch_inputs, batch_labels = make_batch(skipgram_vocab, batch_size)
    _, loss_val = sess.run([optimizer, loss], feed_dict={inputs: batch_inputs, labels: batch_labels})
    
    if step % 200 ==0:
        print("loss : {}".format(step, loss_val))

get_embeddings = sess.run(embeddings)

# 워드 vec 파일 저장 순서는 word_dict 과 동일
with open(r'word_embeddings.pickle', 'wb') as f:
    pickle.dump(get_embeddings, f)

# 전체 sentence vector 화

sentence_vec = {}

for i, line in enumerate(lines):
    sentence = line.replace('\n', '')
    sentence = sentence.split(' ')
    sentence_vec[i] = np.zeros(embedding_size,dtype='float32')
    # 평균 값을 구하기 위해 나눠줌
    sen_len = float(len(sentence))
    for sen in sentence:    
        idx = word_dict[sen]
        vec_word = np.array(get_embeddings[idx])
        sentence_vec[i] += vec_word/sen_len
        
# print(sentence_vec)

# sentence vec 파일저장
with open(r'sentence_embeddings.pickle', 'wb') as f:
    pickle.dump(sentence_vec, f)

