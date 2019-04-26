############################ ngram 으로 vocab 만드는 부분 ############################
import operator

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

### file read 와 같이 명사 + 동사가 의미가 있다고 생각 즉, 2 ngram
line_vocab = {}

def check_line_vocab(ngrams,line_num):
    for ngram in ngrams:
        line_vocab.setdefault(ngram, []).append(line_num)    

# 파일 읽어옴
with open('test.txt', 'r') as f:
    lines = f.readlines()

line_num = 0

for line in lines:
    line_num +=1
    line = line.replace('\n','')
    sentence = tuple(line.split(' '))
    # 보캡 생성 및 빈도수에 따른 정렬
    make_vocab(sentence)
    sorted_vocab = sorted(vocab.items(), key=operator.itemgetter(1), reverse=True)

    # ngram 으로 파일 생성 및 빈도수에 따른 정렬
    # print(sentence)
    ngrams = sen2ngram(sentence, len(sentence), 2)
    # print(ngrams)
    if ngrams == [] : continue
    # ngram 을 이용한 보캡 생성
    check_line_vocab(ngrams, line_num)

### 최종 사용할 ngram 으로 이루어진 vocab 
ngram_vocab = {}
for i in line_vocab:
    ngram_vocab.setdefault(i[0][0], []).append([i, line_vocab[i]])

############################ vocab 만드는 부분 ############################ 만든 뒤 위와 합침
word_list = []

for word in sorted_vocab:
    word_list.append(word[0])

word_dict = {w: i for i,w in enumerate(word_list)}
word_dict_reverse = {i: w for i,w in enumerate(word_list)}
# print(word_dict)

############################ 2 skip gram 방식으로 보캡 생성 ############################ 일단 보캡 때문에 뒤로ㅠㅠ
skipgram_vocab = []
for line in lines:
    line = line.replace('\n','')
    sentence = tuple(line.split(' '))
    print(sentence)
    for i in range(1, len(sentence) -1):
        skipgram_vocab.append([word_dict[sentence[i]] , word_dict[sentence[i-1]]])
        skipgram_vocab.append([word_dict[sentence[i]] , word_dict[sentence[i+1]]])

print(skipgram_vocab)

############################ 트레이닝 with tensorflow ############################
import tensorflow as tf
import numpy as np

epochs = 1000
learning_rate = 0.01
batch_size = 20
embedding_size = 3
voc_size = len(word_list)

inputs = tf.placeholder(tf.int32, shape = [batch_size])
labels = tf.placeholder(tf.int32, shape = [batch_size, 1])

# 각 보캡에 대한 임베딩을 가져오는것 같음 따라서 최종 이것만 호출염..
embeddings = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0,  1.0))
selected_embeddings = tf.nn.embedding_lookup(embeddings, inputs)

# 임베딩 관련 트레이닝때 ncs_loss 사용
nce_weights = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
nce_biases = tf.Variable(tf.zeros([voc_size]))

# 이부분에서 임베딩한 것들을 알려주는 듯하지만
loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, labels, selected_embeddings, line_num, voc_size))
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
# get_embeddings = sess.run(selected_embeddings, feed_dict={inputs: skipgram_vocab})

############################ 2 이상의 차원일 때 차원 축소함 ############################ --> 모든 단어 비교하는 것보다 쉽게 가능할듯
# from sklearn.manifold import TSNE
# # 상대적 차이를 살린채 차원을 축소한다.
# # print(get_embeddings[0])
# model = TSNE(n_components=2)
# print(get_embeddings)
# get_embeddings = model.fit_transform(get_embeddings)
# print(get_embeddings)

############################ 시각화 부분 ############################
# import matplotlib.pyplot as plt
# for i, word in enumerate(word_list):
#     print(word)
#     x, y = get_embeddings[i]
#     plt.scatter(x,y)
#     plt.annotate(word, xy=(x,y))

# plt.show()
############################ 테스트 부분 ############################
test_word = 'firstRunner'
test = 'firstRunner and runs in order'
text = tuple(test.split(' '))

############################ cosine similarity 워드 테스트 부분 ############################
scores = {}
def cosine_similarity(vec, idx, n):
    for i, other_vec in enumerate(get_embeddings):                
        if(i == idx): continue
        scores[i] = np.dot(vec, other_vec)/(np.linalg.norm(vec)*np.linalg.norm(other_vec))
        
    sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
    # print(sorted_scores)
    return sorted_scores[: n]

def n_neighbor(sen, n=1):
    idx = word_dict[sen]
    vec = get_embeddings[idx]
    print(vec)
    print(vec.dtype)
    score = cosine_similarity(vec, idx, n)
    answer = []
    for i in score:
        # print(i)
        answer.append(word_dict_reverse[i[0]])
    print('word_answer : ' , answer)

n_neighbor(test_word, 2)
############################ cosine similarity 문장 테스트 부분 ############################
scores = {}
sentence_vec = {}

def sentence_cosine_similarity(vec, n):
    for i in sentence_vec:
        scores[i] = np.dot(vec, sentence_vec[i])/(np.linalg.norm(vec)*np.linalg.norm(sentence_vec[i]))
    sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
    # print(sorted_scores)
    return sorted_scores[: n]

for i, line in enumerate(lines):
    sentence = line.replace('\n', '')
    sentence = sentence.split(' ')
    # sentence_vec[i] = np.array([0,0],dtype='float32')
    sentence_vec[i] = np.zeros(embedding_size,dtype='float32')
    
    # 평균 값을 구하기 위해 나눠줌
    sen_len = float(len(sentence))
    for sen in sentence:    
        idx = word_dict[sen]
        vec_word = np.array(get_embeddings[idx])
        sentence_vec[i] += vec_word/sen_len
        
print(sentence_vec)
print('~~~~~~~~~~~~~~~')

text_vec = np.zeros(embedding_size,dtype='float32')
text_len = float(len(text))
for sen in text:
    idx = word_dict[sen]
    vec_word = np.array(get_embeddings[idx])
    print(vec_word)
    text_vec += vec_word/text_len

score = sentence_cosine_similarity(text_vec, 2)
print(score)

answer = []
for i in score:
    answer.append(lines[i[0]])
print('sentence_answer : ' , answer)

############################ n_gram 문장 테스트 부분 ############################

def test2ngram(sen, line_len, n):
    leng = line_len - n
    print(sen)
    ngrams = []
    for i in range(0, leng+1):
        ngrams += [sen[i:i+n]]
        ngrams += tuple(reversed([sen[i:i+n]]))
    return ngrams

test_ngram = test2ngram(text,len(text), 2)

result = {}
for i in test_ngram:
    for j in ngram_vocab[i[0][0]]:
        if i == j[0]:
            for k in j[1]:
                if (k in result):
                    result[k] += 1
                else:
                    result[k] =1

sorted_result = sorted(result.items(), key=operator.itemgetter(1), reverse=True)
print('answer : ', lines[sorted_result[0][0]])

############################ 파일 저장 부분 ############################
import pickle
with open(r'sentence.pickle', 'wb') as f:
    pickle.dump(sentence_vec, f)

with open(r'sentence.pickle', 'rb') as f:
    data = pickle.load(f, encoding='UTF-8')

print(data)

##################################################################################

## 키워드 중에 자주 쓰이지 않는 단어가 있다면 그 단어가 포함된 녀석 중에 찾는 경우
## 위와는 반대로 자주 쓰이는 단어가 있다면 이단어가 꼭 포함된 경우
## 자주 사용되는 질문의 경우 미리 일치하는 문장을 리턴하는 로직
## n_gram 으로 기존에 언급된 단어를 우선으로 비교하여 리턴해주는 로직

# vocab_size 설정 및 UNK

'''
sort 와 sorted 차이 및 역순 및 key 순서로 정렬
a = [('a', 2), ('b', 1), ('c', 5), ('d', 4), ('e', 3)]
a.sort(key=lambda x: x[1])
print(a)

import operators
sorted_ngram_list = sorted(ngram_list.items(), key=operator.itemgetter(1), reverse=True)

'''