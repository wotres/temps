import pickle
import operator
import numpy as np

file_name = 'test.txt'
ans_num = 3
embedding_size = 3

## test 코드
# 전체 문장
with open(file_name, 'r') as f:
    lines = f.readlines()
# 보캡 : 인덱스
with open(r'word_dict.pickle', 'rb') as f:
    word_dict = pickle.load(f, encoding='UTF-8')
# 인덱스 : 보캡
with open(r'word_dict_reverse.pickle', 'rb') as f:
    word_dict_reverse = pickle.load(f, encoding='UTF-8')
# n_gram 보캡 == 태그 아래 문장 n_gram 및 라인 숫자 
with open(r'ngram_vocab.pickle', 'rb') as f:
    ngram_vocab = pickle.load(f, encoding='UTF-8')
# 워드 임베딩 된 것 순서는 word_dict 과 동일
with open(r'word_embeddings.pickle', 'rb') as f:
    word_embeddings = pickle.load(f, encoding='UTF-8')
# 문장 임베딩 된 것 순서는 라인 순서
with open(r'sentence_embeddings.pickle', 'rb') as f:
    sentence_embeddings = pickle.load(f, encoding='UTF-8')

# 3가지 경우를 리턴

test = 'java file read'
text = tuple(test.split(' '))

# 1. query 중 n_gram 에 맞는 거만 들고옴
## todo : 일단은 2n_gram만 , 2n, 3n 될수록 가중치 줘야함 --> 그리고 vocab 에서 많이 등장한 경우는 가중치 적게? 고민 해봐야함.. 
def test2ngram(sen, line_len, n):
    leng = line_len - n
    ngrams = []
    for i in range(0, leng+1):
        s1 = tuple([sen[i]])
        s2 = tuple([sen[i+1]])
        ngrams += [(s1+s2)]
        ngrams += [(s2+s1)]
    return ngrams

# query -> n_gram
test_ngram = test2ngram(text,len(text), 2)
print(test_ngram)
result = {}
for i in test_ngram:
    for j in ngram_vocab[i[0][0]]:
        if i == j[0]:
            for k in j[1]:
                if (k in result):
                    result[k] += 1
                else:
                    result[k] =1

print(result)
# n_gram 보캡 가져온 뒤 높은 순으로 정렬
sorted_result = sorted(result.items(), key=operator.itemgetter(1), reverse=True)
print(sorted_result)

for i,s in enumerate(sorted_result):
    print('n_gram answer : ', lines[s[0]-1])
    if i == (ans_num-1) :
        break;
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
# 2. cosine similarity 가 높은거만 들고옴
scores = {}

def sentence_cosine_similarity(vec):
    for i in sentence_embeddings:
        scores[i] = np.dot(vec, sentence_embeddings[i])/(np.linalg.norm(vec)*np.linalg.norm(sentence_embeddings[i]))
    sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_scores

# query -> vec 화 함        
text_vec = np.zeros(embedding_size, dtype='float32')
text_len = float(len(text))
for sen in text:
    idx = word_dict[sen]
    vec_word = np.array(word_embeddings[idx])
    text_vec += vec_word/text_len

score = sentence_cosine_similarity(text_vec)

# 가장 높은 유사도 계산 후 가져옴 3개 정도
for i,s in enumerate(score):
    print('cosine similarity answer : ', s[1] ,' :: ' , lines[s[0]-1])
    if i == (ans_num-1) :
        break
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

# 3. query 중 n_gram 에 맞는 것 중 cosine similarity 순서대로 들고옴
mutual_answer = []

for i in sorted_result:
    for j in score:
        if i[0] == j[0]:
            mutual_answer.append(j)
            break;    
        if j[1] < 0.5 :
            break
    if len(mutual_answer) == (ans_num -1) :
        break

for i in mutual_answer:
    print('mutual answer : ', i[1] ,' :: ' , lines[i[0]-1])
    
# todo: 많이 사용되지 않는 단어에 가중치를 주어 그 단어 위주로 들고옴
