# Задание 1. Напишите два поисковика на BM25: 
# 1) через подсчет метрики по формуле для каждой пары слово-документ, 
# 2) через умножение матрицы на вектор.

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re
from pymorphy2 import MorphAnalyzer
from nltk.tokenize import WordPunctTokenizer
from math import log
from datetime import datetime
from sklearn.metrics import accuracy_score


morph = MorphAnalyzer()
vec = CountVectorizer()

data = pd.read_csv("quora_question_pairs_rus.csv", index_col='Unnamed: 0')

# 1) Подсчет метрики по формуле для каждой пары слово-документ

def preprocessing(s):
    tdirty = str(s)
    tdirty = re.sub(r'[A-Za-z0-9<>«»\.!\(\)?,;:\-\"\ufeff]', r'', tdirty)
    text = WordPunctTokenizer().tokenize(tdirty)
    preprocessing_text = ''
    for w in text:
        new_w = morph.parse(w)[0].normal_form + ' '
        preprocessing_text += new_w
    return preprocessing_text

def indexing(col):
    texts_words = []
    idxs = []
    for idx, text in enumerate(col):
        ws = preprocessing(text)
        texts_words.append(ws)
        idxs.append(idx)
    global vec, vec_bin
    vec = CountVectorizer()
    vec_bin = CountVectorizer(binary=True) 
    X = vec.fit_transform(texts_words)
    Y = vec_bin.fit_transform(texts_words)
    df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names(), index=idxs)
    df_bin = pd.DataFrame(Y.toarray(), columns=vec.get_feature_names(), index=idxs)
    return df, df_bin

def vectorizer(col):
    texts_words = []
    idxs = []
    for idx, text in enumerate(col):
        ws = preprocessing(text)
        texts_words.append(ws)
        idxs.append(idx)
    X = vec.transform(texts_words)
    Y = vec_bin.transform(texts_words)
    df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names(), index=idxs)    
    df_bin = pd.DataFrame(Y.toarray(), columns=vec.get_feature_names(), index=idxs)
    return df, df_bin

# create idf-table
def idf_table(binary_1, binary_q, N): 
    word_counter_dict = {}
    for column in binary_q.columns:
        col = binary_1[column]
        sum_ = col.sum()
        word_counter_dict[column] = sum_
    inverse_counter = pd.DataFrame.from_dict(word_counter_dict, orient='index')
    inverse_counter = inverse_counter.transpose()
    idfs = {}
    for w in inverse_counter.columns:
        idf = log((N - inverse_counter[w] + 0.5)/(inverse_counter[w] +0.5))
        idfs[w] = idf
    idf_table = pd.DataFrame.from_dict(idfs, orient='index')
    idf_table = idf_table.transpose()
    return idf_table  

# create tf-table
def tf_table(df): 
    df['sum'] = df.sum(axis=1)
    tf_table = df.div(df['sum'], axis=0)
    tf_table = tf_table.fillna(0)
    return tf_table

# avgdl 
def avgdl_dl(data): 
    data['sum'] = data.sum(axis = 1, skipna = True) 
    sums = data['sum']
    avg = data['sum'].mean()
    sums_normalized = sums.div(avg)
    return sums_normalized, avg  

# key - query number; value - list of document numbers related to each query 
def connection(dataset): 
    col = dataset['question1']
    connection_dict = {}
    primary_sentences = {}
    for i in range(len(dataset)):
        curr_sent = col[i]
        if curr_sent in primary_sentences:
            primary_sentences[curr_sent].append(i)
        else:
            primary_sentences[curr_sent] = [i]
    for i in range(len(dataset)):
        curr_sent = col[i]
        if len(primary_sentences[curr_sent]) > 1:
            repeating = primary_sentences[curr_sent]
            for r in repeating:
                connection_dict[r] = repeating
        else:
            connection_dict[i] = [i]
    return connection_dict


q1_df, q1_df_bin = indexing(data['question1'][:100])
q2_df, q2_df_bin = vectorizer(data['question2'][:100])

q1_df = q1_df.fillna(0)
q1_df_bin = q1_df_bin.fillna(0)
q2_df = q2_df.fillna(0)
q2_df_bin = q2_df_bin.fillna(0)

is_dupl = data['is_duplicate'][:100]

dls, avg_dl = avgdl_dl(q2_df)

connection_table = connection(data[:100])

tf_table = tf_table(q1_df)

idfs = idf_table(q1_df_bin, q2_df_bin, q1_df_bin.shape[0])


k = 2.0
b = 0.75

# bm25
def bm_25(doc_idx, query, wordlist, docs_num, tfs, idfs, c1, c2): 
    bm_val = 0
    for w in wordlist[:-1]:
        if query[w] != 0:
            idf = float(idfs[w])
            tf_value = float(tfs.iloc[doc_idx][w])
            bm_i = idf * ((tf_value * c1)/(tf_value + c2))
            bm_val += bm_i
    return bm_val

print('оценка для первых 100 документов и запросов')
def evaluation(doc_data, q_data, correspondence, is_rel, tfs, idfs, dls_norm, k, b): #возвращает датафрейм с соответствием запроса, 5 релевантных документов и указанием, есть ли в пятерке хоть один из совпадающих по теме
    start_time = datetime.now()
    similarity_table = pd.DataFrame(columns=['idx_q1', 'relevant_docs', 'match'])
    N = doc_data.shape[0]
    lemmas_list = list(q_data.columns)
    const_little = k + 1
    for q_idx, q_words in q_data.iterrows():
        relevance_dict = {}
        for d_idx, d_words in doc_data.iterrows():
            len_norm = dls_norm[d_idx]
            const_big = k * (1 - b + b * len_norm)
            bm_25_ = bm_25(d_idx, q_words, lemmas_list, N, tfs, idfs, const_little, const_big)
            relevance_dict[bm_25_] = d_idx
        rel_sorted = sorted(relevance_dict.items(), reverse=True)
        best_5_rel = [el[1] for el in rel_sorted[:5]]
        matches = 0
        for d_idx in correspondence[q_idx]:
            if d_idx in best_5_rel and is_rel[d_idx] == 1:
                matches += 1
        if matches > 0:
            matches = 1
        similarity_table = similarity_table.append({'idx_q1': q_idx, 'relevant_docs': best_5_rel, 'match':matches}, ignore_index=True)
    end_time = datetime.now()
    global timing
    timing = 'Время на исполнение функции: {}'.format(end_time - start_time)
    return similarity_table
            
final = evaluation(q2_df, q1_df, connection_table, is_dupl, tf_table, idfs, dls, k, b)


# metric
print(timing)
acc_score = accuracy_score(list(is_dupl), list(final['match']))
print('Метрика близости:', acc_score)
print('Умножение матриц. Тут можно позволить и 1000 запросов. Готовим их...')
q1_df, q1_df_bin = indexing(data['question1'][:1000])
q2_df, q2_df_bin = vectorizer(data['question2'][:1000])

q1_df = q1_df.fillna(0)
q1_df_bin = q1_df_bin.fillna(0)
q2_df = q2_df.fillna(0)
q2_df_bin = q2_df_bin.fillna(0)

is_dupl = data['is_duplicate'][:1000]

dls, avg_dl = avgdl_dl(q2_df)

connection_table = connection(data[:1000])

tf_table = tf_table(q1_df)

idfs = idf_table(q1_df_bin, q2_df_bin, q1_df_bin.shape[0])
print('запросы приготовлены, делаем перемножение матриц')
#bm25 каждого документа к каждому запросу
def multiplication(queries, docs, tfs, idfs, avg_lens, k, b): 
    start_time = datetime.now()
    tfs = tfs.drop(columns=['sum']) 
    conversion_table = docs.mul(tfs) 
    conversion_table_numerator = conversion_table.mul(k+1) 
    coefficient = avg_lens.mul(b) 
    coefficient = coefficient.add(1-b) 
    coefficient = coefficient.mul(k) 
    conversion_table_denominator = conversion_table.mul(avg_lens, axis=0) 
    tf_factor = conversion_table_numerator.divide(conversion_table_denominator) 
    tf_factor = tf_factor.fillna(0) 
    n = tf_factor.shape[0]
    idf_table = pd.concat([idfs]*n, ignore_index=True) 
    term_table = tf_factor.mul(idf_table, axis=1) 
    queries = queries.T
    bm_25_df = term_table.dot(queries)
    end_time = datetime.now()
    global timing
    timing = 'Время на исполнение функции: {}'.format(end_time - start_time)
    return bm_25_df

def best_5(d, correspondence, is_rel): 
    best_5_df = pd.DataFrame(columns=['idx_q1', 'relevant_docs', 'match'])
    for q_idx, q_row in d.iteritems():
        q_row = q_row.astype('int64')
        best_5 = list(q_row.nlargest(5).index)
        matches = 0
        for d_idx in correspondence[q_idx]:
            if d_idx in best_5 and is_rel[d_idx] == 1:
                matches += 1
        if matches > 0:
            matches = 1
        best_5_df = best_5_df.append({'idx_q1': q_idx, 'relevant_docs': best_5, 'match':matches}, ignore_index=True)
    return best_5_df
        
bm_25_matrices = multiplication(q1_df_bin, q2_df_bin, tf_table, idfs, dls, k, b)
results_matrix = best_5(bm_25_matrices, connection_table, is_dupl)
print(timing)
print('Метрика близости:', accuracy_score(list(is_dupl), list(results_matrix['match'])))

print('Press ENTER to procede to Task 2')

input()

print('Задание 2. Инпут - "рождественские каникулы"')

#10 самых релевантных док-ов к запросу
def best_10_res(query_table, ultimate_dataset): 
    query_table = query_table.astype('int64')
    best_10 = list(query_table.nlargest(10, 0).index)
    best_10_df = ultimate_dataset.loc[best_10,'question2']
    best_10_df = pd.DataFrame(best_10_df)
    return best_10_df
	
query = 'рождественские каникулы'
q_vec = vectorizer([query])
result_query = multiplication(q_vec[1], q2_df_bin, tf_table, idfs, dls, k, b)

results = best_10_res(result_query, data)
print(results)

print('Press ENTER to procede to Task 3')

input()

print('Задание 3')
print('BM25, b=0.75')

b2 = 0.75
bm_25_matrix_b2 = multiplication(q1_df_bin, q2_df_bin, tf_table, idfs, dls, k, b2)
results_matrix_b2 = best_5(bm_25_matrices, connection_table, is_dupl)
print(results_matrix_b2)
print('Метрика близости:',accuracy_score(list(is_dupl), list(results_matrix_b2['match'])))
print()
print('Press ENTER to continue')

input()
print('BM15, b=0')
b3 = 0
bm_25_matrix_b3 = multiplication(q1_df_bin, q2_df_bin, tf_table, idfs, dls, k, b3)
results_matrix_b3 = best_5(bm_25_matrices, connection_table, is_dupl)
print(results_matrix_b3)
print('Метрика близости:',accuracy_score(list(is_dupl), list(results_matrix_b3['match'])))
print()
print('Press ENTER to continue')

input()
print('BM11, b=1')
b4 = 1
bm_25_matrix_b4 = multiplication(q1_df_bin, q2_df_bin, tf_table, idfs, dls, k, b4)
results_matrix_b4 = best_5(bm_25_matrices, connection_table, is_dupl)
print(results_matrix_b4)
print('Метрика близости:', accuracy_score(list(is_dupl), list(results_matrix_b4['match'])))

