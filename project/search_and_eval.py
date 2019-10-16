# -*- coding: utf-8 -*-

from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import preproc
import pandas as pd
import numpy as np
#from sklearn.externals 
import joblib
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from preprocessing import getting_fasttext, sent_vectorizer
import warnings
warnings.filterwarnings("ignore")

#w2v = getting_fasttext('fasttext/model.model')#ValueError: array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size.

def query_preprocessing(query, model):
    query_preprocessed = preproc(query)
    return query_preprocessed

def metric(query, model):
    if model == 'TF-IDF':
        df = pd.DataFrame.from_csv('tf_idf_index.csv', index_col=None)
        vectorizer = joblib.load('tf_idf_vectorizer.pkl')
        query_tfidf = vectorizer.transform([query])
        query_tfidf = pd.DataFrame(query_tfidf.toarray(), columns=vectorizer.get_feature_names())
        metric_value = td_metric(query_tfidf, df)
        metric_value = pd.DataFrame.from_dict(metric_value, orient='index',columns=['val'])


    elif model == 'BM25':
        df = pd.DataFrame.from_csv('bm25_index.csv', index_col=None)
        query = query.split(' ')
        
        lemmas_list = list(df.columns)
        query_bm25 = {}
        for lemma in lemmas_list:
            if lemma in query:
                query_bm25[lemma] = [1]
            else:
                query_bm25[lemma] = [0]

        query_bm25 = pd.DataFrame.from_dict(query_bm25, orient='index',columns=['val'])

        result = df.dot(query_bm25)
        result = result.sort_values(by='val', ascending=False)[:10]
        metric_value = pd.DataFrame(result)

    elif model == 'FastText':
      ##  df = pd.DataFrame.from_csv('fasttext_index.csv', index_col=None)
      ##  sent_vector = sent_vectorizer(query, w2v)
      ##  print(sent_vector)
      ##  metric_value = td_metric(query_tfidf, df)
      ##  metric_value = pd.DataFrame.from_dict(metric_value, orient='index',columns=['val'])
      ##  print(metric_value)
        metric_value = 'NAN'

    else:
        metric_value = 'NAN'
    return metric_value

def td_metric(query, data):
    results = {}
    print(data.columns)
    for index, row in data.iterrows():
        vector = row.as_matrix()
        cos_sim = cosine_similarity(vector.reshape(1, -1), query)
        cos_sim = np.asscalar(cos_sim)
        results[cos_sim] = index
    sorted_results = sorted(results.items(), reverse=True)[:10]
    sorted_results = {v:k for k, v in sorted_results}
    return sorted_results
    
#def w2v_metric(query, data):
    

def get_ultimate_sentences(res):
    ultimate_df = pd.DataFrame.from_csv("quora_question_pairs_rus.csv", index_col='Unnamed: 0')
    #res["document"] = pd.Series()
    #res['document'] = res.apply(lambda row: ultimate_df.iloc[row.index, 'question1'])
    triple_dict = {}
    counter = 1
    for idx, row in res.iterrows():
        #print(row)
        triple_dict[counter] = [row['val'], ultimate_df.loc[idx, 'question1']]
        counter += 1
        #res.iloc[idx, 'document'] = ultimate_df.iloc[idx, 'question1']
    return triple_dict    
    #return None

def search(query, model):
    query = query_preprocessing(query, model)
    #print(query)
    evaluation_df = metric(query, model)
    #print(evaluation_df)
    final_df = get_ultimate_sentences(evaluation_df)
    return final_df

#def main():
#    search(input('Введите запрос: '), input('Введите модель: '))

#if __name__ == "__main__":
#    main()
