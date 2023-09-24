import re
import pipeline
from termcolor import colored
import spacy
from rank_bm25 import BM25Okapi
import string 
import numpy as np
from sklearn.feature_extraction import _stop_words

import streamlit as st
import pdfminer
from pdfminer.high_level import extract_text


nlp_spacy=spacy.load("en_core_web_sm")

def bm25_tokenizer(text):
    tokenized_doc = []
    for token in text.lower().split():
        token = token.strip(string.punctuation)

        if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:
            tokenized_doc.append(token)
    return tokenized_doc

def retrieval(query, top_k_retriver):

    bm25_scores = bm25.get_scores(bm25_tokenizer(query))
    top_n = np.argsort(bm25_scores)[::-1][:top_k_retriver]
    bm25_hits = [{'corpus_id': idx, 
                  'score': bm25_scores[idx], 
                  'docs':docs[idx]} for idx in top_n if bm25_scores[idx] > 0]
    bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)
    
    return bm25_hits

def qa_ranker(query, docs_, top_k_ranker):
    ans = []
    for doc in docs_:
        answer = pipeline.qa_model(question = query, context = doc)
        answer['doc'] = doc
        ans.append(answer)
    return sorted(ans, key=lambda x: x['score'], reverse=True)[:top_k_ranker]

# Code to highlight the answer 
def print_colored(text, start_idx, end_idx):
    return (colored(text[:start_idx]) + \
          colored(text[start_idx:end_idx], 'red', 'on_yellow') + \
          colored(text[end_idx:]))



# Streamlit app title
st.title("PDF Text Analysis App")

# Upload PDF file
pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if pdf_file is not None:
    text=extract_text(pdf_file)
    text = re.sub('\n{1,}', '. ', text)
    text = re.sub("\s{1,}", " ", text)

    len_doc = 400
    overlap = 30 
    docs = []

    i = 0
    while i < len(text):
        docs.append(text[i:i+len_doc])
        i = i + len_doc - overlap

    tokenized_corpus = []
    for doc in docs:
        tokenized_corpus.append(bm25_tokenizer(doc))

    bm25 = BM25Okapi(tokenized_corpus)

    query = st.text_input("Enter your query:")

    if query:
        lvl1=retrieval(query,50)
        top_k_ranker=3
        st.write(f"{len(lvl1)} documents retrived using BM25")
        if len(lvl1) > 0:
            fnl_rank = qa_ranker(query, [l["docs"] for l in lvl1], top_k_ranker)
            for fnl_ in fnl_rank:
                st.write("\n")
                st.markdown(print_colored(fnl_['doc'], fnl_['start'], fnl_['end']),unsafe_allow_html=True)
                st.markdown(colored("Confidence score of ") + colored(str(fnl_['score'])[:4], attrs=['bold']),unsafe_allow_html=True)