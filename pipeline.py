import torch
import spacy
from transformers import pipeline

qa_model = pipeline("question-answering", model = "deepset/roberta-base-squad2")

nlp_spacy=spacy.load("en_core_web_sm")
