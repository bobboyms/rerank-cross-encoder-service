from sentence_transformers import SentenceTransformer


model = SentenceTransformer('cross-encoder/ms-marco-TinyBERT-L-2')
model.save('./model')
