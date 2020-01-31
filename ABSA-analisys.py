import pandas as pd
import numpy as np
import urllib.request
import torchtext
import random
from janome.tokenizer import Tokenizer
from torchtext.vocab import Vectors

j_t = Tokenizer()

def get_ABSA_Dataloder_and_Text(max_length = 125, batch_size = 24):
    def tokenizer_janome(text):
        return [tok for tok in j_t.tokenize(text, wakati=True)]
    
    def tokenizer_with_preprocessing(text):
        ret = tokenizer_janome(text)  # Janomeの単語分割
        return ret
    
    TEXT = torchtext.data.Field(sequential=True, tokenize=tokenizer_with_preprocessing,
                            use_vocab=True, lower=True, include_lengths=True, batch_first=True, fix_length=max_length)
    LABEL = torchtext.data.Field(sequential=False, use_vocab=False)
    LABEL2= torchtext.data.Field(sequential=False, use_vocab=False)
    LABEL3= torchtext.data.Field(sequential=False, use_vocab=False)
    
    train_val_ds, test_ds= torchtext.data.TabularDataset.splits(
        path='E:\chABSA-dataset', train='centiment_ch-ABSA2.csv',
        test='centiment_ch-ABSA2.csv', format='csv',
        fields=[('Label', LABEL),('Label2',LABEL2), ('Text', TEXT),('Label3', LABEL3) ])
    
    train_ds, val_ds = train_val_ds.split(
        split_ratio= 0.8, random_state=random.seed(1234))
    
    japanese_fastText_vectors = Vectors(name='C:\\Users\\yamakazu\\vector_neologd\\model.vec')
    
    TEXT.build_vocab(train_ds, vectors=japanese_fastText_vectors, min_freq=10)
    
    train_dl = torchtext.data.Iterator(train_ds, batch_size=batch_size, train=True)
    val_dl = torchtext.data.Iterator(val_ds, batch_size=batch_size, train=False, sort=False)
    
    return train_dl, val_dl, test_dl, TEXT
