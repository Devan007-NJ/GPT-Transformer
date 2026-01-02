import tensorflow as tf
from tensorflow.keras import layers
from positional_encodings import GPTPosEncode

#making the input matrix (token embedding matrix)
#The token embedding matrix for test purposes is not directly from a sentence of such and is
#generated as a tensor of [Batch_size,Seq_len,d_Model]

#--CONFIGURATIONS--
vocab_size=50257 #GPT-2 vocabulary size
max_len = 1024 #GPT-2 max content
D_MODEL = 768 #GPT-2 Embedding Dimensions

#For input we are choosing a random 5 letter word
#then batch_size=1 (1 sentence)
#then seq_len=5
input_id=tf.constant([[11542,10,25,42,369]]) #token ids from the GPT-2 token embedding table
word_embedding_layer=layers.Embedding(vocab_size,D_MODEL)
pos_embedding_layer=GPTPosEncode(max_seq_len=max_len,d_model=D_MODEL)


#--RUNNING THE TEST--
word_vectors=word_embedding_layer(input_id) #shape should be [1,5.768]:[batch,seq,d_model]
result=pos_embedding_layer(word_vectors)
if result.shape == word_vectors.shape:
    print("Position Encoding Successfull")
    print(f"Actual shape {result.shape}")


