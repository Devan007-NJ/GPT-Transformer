import tensorflow as tf
from tensorflow import keras
#This is a module for constructing the positional encodings of a GPT Model which occurs at the input head
class GPTPosEncode(keras.layers.Layer):
    def __init__(self,seq_len,d_model):
        super().__init__()
        self.d_model=d_model
        self.positional_embedding=keras.layers.Embedding(input_dim=seq_len,output_dim=d_model) #Learned Matrix
    def call(self,input):
        seq_len=tf.shape(input)[1]
        pos_vectors=self.positional_embedding(tf.range(start=0, limit=seq_len, delta=1))
        #adding the word and pos embeddings
        return input+ pos_vectors



