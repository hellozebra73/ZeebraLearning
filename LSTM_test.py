import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import time



class RNNModel(nn.Module):
    def __init__(self, vocab_size,embed_size,hidden_dim,num_outputs,batch_size):
        super().__init__()
        self.vocab_size=vocab_size
        self.embed_size=embed_size
        self.hidden_dim=hidden_dim
        self.num_outputs=num_outputs
        self.batch_size=batch_size
        self._embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_dim,batch_first=True, dropout=0.01)
        self._output = nn.Linear(hidden_dim, num_outputs)

    def forward(self):


        tokens=torch.from_numpy(self.testArray())
        start = time.time()
        embeds = self._embed(tokens)
        print(embeds.size(),"embeding")
        hidden,_=self.rnn(embeds)
#         embeds = nn.utils.rnn.pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)
        linear=self._output(hidden[-1])
       # print(linear.size(),"linear")

        end = time.time()
        print(end - start,"tiempo")
        
       
        return linear
    def testArray(self):
        sentence_length=np.random.randint(5,15)
        return np.random.randint(0,self.vocab_size,(self.batch_size,sentence_length))

modelo=RNNModel(vocab_size=1000,embed_size=400,hidden_dim=300,num_outputs=200,batch_size=30)
modelo()
