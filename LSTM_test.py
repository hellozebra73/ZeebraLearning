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


class Decoder(nn.module):
    def __init__(self,vocab_size,embedding_size,hidden_size,output_size,num_layers,p):
        super(Decoder,self).__init__()
        self.embedding=nn.Embedding(vocab_size,embedding_size)
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.rnn=nn.LSTM(embedding_size,hidden_size,num_layers,dropout=p)
        self.fc=nn.linear(hidden_size,output_size)
    
    def forward(self,x,hidden,cell):
        #x has dimmensions [1,N] where N is the batch size. The reason why we have 1 is that we process one word at a time
        # as the next word is usually taken from either the output or of the last step or the target sentence.
        x=x.unsqueeze(0)
        embedding=self.embedding(x)
        #[1,N,embedding_size]
        outputs,(hidden,cell)=self.rnn(embedding,(hidden,cell))
        #outputs.size()=[1,N,hidden_size]
        #the initial hidden comes from the output of the encoder, from the on just from the previous step.

        predictions=self.fc(outputs).squeeze(0)
        #[1,N,target_vocab_size] before squeeze
        #[N,target_vocab_size] after squeeze
        return predictions, hidden, cell

class Seq2Seq(nn.Module):
    def__init__(self,encoder,decoder):
        super(Seq2Seq,self).__init__()
        self.encoder=encoder
        self.decoder=decoder
    def forward(self,source,target,teacher_force_ratio=0.5):
        batch_size=source.shape[1]
        target_len=target.shape[0]
        #here we will save the outputs
        outputs=torch.zeros(target_len,batch_size,target_vocab_size).to(device)
        target_vocab_size=len(english.vocab)
        hidden,cell=self.encoder(source)
        #this is start token
        x=target[0]
        for i in range(1,len(target)):
            output,(hidden,cell)=self.decoder(x,hidden,cell)
            outputs[i]=output
            best_guess=output.argmax[1]
            random=np.rand()
            x=target[i] if random > teacher_force_ratio else best_guess
        return ouputs



