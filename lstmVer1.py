import torch
from torch import autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import wordEmbed
from wordEmbed import Embedding

def getEmbedInput(wordEmbed,query,context,embedDim):
	inputSeq = query.split(" ")+context.split(" ")
	inputLen = len(inputSeq)
	embedInput = Variable(torch.zeros(inputLen,1,embedDim))
	for i in range(inputLen):
	   embedInput[i,0]=Variable(torch.FloatTensor(wordEmbed.getEmbed(inputSeq[i])))
	return embedInput

class Net(nn.Module):
	def __init__(self,embedDim,hiddenDim):
		super(Net,self).__init__()
		self.fLSTM = nn.LSTM(embedDim, hiddenDim)
		self.bLSTM = nn.LSTM(embedDim, hiddenDim)
		self.Fc = nn.Linear(2*hiddenDim,2)
	def forward(self,inputEmbed,fHidden,bHidden):
		fX,(fH,fC) = self.fLSTM(inputEmbed,fHidden)
		bX,(bH,bC) = self.bLSTM(inputEmbed,bHidden)
		x = self.Fc(torch.cat((fX[-1],bX[-1]),1))
		m = nn.Softmax()
		x = m(x)
		return x

embedDim = 50
hiddenDim  = 100

net = Net(embedDim,hiddenDim)

query = "this is query"
context = "this is context"

# input, hidden, target
wordEmbed = Embedding("glove.6B.50d.txt")
inputEmbed = getEmbedInput(wordEmbed,query,context,embedDim)

fHidden = (autograd.Variable(torch.randn(1,1,hiddenDim)),autograd.Variable(torch.randn(1,1,hiddenDim)))
bHidden = (autograd.Variable(torch.randn(1,1,hiddenDim)),autograd.Variable(torch.randn(1,1,hiddenDim)))

target = autograd.Variable(torch.Tensor([0,1]))

# loss
criterion = nn.MSELoss()
print(net(inputEmbed,fHidden,bHidden))
print(target)
loss = criterion(net(inputEmbed,fHidden,bHidden),target)
print (loss)

# optimizer
optimizer = optim.SGD(net.parameters(),lr=0.01)

# train
optimizer.zero_grad()
loss.backward()
optimizer.step()

loss = criterion(net(inputEmbed,fHidden,bHidden),target)
print(loss)

