import torch
from torch import autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys
sys.path.insert(0,'Lib')
import wordEmbed
from wordEmbed import Embedding


import re

oodDic = {}
def getEmbedInput(wordEmbed,query,context,embedDim):
	queryTokenList = query.split(" ")
	contextTokenList = context.split(" ")
	newQueryTokenList = []
	newContextTokenList = []
	for i in range(len(queryTokenList)):
		temp = re.sub("[^A-Za-z0-9]+","",queryTokenList[i])
		if (temp != ""):
			newQueryTokenList.append(temp)
	for i in range(len(contextTokenList)):
	   temp = re.sub("[^A-Za-z0-9]+","",contextTokenList[i])
	   if( temp != ""):
		   newContextTokenList.append(temp)

	inputSeq = newQueryTokenList+["::"]+newContextTokenList
	inputLen = len(inputSeq)
	embedInput = Variable(torch.zeros(inputLen,1,embedDim))
	for i in range(inputLen):
		if((inputSeq[i]=="::") and (i == len(queryTokenList)) ):
			embedInput[i,0] = Variable(torch.zeros(50))
		else:
			if(wordEmbed.checkOOD(inputSeq[i])==True):
				embedInput[i,0]=Variable(torch.rand(50))
			else:
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

# EMBEDDING DIM : 50
embedDim = 50
hiddenDim  = 100

net = Net(embedDim,hiddenDim)
net.load_state_dict(torch.load('model_13999'))

data = []
# input train Data
with open(sys.argv[1],'r') as fin:
	for line in fin:
	   data.append(line.rstrip().split("\t"))
fout = open(sys.argv[2],'w')
 
wordEmbed = Embedding("glove.6B.50d.txt")

fHidden = (autograd.Variable(torch.randn(1,1,hiddenDim)),autograd.Variable(torch.randn(1,1,hiddenDim)))
bHidden = (autograd.Variable(torch.randn(1,1,hiddenDim)),autograd.Variable(torch.randn(1,1,hiddenDim)))

rightCount = 0
for i in range(len(data)):

# input, hidden, target
   loss = 0
   context = data[i][0].lower()
   query =  data[i][1].lower()
   targetValue = data[i][2]

   inputEmbed = getEmbedInput(wordEmbed,query,context,embedDim)

   if(targetValue == "T"):
	   target = autograd.Variable(torch.Tensor([0,1]))
   else:
	   target = autograd.Variable(torch.Tensor([1,0]))

   result = net(inputEmbed,fHidden,bHidden)
   print(result)
   if(result.data[0][0]>result.data[0][1]):
	  resultValue = "F"
   else:
	  resultValue = "T"
  
   fout.write(data[i][0]+"\t"+data[i][1]+"\t"+data[i][2]+"\t"+resultValue+"\n")
   if(data[i][2] == resultValue):
	  rightCount += 1

print(float(rightCount)/len(data))
fout.close()
