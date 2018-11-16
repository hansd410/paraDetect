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

import time

oodDic = {}
def getEmbedInput(wordEmbed,query,context,embedDim):
	queryTokenList = query.split(" ")
	contextTokenList = context.split(" ")
	newQueryTokenList = []
	newContextTokenList = []
	print("time a")
	print (time.time())

	for i in range(len(queryTokenList)):
		temp = re.sub("[^A-Za-z0-9]+","",queryTokenList[i])
		if (temp != ""):
			newQueryTokenList.append(temp)
	for i in range(len(contextTokenList)):
	   temp = re.sub("[^A-Za-z0-9]+","",contextTokenList[i])
	   if( temp != ""):
		   newContextTokenList.append(temp)
	print("time b")
	print (time.time())

	inputSeq = newQueryTokenList+["::"]+newContextTokenList
	inputLen = len(inputSeq)
	embedInput = Variable(torch.zeros(inputLen,1,embedDim))
	for i in range(inputLen):
		print("time x")
		print (time.time())

		if((inputSeq[i]=="::") and (i == len(queryTokenList)) ):
			print("time y")
			print (time.time())

			embedInput[i,0] = Variable(torch.zeros(50))
		else:
			print("time z")
			print (time.time())

			if(wordEmbed.checkOOD(inputSeq[i])==True):
				embedInput[i,0]=Variable(torch.rand(50))
			else:
				embedInput[i,0]=Variable(torch.FloatTensor(wordEmbed.getEmbed(inputSeq[i])))
	print("time c")
	print (time.time())

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

positiveData = []
negativeData = []
# input train Data
with open(sys.argv[1],'r') as fin:
    for line in fin:
	   if(line[-2]=="T"):
		  positiveData.append(line.rstrip().split("\t"))
	   else:
		  negativeData.append(line.rstrip().split("\t"))

positiveDataNum =len(positiveData)
negativeDataNum = len(negativeData)
dataSeq = 0
 
wordEmbed = Embedding("glove.6B.50d.txt")

fHidden = (autograd.Variable(torch.randn(1,1,hiddenDim)),autograd.Variable(torch.randn(1,1,hiddenDim)))
bHidden = (autograd.Variable(torch.randn(1,1,hiddenDim)),autograd.Variable(torch.randn(1,1,hiddenDim)))


for i in range(20000):
   print("time1")
   print (time.time())
# input, hidden, target
   batchData = []
   batchSize = 50
   for j in range(batchSize/2):
	  batchData.append(positiveData[dataSeq%positiveDataNum])
	  batchData.append(negativeData[dataSeq%negativeDataNum])
	  dataSeq+=1
   print("time2")
   print (time.time())

   #training
   loss = 0
   print(str(i)+"th epoch running")
   for j in range(batchSize):
	  print("time3")
	  print (time.time())

	  context = batchData[j][0].lower()
	  query =  batchData[j][1].lower()
	  targetValue = batchData[j][2]
	  #print(context)
	  #print(query)

	  inputEmbed = getEmbedInput(wordEmbed,query,context,embedDim)

	  if(targetValue == "T"):
		  target = autograd.Variable(torch.Tensor([0,1]))
	  else:
		  target = autograd.Variable(torch.Tensor([1,0]))
	  print("time4")
	  print (time.time())


# loss
	  criterion = nn.MSELoss()
#   print(net(inputEmbed,fHidden,bHidden))
#   print(target)
	  loss += criterion(net(inputEmbed,fHidden,bHidden),target)
	  print("time5")
	  print (time.time())


# optimizer
   loss = loss/batchSize
   print("from loss :"+str(loss))
   optimizer = optim.SGD(net.parameters(),lr=0.5)
   print("time6")
   print (time.time())

# train
   optimizer.zero_grad()
   loss.backward()
   optimizer.step()
   print("time7")
   print (time.time())

   loss = criterion(net(inputEmbed,fHidden,bHidden),target)
   print("to loss : "+str(loss))

   print("time8")
   print (time.time())

   if(i%1000==999):
	  torch.save(net.state_dict(),'./model_'+str(i))
