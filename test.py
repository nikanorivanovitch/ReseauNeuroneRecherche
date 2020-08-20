import os
import numpy as np
import random


Neg=[[0, 1, 1, 0, 1, 0, 1, 1],[1, 0]]
Pos=[[1, 1, 0, 0, 2, 0, 5, 1],[0, 1]]

def Sigmoid(x):
	return 1/(1+exp(-x))

def Error(V1, V2):
	#print(np.sum(V1-V2)**2)
	return np.sum((V1-V2)**2)

class Neuron:

	def __init__(self, N, V, L=None):

		self.InputNeuron = L
		self.Weights = np.array([random.random()*random.randint(-1, 1) for x in range(N)])
		self.Bias = np.array([random.random()])
		self.Weights/=np.sum(np.abs(self.Weights))
		self.CurrentValue = V
		self.WantedValue = V

		self.Error=0

		self.DeltaWeights = self.Weights*0

	def Compute(self):

		if(self.InputNeuron==None):
			return 0

		self.CurrentValue = np.sum(self.Weights*np.array([N.CurrentValue for N in self.InputNeuron]))

		if(self.CurrentValue<0):
			self.CurrentValue*=0.1

	def Train(self, DataSet):

		Delta = 0.1
		Lambda = 0.1

		for Exemple in DataSet:
			for W in range(len(self.Weights)):

				Network.InputArray(Exemple[0])
				K1 = Network.Compute()
				K1 = Error(K1, Exemple[1])

				self.Weights[W]+=Delta

				Network.InputArray(Exemple[0])
				K2 = Network.Compute()
				K2 = Error(K2, Exemple[1])

				self.Weights[W]-= ((K2-K1)/Delta)*Lambda/len(DataSet) + Delta

class Network:

	def __init__(self):

		self.Layer1 = [Neuron(0, 0, None) for x in range(8)]
		self.Layer2 = [Neuron(8, 0, self.Layer1) for x in range(4)]
		self.Layer3 = [Neuron(4, 0, self.Layer2) for x in range(2)]

	def Compute(self):

		for N in self.Layer1:
			N.Compute()

		for N in self.Layer2:
			N.Compute()

		for N in self.Layer3:
			N.Compute()

		return np.array([self.Layer3[0].CurrentValue, self.Layer3[1].CurrentValue])

	def Train(self, N):

		global Neg
		global Pos

		DataSet = [Neg, Pos]

		for x in range(N):

			for N in self.Layer2:
				N.Train(DataSet)

			for N in self.Layer3:
				N.Train(DataSet)

	def InputArray(self, Array):

		for N in range(len(self.Layer1)):
			self.Layer1[N].CurrentValue=Array[N]


Network = Network()

Network.InputArray(Neg[0])
print(Network.Compute(), Neg[1])
print(Error(Network.Compute(), Neg[1]))

Network.InputArray(Pos[0])
print(Network.Compute(), Pos[1])
print(Error(Network.Compute(), Pos[1]))

Network.Train(80)

Network.InputArray(Neg[0])
print(Network.Compute(), Neg[1])
print(Error(Network.Compute(), Neg[1]))

Network.InputArray(Pos[0])
print(Network.Compute(), Pos[1])
print(Error(Network.Compute(), Pos[1]))






