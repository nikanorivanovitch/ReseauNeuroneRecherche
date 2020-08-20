import PIL
import random
import os
from multiprocessing import Process, Pipe
import numpy as np
import time
import pickle

from PIL import Image

def Error(Vector1, Vector2):
	Vector3 = Vector1 - Vector2
	return(np.sqrt(np.sum(Vector3*Vector3)))

class DataSet:

	def __init__(self, PathToPositiveData, PathToNegativeData):

		self.PathToPositiveData = PathToPositiveData
		self.PathToNegativeData = PathToNegativeData

		self.ListFileNamePositive = [FileName for FileName in os.listdir(PathToPositiveData) if os.path.isfile(os.path.join(PathToPositiveData, FileName))]
		self.ListFileNameNegative = [FileName for FileName in os.listdir(PathToNegativeData) if os.path.isfile(os.path.join(PathToNegativeData, FileName))]

		print(self.ListFileNamePositive)
		print(self.ListFileNameNegative)

		self.PositiveExemples = [Face(os.path.join(PathToPositiveData, FileName), True) for FileName in self.ListFileNamePositive]
		self.NegativeExemples = [Face(os.path.join(PathToNegativeData, FileName), False) for FileName in self.ListFileNameNegative]

	def Send_N_DataSets(self, N_DataSets, Size):

		DataSetsList = []

		for N in range(N_DataSets):

			print("\n")

			DataSet = []

			for x in range(Size):

				if(random.random()>0.5):

					A = random.randint(0, len(self.PositiveExemples)-1)
					print(A)
					DataSet.append(self.PositiveExemples[A])
					print(self.ListFileNamePositive[A])

				else:

					A = random.randint(0, len(self.NegativeExemples)-1)
					print(A)
					DataSet.append(self.NegativeExemples[A])
					print(self.ListFileNameNegative[A])

			DataSetsList.append(DataSet)

		return DataSetsList

	def Send_All(self):

		R = []

		for P in self.PositiveExemples:

			R.append(P)

		for N in self.NegativeExemples:

			R.append(N)

		return R

class Face:

	def __init__(self, PathToFile, Positive):

		self.IMG = Image.open(PathToFile).convert('L').resize((20,20))
		self.MTX = (np.array(self.IMG)-np.mean(self.IMG))/np.sqrt(np.var(self.IMG))
		self.MTX = [x for vector in self.MTX for x in vector]

		if(Positive):
			self.WantedOuput=np.array([1,0])

		else:
			self.WantedOuput=np.array([0,1])

class Neuron:

	def __init__(self, MyLayerSize, PathToConfigFile=None, NeuronArray=None):

		self.Tag = ""
		self.InputNeuron = []
		self.Weights = []
		self.DeltaWeights = []
		self.MyLayerSize = MyLayerSize

		self.CurrentValue = None
		self.WantedValue = 0

		self.PathToConfigFile=PathToConfigFile

		self.LinkTo(NeuronArray)

	def LinkTo(self, NeuronArray):

		if(NeuronArray!=None):
			for N in NeuronArray:
				self.InputNeuron.append(N)

		self.LoadConfig()

	def LatchOn(self, CopyWantedValue=True):

		self.CurrentValue = 0
		K = len(self.InputNeuron)

		InputValues = np.array([self.InputNeuron[N].CurrentValue for N in range(K)])

		self.CurrentValue = np.sum(InputValues*self.Weights)/(K)

		if(self.CurrentValue<0):
			self.CurrentValue = 0

		if(CopyWantedValue):
			self.WantedValue = self.CurrentValue

	def LoadConfig(self):

		if(self.PathToConfigFile!=None):
			if(os.path.isfile(self.PathToConfigFile)):
				with open(self.PathToConfigFile, "rb") as fp:
					self.Weights = np.load(fp)

		else:
			self.Weights = np.array([random.random() for x in range(len(self.InputNeuron))])

		self.DeltaWeights = self.Weights * 0

	def SaltWeights(self):

		self.Weights += np.array([random.random()*random.randint(-1,1) for x in range(len(self.InputNeuron))])

	def RandomiseWeights(self):

		self.Weights = np.array([random.random() for x in range(len(self.InputNeuron))])

	def SaveConfig(self):

		if(self.PathToConfigFile!=None):
			with open(self.PathToConfigFile, "wb") as fp:
				np.save(fp, self.Weights)

	def PropagateGradient(self, Ponderation):

		Delta = 0.01
		LAMBDA = 0.6

		ErrorConstant = abs(self.WantedValue - self.CurrentValue)

		for W in range(len(self.Weights)):

			self.Weights[W] += Delta

			self.LatchOn(False)

			self.DeltaWeights[W] += ((abs(self.WantedValue - self.CurrentValue) - ErrorConstant)/Delta)*LAMBDA*Ponderation

			self.InputNeuron[W].WantedValue += self.DeltaWeights[W]/self.MyLayerSize

			self.Weights[W] -= Delta

			self.LatchOn(False)

class Network:

	def __init__(self, PathToConfigFile=None):

		self.NeuronPool=[]
		self.InputLayer=[]
		self.HiddenLayer1=[]
		self.HiddenLayer2=[]
		self.OutputLayer=[]

		self.TrainingProgress=0

		print("Path",PathToConfigFile)

		if(PathToConfigFile!=None):
			with open(PathToConfigFile, "r") as fp:
				ListOfCFG = fp.read()

		ListOfCFG = ListOfCFG.split("\n")


		S=0
		for x in range(400):
			self.InputLayer.append(Neuron(400, "cfg/"+ListOfCFG[S+x]))
			self.NeuronPool.append(self.InputLayer[x])

		S+=x+1
		for x in range(40):
			self.HiddenLayer1.append(Neuron(40, "cfg/"+ListOfCFG[S+x], self.InputLayer))
			self.NeuronPool.append(self.HiddenLayer1[x])

		S+=x+1
		for x in range(10):
			self.HiddenLayer2.append(Neuron(10, "cfg/"+ListOfCFG[S+x], self.HiddenLayer1))
			self.NeuronPool.append(self.HiddenLayer2[x])

		S+=x+1
		for x in range(2):
			self.OutputLayer.append(Neuron(2, "cfg/"+ListOfCFG[S+x], self.HiddenLayer2))
			self.NeuronPool.append(self.OutputLayer[x])

		S+=x

	def RandomWeights(self):
		for Neuron in self.NeuronPool:
			Neuron.RandomiseWeights()

	def Salt(self):

		for Neuron in self.NeuronPool:
			Neuron.SaltWeights()

	def Save(self):

		for Neuron in self.NeuronPool:
			Neuron.SaveConfig()

	def SetImage(self, Array):

		for x in range(400):
			self.InputLayer[x].CurrentValue = Array[x]

	def ComputeImage(self):

		for x in range(40):
			self.HiddenLayer1[x].LatchOn()

		for x in range(10):
			self.HiddenLayer2[x].LatchOn()

		for x in range(2):
			self.OutputLayer[x].LatchOn()

		return np.array((self.OutputLayer[0].CurrentValue, self.OutputLayer[1].CurrentValue))/(self.OutputLayer[0].CurrentValue+self.OutputLayer[1].CurrentValue)

	def OutputDeltaWeights(self, Path, N):

		DeltaWeightsArray = []

		for Neuron in self.NeuronPool:

			DeltaWeightsArray.append(Neuron.DeltaWeights.tolist())
			if(N==1 and len(Neuron.DeltaWeights)==40):
				print(Neuron.CurrentValue)
				print(Neuron.DeltaWeights)
				time.sleep(10)

		with open(Path, "wb") as fp:
			pickle.dump(DeltaWeightsArray, fp)

	def Train(self, DataSet, ThreadNumber, OutputPath):

		Delta = 0.001

		Progress = 0

		K = 1/len(DataSet)
		K1 = 1/K

		for Exemple in DataSet:

			self.SetImage(Exemple.MTX)
			self.ComputeImage()

			for x in range(len(self.OutputLayer)):

				self.OutputLayer[x].WantedValue = Exemple.WantedOuput[x]
				self.OutputLayer[x].PropagateGradient(1/len(self.OutputLayer))

				#print(str(ThreadNumber)+" - "+str(round(Progress/(52*K1)*100,2)))
				Progress+=1

			for x in range(len(self.HiddenLayer2)):

				self.HiddenLayer2[x].PropagateGradient(1/len(self.HiddenLayer2))

				#print(str(ThreadNumber)+" - "+str(round(Progress/(52*K1)*100,2)))
				Progress+=1

			for x in range(len(self.HiddenLayer1)):

				self.HiddenLayer1[x].PropagateGradient(1/len(self.HiddenLayer1))

				#print(str(ThreadNumber)+" - "+str(round(Progress/(52*K1)*100,2)))
				Progress+=1

		self.OutputDeltaWeights(OutputPath, ThreadNumber)

	def ShowWeights(self):

		for Neuron in self.NeuronPool:

			print(Neuron.Weights)

	def ShowState(self):

		print([round(N.CurrentValue,4) for N in self.InputLayer])
		print("\n")
		print([round(N.CurrentValue,4) for N in self.HiddenLayer1])
		print("\n")
		print([round(N.CurrentValue,4) for N in self.HiddenLayer2])
		print("\n")
		print([round(N.CurrentValue,4) for N in self.OutputLayer])
		print("\n")

def Test(Network, Image):

	Network.SetImage(Image.MTX)
	# print(Network.ComputeImage(), Image.WantedOuput)
	return Error(Network.ComputeImage(), Image.WantedOuput)


if __name__ == '__main__':

	t0 = time.time()

	Reseau_principal = Network("cfg/main.cfg")

	#Reseau_principal.RandomWeights()

	#Reseau_principal.Save()

	TrainingDataSet = DataSet("data/positif","data/negatif")

	DeltaWeightsOutputPath = ["tmp/1.tmp", "tmp/2.tmp", "tmp/3.tmp", "tmp/4.tmp"]

	DataSets = TrainingDataSet.Send_N_DataSets(4, 8)

	TestArray = TrainingDataSet.Send_All() 

	MeanErrorBefore = np.mean([Test(Reseau_principal, TestArray[x]) for x in range(len(TestArray))])

	print(MeanErrorBefore)

	#Reseau_principal.ShowWeights()

	T1 = Process(target=Reseau_principal.Train, args=(DataSets[0], 1, DeltaWeightsOutputPath[0],))
	T2 = Process(target=Reseau_principal.Train, args=(DataSets[1], 2, DeltaWeightsOutputPath[1],))
	T3 = Process(target=Reseau_principal.Train, args=(DataSets[2], 3, DeltaWeightsOutputPath[2],))
	T4 = Process(target=Reseau_principal.Train, args=(DataSets[3], 4, DeltaWeightsOutputPath[3],))

	T1.start()
	T2.start()
	T3.start()
	T4.start()

	T1.join()
	T2.join()
	T3.join()
	T4.join()

	DeltaWeightsArray=[]

	for Path in DeltaWeightsOutputPath:

		with open(Path, "rb") as fp:
				
			DeltaWeightsArray.append(pickle.load(fp))

	for List in DeltaWeightsArray:
		print(np.sum([x for vector in List for x in vector]))


	for x in range(len(Reseau_principal.NeuronPool)):

		Reset = True
		DeltaAccu = None

		'''print(DeltaWeightsArray[0][x])
		print(DeltaWeightsArray[1][x])
		print(DeltaWeightsArray[2][x])
		print(DeltaWeightsArray[3][x])
		input()'''

		for y in range(len(DeltaWeightsArray)):

			if(Reset):
				DeltaAccu = np.array(DeltaWeightsArray[y][x])/len(DataSets)
				Reset = False
				print(np.array(DeltaWeightsArray[y][x])/len(DataSets))

			else:
				DeltaAccu += np.array(DeltaWeightsArray[y][x])/len(DataSets)
				print(np.array(DeltaWeightsArray[y][x])/len(DataSets))

			
		print("\n")
		print(len(DataSets))
		print(DeltaAccu)
		print("\n")

		input()

		Reseau_principal.NeuronPool[x].Weights += DeltaAccu

	Reseau_principal.Save()
	print(round(time.time()-t0),"s")

	MeanErrorAfter = np.mean([Test(Reseau_principal, TestArray[x]) for x in range(len(TestArray))])

	print(MeanErrorBefore, MeanErrorAfter)