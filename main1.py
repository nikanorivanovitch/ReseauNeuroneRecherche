import PIL
import random
import os
from multiprocessing import Process, Pipe
import numpy as np
import time
import pickle

from PIL import Image

def SignOf(x):
	if(x<0):
		return -1
	else:
		return 1

def Error(Vector1, Vector2):
	return np.sum(np.abs(Vector1-Vector2))

#Erreur quadratique
def MSE(Vector1, Vector2):
	return np.sum((Vector1-Vector2)**2)

def Validate(Vector1, Vector2):
	if((Vector1[0]>Vector1[1] and Vector2[0]>Vector2[1]) or (Vector1[0]<Vector1[1] and Vector2[0]<Vector2[1])):
		return 1
	else:
		return 0

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

	def Send_N_DataSets(self, N_DataSets):

		DataSetsList = []

		DataToInsert = self.Send_All()

		K1 = len(DataToInsert)
		K2 = int(K1/N_DataSets)

		for N in range(N_DataSets):

			DataSetsList.append(DataToInsert[N*K2:(N+1)*K2])

		return DataSetsList

	def Send_NK_DataSets(self, N_DataSets, K_Images):

		DataSetsList = []

		DataToInsert = self.Send_All()

		for x in range(N_DataSets):

			DataSet = []

			for y in range(K_Images):

				A = random.randint(0, len(DataToInsert)-1)

				DataSet.append(DataToInsert[A])
				DataToInsert.pop(A)

			DataSetsList.append(DataSet)

		return DataSetsList

	def Send_All(self):

		R = []

		for P in self.PositiveExemples:

			R.append(P)

		for N in self.NegativeExemples:

			R.append(N)

		random.shuffle(R)

		return R

class Face:

	def __init__(self, PathToFile, Positive):

		#On ouvre l'image en nuance de gris et on redimensionne à 20*20
		self.IMG = Image.open(PathToFile).convert('L').resize((20,20))
		self.MTX = (np.array(self.IMG)-np.mean(self.IMG))/np.sqrt(np.var(self.IMG)*0.5)
		self.MTX = [x for vector in self.MTX for x in vector]
		self.FileName = PathToFile

		if(Positive):
			self.WantedOuput=np.array([1,0])

		else:
			self.WantedOuput=np.array([0,1])

	def Print(self):
		print(self.MTX)

	def Show(self):
		self.IMG.show()

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

		self.HasToBeRecomputed = False

		self.LinkTo(NeuronArray)

	def LinkTo(self, NeuronArray):

		if(NeuronArray!=None):
			for N in NeuronArray:
				self.InputNeuron.append(N)

		self.LoadConfig()

	#Fonction d'agrégation et d'activation du neurone (référence à la bascule verrou D)
	def LatchOn(self, CheckBeforeCompute=False):

		if(CheckBeforeCompute):
			if(not self.HasToBeRecomputed):
				return 0

		K = len(self.InputNeuron)

		self.InputValues = np.array([self.InputNeuron[N].CurrentValue for N in range(K)])

		#On multiplie les entrées par les poids
		self.CurrentValue = (np.sum(self.InputValues*self.Weights))

		#On active la fonction Leaky ReLU
		if(self.CurrentValue<0):
			self.CurrentValue = self.CurrentValue*0.5

	#Chargement des poids du neurone
	def LoadConfig(self):

		if(self.PathToConfigFile!=None):
			if(os.path.isfile(self.PathToConfigFile)):
				with open(self.PathToConfigFile, "rb") as fp:
					self.Weights = np.load(fp)

		else:
			self.Weights = np.array([random.random() for x in range(len(self.InputNeuron))])

		self.DeltaWeights = self.Weights * 0

	def NormaliseWeights(self):
		self.Weights = self.Weights / np.sum(np.abs(self.Weights))

	def UniformiseWeights(self):
		N=len(self.InputNeuron)
		self.Weights = np.array([(1/N)*random.randint(-3,3) for x in range(N)])

	def RandomiseWeights(self):

		self.Weights = np.array([random.random()*random.randint(-1, 1) for x in range(len(self.InputNeuron))])
		self.Weights = self.Weights / np.sum(np.abs(self.Weights))

	#Sauvegarde des poids du neurone
	def SaveConfig(self):

		if(self.PathToConfigFile!=None):
			with open(self.PathToConfigFile, "wb") as fp:
				np.save(fp, self.Weights)

	#Fonction d'entraînement
	def FindBestWeights(self, Network, DataSet, Level):

		Delta = 0.01
		Lambda = 0.0001

		self.HasToBeRecomputed=True

		#Pour chaque Image
		for Exemple in DataSet:

			Network.SetImage(Exemple.MTX)
			Network.ComputeImage()

			#Pour chaque Poids du neurone
			for W in range(len(self.Weights)):

				K1 = Network.ComputeImage(StartAtLevel=Level, Optimise=True)
				K1 = MSE(K1, Exemple.WantedOuput)

				self.Weights[W]+=Delta

				K2 = Network.ComputeImage(StartAtLevel=Level, Optimise=True)
				K2 = MSE(K2, Exemple.WantedOuput)

				self.Weights[W]-=Delta

				#On enlève la dérivée de l'erreur de cette image par le poids, à ce poids
				self.DeltaWeights[W] -= ((K2-K1)/Delta)*Lambda
				#Les changements prennent effet à la fin de la séquence d'entraînement.

		self.HasToBeRecomputed=False

class Network:

	def __init__(self, PathToConfigFile=None):

		self.NeuronPool=[]
		self.InputLayer=[]   #Première couche
		self.HiddenLayer1=[] #Deuxième couche
		self.HiddenLayer2=[] #Troisième couche
		self.OutputLayer=[]  #Quatrième couche

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

	def UniformWeights(self):
		for Neuron in self.NeuronPool:
			Neuron.UniformiseWeights()

	def Salt(self):

		for Neuron in self.NeuronPool:
			Neuron.SaltWeights()

	def Save(self):

		for Neuron in self.NeuronPool:
			Neuron.SaveConfig()

	def SetImage(self, Array):

		#Chargement de l'image dans le neurone
		for x in range(400):
			self.InputLayer[x].CurrentValue = Array[x]

	#Fonction qui fait la propagation en avant (calcul des sorties)
	def ComputeImage(self, Optimise=False, StartAtLevel=0):

		#La variable StartAtLevel premet de sauter le calcul d'un étage s'il est inutile (permet de gargner du temps pendant l'entraînement)

		if(StartAtLevel==0):
			for x in range(40):
				self.HiddenLayer1[x].LatchOn()

			for x in range(10):
				self.HiddenLayer2[x].LatchOn()

			for x in range(2):
				self.OutputLayer[x].LatchOn()

		if(StartAtLevel==1):
			for x in range(40):
				self.HiddenLayer1[x].LatchOn(Optimise)

			for x in range(10):
				self.HiddenLayer2[x].LatchOn()

			for x in range(2):
				self.OutputLayer[x].LatchOn()

		if(StartAtLevel==2):
			for x in range(10):
				self.HiddenLayer2[x].LatchOn(Optimise)

			for x in range(2):
				self.OutputLayer[x].LatchOn()

		if(StartAtLevel==3):
			for x in range(2):
				self.OutputLayer[x].LatchOn(Optimise)

		return np.array((self.OutputLayer[0].CurrentValue, self.OutputLayer[1].CurrentValue))

	#Enregistre les modifications à faire sur les poids (le partage de mémoire étant complexe avec multiprocessing) dans un fichier
	def OutputWeights(self, Path, N):

		DeltaWeightsArray = []

		for Neuron in self.NeuronPool:

			DeltaWeightsArray.append(Neuron.DeltaWeights.tolist())

		with open(Path, "wb") as fp:
			pickle.dump(DeltaWeightsArray, fp)

	#Fonction qui fait s'entraîner chaque neurone du réseau
	def Train(self, DataSet, ThreadNumber, OutputPath, Iterations):

		Progress = 0
		K1=1

		#print(self.ComputeImage(), Exemple.WantedOuput)

		for I in range(Iterations):

			for x in range(len(self.OutputLayer)):

				#print("OUTPUT LAYER : ",x,"\n")
				self.OutputLayer[x].FindBestWeights(self, DataSet, 3)

				print(str(ThreadNumber)+" - "+str(round(Progress/(52*K1)*100,2)))
				Progress+=1

			for x in range(len(self.HiddenLayer2)):

				#print("HIDDEN LAYER 2 : ",x,"\n")
				self.HiddenLayer2[x].FindBestWeights(self, DataSet, 2)

				print(str(ThreadNumber)+" - "+str(round(Progress/(52*K1)*100,2)))
				Progress+=1

			for x in range(len(self.HiddenLayer1)):

				#print("HIDDEN LAYER 1 : ",x,"\n")
				self.HiddenLayer1[x].FindBestWeights(self, DataSet, 1)

				print(str(ThreadNumber)+" - "+str(round(Progress/(52*K1)*100,2)))
				Progress+=1

			print("PROGRESS : ",round(100*((I+1)/Iterations),2),"%")

		self.OutputWeights(OutputPath, ThreadNumber)

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

	def NormaliseWeights(self):

		for N in self.NeuronPool:
			N.NormaliseWeights()


# Fonctions d'évaluation (pourcentage de bonne détection et erreur quadratique moyenne)

def Test(Network, Image):

	Network.SetImage(Image.MTX)
	#print(Network.ComputeImage(), Image.WantedOuput)
	return MSE(Network.ComputeImage(), Image.WantedOuput)

def Pourcentage(Network, Image):

	Network.SetImage(Image.MTX)
	
	A = Validate(Network.ComputeImage(), Image.WantedOuput)

	if(A==0):
		1
		#Image.Show()

	return A


if __name__ == '__main__':

	t0 = time.time()

	#On créé le réseau qui va créer tous les neurones avec les poids stockés dans la liste de fichiers main.cfg
	Reseau_principal = Network("cfg/main.cfg")

	#On charge les images
	TrainingDataSet = DataSet("data/positif","data/negatif")

	#Reseau_principal.RandomWeights()
	#Reseau_principal.NormaliseWeights()
	#Reseau_principal.Save()

	#DeltaWeightsOutputPath = ["tmp/1.tmp"]

	#Liste des ficheirs par lesquels on retourne les poids
	DeltaWeightsOutputPath = ["tmp/1.tmp", "tmp/2.tmp", "tmp/3.tmp", "tmp/4.tmp"]

	#On demande d'envoyer 4 listes de 30 images (tirés sans remise aléatoirement)
	DataSets = TrainingDataSet.Send_NK_DataSets(4, 30)

	#On demande d'envoyer toutes les images (pour faire calculer l'erreur moyenne)
	TestArray = TrainingDataSet.Send_All()

	t0 = time.time()
	
	MeanErrorBefore = np.mean([Test(Reseau_principal, TestArray[x]) for x in range(len(TestArray))])
	PercentBefore = np.mean([Pourcentage(Reseau_principal, TestArray[x]) for x in range(len(TestArray))])

	print(MeanErrorBefore)
	print(PercentBefore)
	
	#On envoie 4 threads avec une copie du réseau dans son état actuel, un jeu de photo et un fichier par lequel retourner les modifications à effectuer
	T1 = Process(target=Reseau_principal.Train, args=(DataSets[0], 1, DeltaWeightsOutputPath[0], 1,))
	T2 = Process(target=Reseau_principal.Train, args=(DataSets[1], 2, DeltaWeightsOutputPath[1], 1,))
	T3 = Process(target=Reseau_principal.Train, args=(DataSets[2], 3, DeltaWeightsOutputPath[2], 1,))
	T4 = Process(target=Reseau_principal.Train, args=(DataSets[3], 4, DeltaWeightsOutputPath[3], 1,))

	#On les lance
	T1.start()
	T2.start()
	T3.start()
	T4.start()

	#On en attend la fin
	T1.join()
	T2.join()
	T3.join()
	T4.join()

	
	DeltaWeightsArray=[]

	#On ouvre les fichiers de retour et on en récupère le contenu, pickle permet de stocker des listes en binaire
	for Path in DeltaWeightsOutputPath:

		with open(Path, "rb") as fp:
				
			DeltaWeightsArray.append(pickle.load(fp))

	#Pour chaque neurone
	for x in range(len(Reseau_principal.NeuronPool)):

		Reset = True
		DeltaAccu = []

		#Pour chaque retour de processus
		for y in range(len(DeltaWeightsArray)):

			#On récupère la valeur des changements de poids
			if(Reset):
				DeltaAccu = np.array(DeltaWeightsArray[y][x])
				Reset = False
				#print(np.array(DeltaWeightsArray[y][x])/len(DataSets))

			else:
				DeltaAccu += np.array(DeltaWeightsArray[y][x])
				#print(np.array(DeltaWeightsArray[y][x])/len(DataSets))

		print(DeltaAccu)

		#Et on effectue ces changements de poids
		Reseau_principal.NeuronPool[x].Weights += DeltaAccu

	print("DeltaAccu")

	#On sauvegarde les modifications
	Reseau_principal.Save()
	print(round(time.time()-t0),"s")
	
	MeanErrorAfter = np.mean([Test(Reseau_principal, TestArray[x]) for x in range(len(TestArray))])
	PercentAfter = np.mean([Pourcentage(Reseau_principal, TestArray[x]) for x in range(len(TestArray))])

	#Reseau_principal.ShowState()

	print(MeanErrorBefore, MeanErrorAfter)
	print(PercentBefore, PercentAfter)