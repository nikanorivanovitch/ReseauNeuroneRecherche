from multiprocessing import Process
def g(x):
	x.x+=1

def f(x):
	g(x)

class neuneu:

	def __init__(self, X):
		self.x = X

if __name__ == '__main__':
	
	I=neuneu(2)
	T1 = Process(target=f, args=(I,))

	T1.start()
	T1.join()

	print(I.x)