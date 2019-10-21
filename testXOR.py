import numpy as np
import matplotlib.pyplot as plt
import NN as nn
x = np.array([[0, 0],
	 [0, 1],
	 [1, 0],
	 [1, 1]])

y = np.array([[0], [1], [1], [0]])

input_neurons = 2
hidden_neurons = 3
output_neurons = 1

net = nn.NN(input_neurons,hidden_neurons,output_neurons)

epochs = 50000

learning_rate = 0.1

costs = []

for i in range(epochs):
	net.foward(x)
	net.backprop(y)
	net.update(learning_rate)

	cost = net.cost()
	costs.append(cost)

	if i % 1000 == 0:
		print(f"Iter: {i}. Error: {cost}")

print("Acabou de treinar")

x_teste = np.array([[1,1], [0,1], [0, 0], [1,0]])

print("\nTeste do treinamento: ")
print(x_teste)

predict = net.foward(x_teste)

print("\nPorcentagem: ")
print(predict)

print("\nPredict")
print(np.round(predict))