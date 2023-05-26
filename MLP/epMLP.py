# - ALGORITMO MLP -
# - problema de identificação de caracteres -
#       63 neuronios na camada de entrada
#       uma camada escondida
#       7 neuronios na camada de saida

import random
import math
#import numpy as np
#import pandas as pd

def activation_function(t):
    return (1 - math.exp(-t)/(1 + math.exp(-t)))

def derivada_activation(f):
    return (f * (1 - f))

def mlp_architecture(input_lenght, hidden_lenght, output_lenght):
    weights_hidden = [[random.uniform(-0.5, 0.5) for _ in range(hidden_lenght)] for _ in range(input_lenght)]
    weights_output = [[random.uniform(-0.5, 0.5) for _ in range(output_lenght)] for _ in range(hidden_lenght)]

    return weights_hidden, weights_output

def mlp_forward(input, hidden, output, weights_hidden, weights_output):

# combinação linear entradas com pesos para hidden
    for i in range(len(input)):
        for j in range(len(hidden)):
           hidden[j] += input[i] * weights_hidden[i][j]

# função de ativação nos neuronios da hidden 
    for i in range(len(hidden)):
        hidden[i] = activation_function(hidden[i])

# combinação linear entradas com pesos para output
    for i in range(len(hidden)):
        for j in range(len(output)):
           output[j] += hidden[i] * weights_output[i][j]

# função de ativação nos neuronios de saída
    for i in range(len(hidden)):
        output[i] = activation_function(output[i])

    return output, weights_hidden, weights_output

def mlp_backpropagation(output, target, weights_hidden, weights_output):
    erro = 0 
    for i in range(len(output)):
        erro += 0.5*((target[i] - output[i])**2)
    print(erro)

def main():
    lRate = 0.1
    epocs = 1000
    maxError = 0.1 
    input = [-1,-1,1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,1,1,1,1,-1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,1,1,1,-1,1,1,1]
    target = [1,-1,-1,-1,-1,-1,-1]
    hidden = [0] * 7
    output = [0] * 7
    weights_hidden, weights_output = mlp_architecture(len(input), len(hidden), len(output))
    output, weights_hidden, weights_output = mlp_forward(input, hidden, output, weights_hidden, weights_output)
    mlp_backpropagation(output, target, weights_hidden, weights_output)

   #print(random.uniform(-1, 1))


main()
