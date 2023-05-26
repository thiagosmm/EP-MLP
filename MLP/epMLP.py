# - ALGORITMO MLP -
# - problema de identificação de caracteres -
#       63 neuronios na camada de entrada
#       7 neuronios na camada escondida
#       7 neuronios na camada de saida
#       função de ativaçao: sigmoid

import random
import math
#import numpy as np
#import pandas as pd

def activation_function(t):
    return (1/(1 + math.exp(-t)))

def derivada_activation(f):
    return (f * (1 - f))

def mlp_architecture(input_lenght, hidden_lenght, output_lenght):
    weights_hidden = [[random.uniform(-0.5, 0.5) for _ in range(input_lenght)] for _ in range(hidden_lenght)]
    weights_output = [[random.uniform(-0.5, 0.5) for _ in range(hidden_lenght)] for _ in range(output_lenght)]

    return weights_hidden, weights_output

def mlp_forward(input, hidden, output, weights_hidden, weights_output):

# combinação linear entradas com pesos para hidden
    for i in range(len(input)):
        for j in range(len(hidden)):
           hidden[j] += input[i] * weights_hidden[j][i]

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

    return hidden, output, weights_hidden, weights_output

def mlp_backpropagation(input, output, target, hidden, weights_hidden, weights_output, lRate):
    erroTotal = 0 
    for i in range(len(output)):
        erroTotal += 0.5*((target[i] - output[i])**2)
    #print(erroTotal)

    deltaOutput = [0] * 7
    deltaHidden= [0] * 7

# backpropagation output -> hidden
    for i in range(len(output)):
        deltaOutput[i] = (target[i] - output[i]) * derivada_activation(output[i])
        for j in range(len(hidden)):
            weights_output[i][j] = weights_output[i][j] + lRate * deltaOutput[i] * hidden[j]
            deltaHidden[i] += deltaOutput[i] * weights_output[i][j] * derivada_activation(hidden[i]) #verificar

#  backpropagation hidden -> input
    for i in range(len(input)):
        for j in range(len(hidden)):
            weights_hidden[j][i] = weights_hidden[j][i] * lRate * input[j] 
    
    return output, hidden, weights_hidden, weights_output

def main():
    lRate = 0.1
    epocs = 10000
    maxError = 0.1
    input_lenght = 63 
    hidden_lenght = 7
    output_lenght = 7
    weights_hidden, weights_output = mlp_architecture(input_lenght, hidden_lenght, output_lenght)

    for i in range(epocs):
        input = [-1,-1,1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,1,1,1,1,-1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,1,1,1,-1,1,1,1]
        target = [1,-1,-1,-1,-1,-1,-1]
        hidden = [0] * hidden_lenght
        output = [0] * output_lenght
        hidden, output, weights_hidden, weights_output = mlp_forward(input, hidden, output, weights_hidden, weights_output)
        output, hidden, weights_hidden, weights_output = mlp_backpropagation(input, output, target, hidden, weights_hidden, weights_output, lRate)

    print(weights_hidden, '\n', weights_output)

   #print(random.uniform(-1, 1))


main()
