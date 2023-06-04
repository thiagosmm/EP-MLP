# - ALGORITMO MLP -
# - problema de identificação de caracteres -
#       63 neuronios na camada de entrada
#       7 neuronios na camada escondida
#       7 neuronios na camada de saida
#       função de ativaçao: sigmoid

# arrumar testes - iteração sobre épocas

import random
import math
import numpy as np
#import pandas as pd

def activation_function(t):
    return (1/(1 + np.exp(-t)))

def derivada_activation(f):
    return (f * (1 - f))

def mlp_architecture(input_length, hidden_length, output_length):
    weights_hidden = [[round(random.uniform(-50000, 50000) / 100000, 1) for _ in range(input_length)] for _ in range(hidden_length)]
    print(weights_hidden)
    weights_output = [[round(random.uniform(-50000, 50000) / 100000, 1) for _ in range(hidden_length)] for _ in range(output_length)]
    print('\nPESOS HIDDEN->OUTPUT', weights_output)

    return weights_hidden, weights_output

def mlp_forward(input, hidden, output, weights_hidden, weights_output):

# combinação linear entradas com pesos para hidden
    for i in range(len(hidden)):
        for j in range(len(input)):
           hidden[i] += input[j] * weights_hidden[i][j]

    hiddenActivation = [0] * len(hidden)
    print('OLHA AQUI', len(hidden), len(hiddenActivation))

# função de ativação nos neuronios da hidden 
    for i in range(len(hidden)):
        hiddenActivation[i] = hidden[i]
        hiddenActivation[i] = activation_function(hiddenActivation[i])

# combinação linear entradas com pesos para output
    for i in range(len(output)):
        for j in range(len(hidden)):
           output[i] += hidden[j] * weights_output[i][j]

    outputFinal = [0] * len(output)

# função de ativação nos neuronios de saída
    for i in range(len(output)):
        outputFinal[i] = output[i]
        outputFinal[i] = activation_function(output[i])

    return hidden, output, hiddenActivation, outputFinal, weights_hidden, weights_output

def mlp_backpropagation(input, output, outputFinal, target, hidden, hiddenActivation, weights_hidden, weights_output, learningRate):
    erroTotal = 0 
    for i in range(len(output)):
        erroTotal += 0.5*((target[i] - output[i])**2)
    #print(erroTotal)

    deltaOutput = [0] * len(output)
    deltaHidden= [0] * len(hidden)

# backpropagation output -> hidden
    for i in range(len(output)):
        deltaOutput[i] = (target[i] - output[i]) * derivada_activation(output[i])
        for j in range(len(hidden)):
            weights_output[i][j] += learningRate * deltaOutput[i] * hiddenActivation[j]
            deltaHidden[i] += deltaOutput[i] * weights_output[i][j] * derivada_activation(hidden[j]) #verificar

#  backpropagation hidden -> input
    for i in range(len(hidden)):
        for j in range(len(input)):
            weights_hidden[i][j] += learningRate * deltaHidden[i] * input[j] 
    
    return output, outputFinal, hidden, weights_hidden, weights_output

def main():
    learningRate = 0.2
    epocs = 10000
    maxError = 0.1
    input_length = 63 
    hidden_length = 10
    output_length = 7
    inputs = [(-1,-1,1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,1,1,1,1,-1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,1,1,1,-1,1,1,1), (), (), (), ()]
    targets = [(), (), (), ()]
    weights_hidden, weights_output = mlp_architecture(input_length, hidden_length, output_length)

    for i in range(epocs):
        input = [-1,-1,1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,1,1,1,1,-1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,1,1,1,-1,1,1,1]
        target = [1,-1,-1,-1,-1,-1,-1]
        hidden = [0] * hidden_length
        hiddenActivation = [] * hidden_length
        output = [0] * output_length
        outputFinal = [] * output_length

        hidden, output, hiddenActivation, outputFinal, weights_hidden, weights_output = mlp_forward(input, hidden, output, weights_hidden, weights_output)
        output, outputFinal, hidden, weights_hidden, weights_output = mlp_backpropagation(input, output, outputFinal, target, hidden, hiddenActivation, weights_hidden, weights_output, learningRate)
        print('época numero: ', i, '\noutput: ', output)
        
        erro = 0 
        for i in range(len(outputFinal)):
            erro += 0.5*((target[i] - outputFinal[i])**2)
        print(' erro: ', erro)
        if erro < maxError: print('parada por erro'); break
    
    print('FIM DAS EPOCAS')


    #print(weights_hidden, '\n', weights_output)

    #print(random.uniform(-1, 1))


main()
