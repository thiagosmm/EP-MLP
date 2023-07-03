# - ALGORITMO MLP -
# - problema de identificação de caracteres -
#       63 neuronios na camada de entrada
#       10 neuronios na camada escondida
#       7 neuronios na camada de saida
#       função de ativaçao: tangente hiberbolica

import random
import math
import numpy as np
import pandas as pd

def activation_function(t):
    return (np.exp(2*t)-1)/(np.exp(2*t)+1)

def derivada_activation(f):
    return (1/np.cosh(f)**2)

def mlp_architecture(input_length, hidden_length, output_length):
    weights_hidden = np.random.uniform(-0.5, 0.5, size=(hidden_length, input_length))
    weights_output = np.random.uniform(-0.5, 0.5, size=(output_length, hidden_length))

    return weights_hidden, weights_output

def mlp_forward(input, hidden, output, weights_hidden, weights_output):
    hidden = np.dot(weights_hidden, input)
    hiddenActivation = activation_function(hidden)

    output = np.dot(weights_output, hiddenActivation)
    outputFinal = activation_function(output)

    return hidden, output, hiddenActivation, outputFinal, weights_hidden, weights_output

def mlp_backpropagation(input, output, outputFinal, target, hidden, hiddenActivation, weights_hidden, weights_output, learningRate):
    deltaOutput = (target - output) * derivada_activation(output)
    deltaHidden = np.dot(deltaOutput, weights_output) * derivada_activation(hidden)

    weights_output += learningRate * np.outer(deltaOutput, hiddenActivation)
    weights_hidden += learningRate * np.outer(deltaHidden, input)

    return output, outputFinal, hidden, weights_hidden, weights_output

def extracaoDataTreino():
    df1 = pd.read_csv('caracteres-ruido.csv', header=None)
    inputs = df1.iloc[ :, :63].values
    targets = df1.iloc[ :, -7:].values
    df2 = pd.read_csv('caracteres_ruido20.csv', header=None)
    inputs = np.vstack([inputs, df2.iloc[:, :63].values])
    targets = np.vstack([targets, df2.iloc[:, -7:].values])

    return inputs, targets

def extracaoDataTeste():
    df1 = pd.read_csv('caracteres-limpo.csv', header=None)
    inputsTeste = df1.iloc[:, :63].values
    targetsTeste = df1.iloc[:, -7:].values

    return inputsTeste, targetsTeste

def extracaoPortaLogica():
    df1 = pd.read_csv('problemAND.csv', header=None)
    inputs = df1.iloc[:4, :2].values
    targets = df1.iloc[:4, -1:].values
    df2 = pd.read_csv('problemOR.csv', header=None)
    inputs = np.vstack([inputs, df2.iloc[:4, :2]])
    targets = np.vstack([targets, df2.loc[:4, -1:]])
    df3 = pd.read_csv('problemXOR.csv', header=None)
    inputs = np.vstack([inputs, df3.iloc[:4, :2]])
    targets = np.vstack([targets, df3.loc[:4, -1:]])

    df4 = pd.read_csv('problemAND.csv', header=None)
    inputsTeste = df4.iloc[:4, :2].values
    targetsTeste = df4.iloc[:4, -1:].values
    df5 = pd.read_csv('problemOR.csv', header=None)
    inputsTeste = np.vstack([inputsTeste, df5.iloc[:4, :2]])
    targetsTeste = np.vstack([targetsTeste, df5.loc[:4, -1:]])
    df6 = pd.read_csv('problemXOR.csv', header=None)
    inputsTeste = np.vstack([inputsTeste, df6.iloc[:4, :2]])
    targetsTeste = np.vstack([targetsTeste, df6.loc[:4, -1:]])

    return inputs, targets, inputsTeste, targetsTeste

def main():
    inputs, targets = extracaoDataTreino()
    inputsTeste, targetsTeste = extracaoDataTeste()
    #inputs, targets, inputsTeste, targetsTeste = extracaoPortaLogica()

    learningRate = 0.2
    epochs = 10000
    maxError = 0.001
    input_length = 63
    hidden_length = 30
    output_length = 7
    hidden = np.zeros(hidden_length)
    output = np.zeros(output_length)
    weights_hidden, weights_output = mlp_architecture(input_length, hidden_length, output_length)

# TREINAMENTO - arquivos .csv que contém ruído

    for j in range(epochs):
        error = 0
        for i in range(len(inputs)):
            input = inputs[i]
            target = targets[i]
            hidden, output, hiddenActivation, outputFinal, weights_hidden, weights_output = mlp_forward(input, hidden, output, weights_hidden, weights_output)
            output, outputFinal, hidden, weights_hidden, weights_output = mlp_backpropagation(input, output, outputFinal, target, hidden, hiddenActivation, weights_hidden, weights_output, learningRate)

            error += 1/len(inputs) * (np.sum(target - outputFinal))**2
            print('época: ', j, 'input: ',i, 'valor do erro: ', error)

            if error <= maxError:
                print('parada devido ao valor do erro --- erro: ', error)
                break
        print('ERRO QUADRÁTICO MÉDIO da epoca ', j,': ', error)
    print('fim das épocas')

# -------------- TESTES --------------------
# print dos valores finais dos testes
    rightAns = 0
    for i in range(len(inputsTeste)):
        input = inputsTeste[i]
        target = targetsTeste[i]
        hidden = np.zeros(hidden_length)
        output = np.zeros(output_length)
        hidden, output, hiddenActivation, outputFinal, wsHidden, wsOutput = mlp_forward(input, hidden, output, weights_hidden, weights_output)
        print('Entrada:', input)
        print('Target:', target)
        print('Saída:', outputFinal)
        for i in range(len(target)):
            if (target[i] > 0) and (outputFinal[i] > 0):
                print('OUTPUT CORRETO')
                rightAns += 1
                break
    print(rightAns,'/',len(inputsTeste))

main()
