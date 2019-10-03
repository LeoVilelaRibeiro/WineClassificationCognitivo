# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 15:05:01 2019

@author: Leonardo Vilela Ribeiro
Este código executa a chamada de 3 algoritmos para tratar o problema da indicação de vinhos, e ao fim 
imprime um gráfico avaliando cada um deles. As classes que são chamadas estão implementadas no mesmo 
diretório> KNN.py, SVM.py e naivebayes.py
Este é um teste do framework utilizando o arquivo de vinhos com dimensionalidade reduzida por PCA
chamado "PCAwinw.csv", foi feito apenas para avaliar o comportamento do código porém os resultados
são inferiores
"""

import os
os.chdir(r'C:\cognitivo')

from knn import KNN
from svm import SVM
from naivebayes import NaiveBayes
import numpy as np
from tqdm import tqdm
#Método para cálculo da acurácia a partir da matriz de confusão
def getAccuracy(confusionMatrix):
    numerador=confusionMatrix[0][0] + confusionMatrix[1][1]
    denominador=confusionMatrix[0][0] + confusionMatrix[1][0] + confusionMatrix[0][1] + confusionMatrix[1][1]
    if (denominador == 0):
        accuracy = 0
    else:
        accuracy = (numerador/denominador)*100
    return accuracy 
#intancia arrays de resultado de acuracia e funções de custo para ao final apresentar a média
#das mesmas em n iterações

knnArray = []
maelossArray = []
mselossArray = []
#Executa o Knn por 200 iterações e exibe as métricas
for i in tqdm(range(0, 200)):
    cmKnn,maeloss,mseloss = KNN.computeExample("PCAwinw.csv")
    knnArray.append(getAccuracy(cmKnn))
    maelossArray.append(maeloss)
    mselossArray.append(mseloss)
print("\nMédia do KNN: %.2f" % np.mean(knnArray))
print("Desvio Padrão do KNN: %.2f" % np.std(knnArray))
print("Perda MAE do KNN: %.2F" % np.mean(maelossArray))
print("Perda MSE do KNN: %.2F" % np.mean(mselossArray))
#intancia arrays de resultado de acuracia e funções de custo para ao final apresentar a média
#das mesmas em n iterações

svmGaussArray = []
maelossArray = []
mselossArray = []
#Executa o SVM por 200 iterações e exibe as métricas
for i in tqdm(range(0, 200)):
    cmSVMG,maeloss,mseloss = SVM.computeExample("PCAwinw.csv", "rbf", 0)
    svmGaussArray.append(getAccuracy(cmSVMG))
    maelossArray.append(maeloss)
    mselossArray.append(mseloss)
print("\nMédia do SVM Gaussiano: %.2f" % np.mean(svmGaussArray))
print("Desvio Padrão do SVM Gaussiano: %.2f" % np.std(svmGaussArray))
print("Perda MAE do SVM: %.2F" % np.mean(maelossArray))
print("Perda MSE do SVM: %.2F" % np.mean(mselossArray))
#intancia arrays de resultado de acuracia e funções de custo para ao final apresentar a média
#das mesmas em n iterações
svmNBArray = []
maelossArray = []
mselossArray = []
#Executa o Naive Bayes por 200 iterações e exibe as métricas
for i in tqdm(range(0, 200)):
    cmNB,maeloss,mseloss = NaiveBayes.computeExample("PCAwinw.csv")
    svmNBArray.append(getAccuracy(cmNB))
    maelossArray.append(maeloss)
    mselossArray.append(mseloss)
print("\nMédia do NB: %.2f" % np.mean(svmNBArray))
print("Desvio Padrão do NB: %.2f" % np.std(svmNBArray))
print("Perda MAE do NB: %.2F" % np.mean(maelossArray))
print("Perda MSE do NB: %.2F" % np.mean(mselossArray))

#plota um gráfico comparativo dos resultados de acurácia

import matplotlib.pyplot as plt
plt.plot(knnArray, 'g--', svmGaussArray, 'b^',svmNBArray,'r^')
plt.ylabel("Acurácia")
plt.xlabel("Tentativas")
plt.show()