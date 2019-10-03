# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 13:14:51 2019

@author: Leonardo Vilela Ribeiro
Este código foi criado com o objetivo de instanciar o Classificador 
KNN, que foi o de melhor resultado entre os testados, para simular 
uma classificação em código para produção

"""
import os
os.chdir(r'C:\cognitivo')   
from classification import ClassificationModel
from knn import KNN
import pandas as pd
import preprocessing as pre
#Recebe o arquivo instanciando o Classificador    
classifier=KNN.createClassifier("winecorreto.csv")
#recebe um imput com os dados de vinho, neste caso como é uma simulação 
#eu peguei um pedaço pequeno do teste, por isso estou retirando o Y dele
#em um cenário de produção a entrada logicamente virá sem o Y
DataInput= pd.read_csv('input.csv', delimiter=',')
X = DataInput.iloc[:,:-1].values
#normaliza
X=pre.computeScaling(X)
#prediz
y = ClassificationModel.predictModel(classifier, X)
#impressão dos resultados da predição
print( DataInput['quality'])
print(y)