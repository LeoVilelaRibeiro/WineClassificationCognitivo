# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 08:53:43 2019

@author: Leonardo Vilela Ribeiro
Este trabalho foi realizado na tentativa de reduzir a dimensionalidade de um arquivo
contendo dados sobre vinhos, de 12 colunas para 2.
"""
import os
os.chdir(r'C:\cognitivo')
import pandas as pd 
import matplotlib.pyplot as plt

#Com o objetivo de melhorar a massa de dados, 
#já que a correlação não possibilitou nenhum avanço, 
#realizaremos a Análise de componentes principais

# carregar conjunto de dados no Pandas DataFrame 
df = pd.read_csv ("winecorreto.csv", sep=',')

from sklearn.preprocessing import StandardScaler 
features = ['id','type','fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
# # Separando os recursos 
x = df.loc [:, features] .values 
# Separando o target 
y = df.loc [:, ['quality']]. values 
# Padronizando os recursos 
#Para realizar processamentos de dataset com colunas representando
# grandezas diferentes como este, temos de normalizar,
# sendo assim executaremos a normalização antes de extrair componentes principais da nossa base
x = StandardScaler (). fit_transform (x)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, df[['quality']]], axis = 1)
finalDf.to_csv('PCAwinw.csv')

#Reduzindo para duas dimensões, podemos plotar o gráfico
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [3, 4, 5,6,7,8,9]
colors = ['r', 'g', 'b','c','m','y','k']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['quality'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
