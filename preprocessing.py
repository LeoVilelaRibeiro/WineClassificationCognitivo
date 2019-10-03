import numpy as np
import pandas as pd
#Essa Classe foi criada com o objetivo de oferecer métodos de pré-processamento
#este método carrega o dataset inteiro e separa em massa de entrada X e saída y
def loadDataset(filename, deli):
 
    baseDeDados = pd.read_csv(filename, delimiter=deli)
    X = baseDeDados.iloc[:,:-1].values
    y = baseDeDados.iloc[:,-1].values
      
    return X, y, baseDeDados
#este método preenche os campos vazios pela média, não está sendo chamado nesta
#implementação pois fiz isso no excel
def fillMissingData(X, inicioColuna, fimColuna):
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    X[:,inicioColuna:fimColuna + 1] = imputer.fit_transform(X[:,inicioColuna:fimColuna + 1])
    return X
#este método identifica categorias na massa,pivota cada categoria em coluna,
#fazendo com que na coluna de cada categoria tenha 1 quando o registro for daquela
#categoria e 0 quando não. Não está sendo chamado nesta implementação pois também
#fiz isso no excel
def computeCategorization(X):
    from sklearn.preprocessing import LabelEncoder
    labelencoder_X = LabelEncoder()
    X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

    #one hot encoding
    D = pd.get_dummies(X[:,0]).values
    
    X = X[:,1:]
    for ii in range(0, D.shape[1]):
        X = np.insert(X, X.shape[1], D[:,ii], axis=1)
    X = X[:,:X.shape[1] - 1]

    return X
#este método faz a separação da massa em treino e teste.

def splitTrainTestSets(X, y, testSize):
    from sklearn.model_selection import train_test_split
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size = testSize)

    return XTrain, XTest, yTrain, yTest
#este método faz a normalização
def computeScaling(X):
    from sklearn.preprocessing import StandardScaler
    scaleobj = StandardScaler()
    X = scaleobj.fit_transform(X.astype(float))

    return X, scaleobj
