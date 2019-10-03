import preprocessing as pre
import numpy as np
#Esta classe foi criada com o objetivo de oferecer métodos auxiliares no trabalho, 
#genéricos a qualquer um dos algoritmos avaliados neste trabalho
class ClassificationModel:
    def __init__(self):
        pass    
    #recebe um classificador e realiza a predição
    def predictModel(classifier, X):
        return classifier.predict(X[0])
    #recebe o resultado de uma predição e cria a matriz de confusão
    def evaluateModel(yPred, yTest):
        from sklearn.metrics import confusion_matrix
        confusionMatrix = confusion_matrix(yTest, yPred)

        return confusionMatrix
    #chama o pre-processamento , cria base de treino e teste e faz normalização
    #neste caso estamos usando 15% para teste e o restante para treino.
    def preprocessData(filename):
        X, y, csv = pre.loadDataset(filename, ",")
        XTrain, XTest, yTrain, yTest = pre.splitTrainTestSets(X, y, 0.15)
        XTrain = pre.computeScaling(XTrain)
        XTest = pre.computeScaling(XTest)

        return XTrain, XTest, yTrain, yTest
    # calcula o custo através da métrica mae
    def mae_loss(y_pred, y_true):
        abs_error = np.abs(y_pred - y_true)
        sum_abs_error = np.sum(abs_error)
        loss = sum_abs_error / y_true.size
        return loss
    #calcula o custo através da métrica mse
    def mse_loss(y_pred, y_true):
        squared_error = (y_pred - y_true) ** 2
        sum_squared_error = np.sum(squared_error)
        loss = sum_squared_error / y_true.size
        return loss
