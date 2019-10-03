from classification import ClassificationModel
#Classe Criada para instanciar um classificador de Naive Baies
class NaiveBayes(ClassificationModel):
    #o método compute model instancia o classificador recebendo os dados de treino (XTrain, yTrain)
    def computeModel(XTrain, yTrain):
        from sklearn.naive_bayes import GaussianNB

        classifier = GaussianNB()
        classifier.fit(XTrain[0], yTrain)

        return classifier
    # o método compute example executa o pré processamento, cria o clasificador, faz a predição e retorna
    #uma matriz de confusão criada pelo evaluateModel
    #este método chama na clase importada os métodos de calculo da função de perda(custo) mae e mse
    def computeExample(filename):
        XTrain, XTest, yTrain, yTest = ClassificationModel.preprocessData(filename)

        classifier = NaiveBayes.computeModel(XTrain, yTrain)
        yPred = ClassificationModel.predictModel(classifier, XTest)
        return ClassificationModel.evaluateModel(yPred, yTest),ClassificationModel.mae_loss(yPred,yTest),ClassificationModel.mse_loss(yPred,yTest)


