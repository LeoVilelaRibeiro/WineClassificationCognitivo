from classification import ClassificationModel
#classe criada para fornecer um objeto Knn e alguns metodos para instanciar

class KNN(ClassificationModel):
    #metodo que computa o modelo recebendo os dados de treino(XTrain, yTrain)
    def computeModel(XTrain, yTrain):
        from sklearn.neighbors import KNeighborsClassifier
        # o classificador é chamado passando o número de vizinhos próximos que serão analisados n_neighbors (5) e
        # e p passando o tipo de distância a ser calculada entre os vizinhos (manhattan_distance (1), and euclidean_distance (2) )
        classifier = KNeighborsClassifier(n_neighbors = 5, p = 2)
        classifier.fit(XTrain[0], yTrain)

        return classifier
    #método criado para receber o arquivo, executar o pré processamento e instanciar o classificador.
    #este método chama na clase importada os métodos de calculo da função de perda(custo) mae e mse
    def computeExample(filename):
        XTrain, XTest, yTrain, yTest = ClassificationModel.preprocessData(filename)

        classifier = KNN.computeModel(XTrain, yTrain)
        yPred = ClassificationModel.predictModel(classifier, XTest)
        return ClassificationModel.evaluateModel(yPred, yTest),ClassificationModel.mae_loss(yPred,yTest),ClassificationModel.mse_loss(yPred,yTest)
    #este método foi criado apra se criar o clasificador e utiliza-lo em um caso real e não nas partes avaliativas como o compute example.
    def createClassifier(filename):
        XTrain, XTest, yTrain, yTest = ClassificationModel.preprocessData(filename)

        classifier = KNN.computeModel(XTrain, yTrain)
        
        return classifier


