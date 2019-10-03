"""
Leonardo Vilela Ribeiro
Classe Criada para instanciar um modelo classificador  para 
Supor Vector Machine
"""

from classification import ClassificationModel

class SVM(ClassificationModel):
    #método que recebe os dados de treinamento(XTrain, yTrain) e o grau do SVM (d) a ser utilizado
    # e o kernel
    #retorna um classificador instanciado e formatado
    def computeModel(XTrain, yTrain, k, d):
        from sklearn.svm import SVC

        classifier = SVC(kernel=k, degree=d)
        classifier.fit(XTrain[0], yTrain)

        return classifier
    #método que recebe o arquivo a ser trabalhado, executa alguns pré-processamentos
    #(nem todos porque alguns préprocessamentos foram feitos analisando o arquivo original
    #com a ferramenta microsoft excell)
    #retorna uma matriz de confusão criada com o método evaluate model
    #este método chama na clase importada os métodos de calculo da função de perda(custo) mae e mse
    def computeExample(filename, kernel, degree):
        XTrain, XTest, yTrain, yTest = ClassificationModel.preprocessData(filename)

        classifier = SVM.computeModel(XTrain, yTrain, kernel, degree)
        yPred = ClassificationModel.predictModel(classifier, XTest)
        return ClassificationModel.evaluateModel(yPred, yTest),ClassificationModel.mae_loss(yPred,yTest),ClassificationModel.mse_loss(yPred,yTest)


