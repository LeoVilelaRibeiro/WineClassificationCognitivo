# WineClassificationCognitivo
Está sendo utilizado um framework de machine learning preconizado pelo Cientista Lucas Lattari e refatorado por mim para atender diversas demandas de Classificação. Ele oferece a classificação da massa de dados por 3 algoritmos: KNN, SVM e Naive Bayes. Executa o teste com cada um deles e exibe um gráfico comparando as acurácias.
Para que você teste este trabalho com facilidade, coloque tudo em um diretório C:\cognitivo. Caso trabalhe com outra estrutura de arquivos, basta modificar a chamada os.chdir() presente do cabeçalho dos código para o endereço onde seus arquivos estarão.
Temos 4 arquivos principais, onde estão as linhas de execução para processar as entradas de dados:<br/>

  algo_test_winecorrigido.py - Contem a rotina de processamento da fonte de dados sobre vinhos pré_processada e corrigida em ouliers, campos vazios e outras correções. Na pasta Arquivos de Exploração contém resultados da análise exploratória que definiu a limpeza a ser feita.<br/>
  algo_test_comPCA.py - Contem a rotina de processamento da fonte de dados sobre vinhos corrigida e depois reduzida dimensionalmente por Análise de Componentes Principais (PCA) para 2 componentes principais em X . Na pasta Arquivos de Exploração você pode observar o gráfico plotado do resultado da PCA. A qualidade dos resultados com a PCA foi inferior.<br/>
  algo_test_comtitanic.py - Contem uma rotina de processamento utilizando este Framework, para uma fonte de dados sobre os sobreviventes do Titanic. Usei este arquivo para avaliar meu código e as métricas.<br/>
  main_application.py - Contem um exemplo da chamada do classificador em produção, sem componentes de avaliação de performance ou métricas. Simplesmente a chamada e o resultado.<br/>
  Temos também as classes criadas para executar o processamento chamado nas linhas de execução:<br/>
  classification.py - fornece métodos comuns a chamada de qualquer dos algoritmos citados, inclusive o cálculo da função de custo.<br/>
  preprocessing.py - fornece métodos comuns ao pré processamento de qualquer arquivo de entrada.<br/>
  Wine Pre-Processamento.ipynb - mostra no júpiter os estudos exploratórios iniciais para limpeza dos dados.<br/>
  PCA.py - executa análise de componentes principais no arquivo de vinhos reduzindo a dimensionalidade para 2 componentes e plota o grafico.<br/>
  svm.py - instancia o classificador SVM<br/>
  knn.py - instancia o classificador KNN<br/>
  naivebayes.py - instancia o classificador de Naive Bayes <br/>
  
  a. Como foi a definição da sua estratégia de modelagem? <br/>
  A modelagem começou com a transformação do campo type que era "red" ou "white"  em 0 e 1, e a pré suposição de dois formatos de        entrada: todas as variáveis presentes x arquivo com dimensionalidade reduzida. Considerei Y como um resultado numérico viável pois 
  comporta-se como um score.<br/>
   b. Como foi definida a função de custo utilizada?<br/>
  Trabalhamos com duas funções de custo: mae e mse. Utilizei as mesmas por serem bem comuns para modelos de regressão ou classificação     não binária onde o resultado se dá como um score, como foi o caso do exemplo. Considerando que retirei os outliers no arquivo de entrada substituindo os mesmos pela média, a Mae não seria tão penalizada. Como é simples a programação dessas métricas, resolvi então apresentar as duas.<br/>
  c. Qual foi o critério utilizado na seleção do modelo final?<br/>
  Selecionei o modelo final após os testes feitos e avaliação dos resultados apresentado para os dois modelos que preconizei : dados completos ou dados reduzidos por PCA.<br/>
  d. Qual foi o critério utilizado para validação do modelo? Por que escolheu utilizar este método?<br/>
  Fiz 200 iterações com cada um dos algoritmo, estrai a média das métricas ACURÁCIA, MAE E MSE e verifiquei qual o melhor resultado.
  e. Quais evidências você possui de que seu modelo é suficientemente bom?<br/>
  O algoritmo KNN retornou os melhores resultados para a acurácia, porém quanto testamos com a fonte de dados titanic, vemos que pode ainda melhorar. Digo então que inicialmente escolho utilizar o KNN pela melhor acurácia, porém transformações no arquivo ainda devem ser feitas pois o função de custo está alta.<br/>
  Considero que o modelo proposto é suficiente bom para uma avaliação do framework proposto, porém acredito em melhoras. Navegando pelo Github vi implementações de análise deste dataset com 60% de acuracia e no nosso caso em alguns processamentos conseguimos mais de 80%.
  


  
  
  


