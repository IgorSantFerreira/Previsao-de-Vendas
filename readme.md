# Previsão de Vendas com Machine Learning

Neste projeto, trabalhei com uma base de investimentos em publicidade para prever vendas a partir de três canais: TV, rádio e jornal. A ideia foi ir além de ajustar um único modelo: quis entender as relações entre as variáveis, testar se cada canal realmente ajudava na previsão e comparar abordagens com diferentes níveis de complexidade.

Desenvolvi a análise em Python, no Jupyter Notebook, passando pela exploração dos dados, análise estatística, treinamento, validação cruzada e avaliação final. O foco é **previsão de vendas**, não medir o impacto causal de cada investimento.

## Base de dados

Utilizei o arquivo [`advertising.csv`](advertising.csv), que contém **200 observações** e quatro colunas:

| Coluna | Descrição |
| --- | --- |
| `TV` | Investimento em publicidade na TV |
| `Radio` | Investimento em publicidade no rádio |
| `Newspaper` | Investimento em publicidade em jornal |
| `Sales` | Vendas, variável que quero prever |

Separei **160 registros (80%) para análise e treinamento** e **40 (20%) para teste**, com `random_state=42`. Fiz a exploração e a seleção dos modelos apenas com os dados de treino, deixando o teste reservado para a etapa final.

A base não informa de forma explícita moeda, unidades de medida, período ou método de coleta. Por isso, mantive os valores nas unidades originais do CSV, sem tratá-los como valores monetários.

## Etapas da análise

### 1. Exploração dos dados

Comecei verificando a estrutura da base, valores ausentes, duplicatas e estatísticas descritivas. Também calculei a assimetria e usei a regra de **1,5 × IQR** para identificar possíveis valores extremos. Optei por manter esses registros, já que não havia evidência de que fossem erros.

Depois, investiguei a relação entre investimentos e vendas com:

- Correlações de **Pearson** e **Spearman**;
- Correlação parcial entre rádio e vendas, controlando por TV, e entre jornal e vendas, controlando por TV e rádio;
- Histogramas e gráficos de dispersão feitos com **Plotly**, incluindo curvas LOWESS para visualizar tendências.

Essa etapa sugeriu uma relação mais forte entre TV e vendas, com possível contribuição adicional do rádio. Usei essas observações para formular comparações entre modelos, sem assumir que correlação significa causalidade.

### 2. Comparação dos modelos

Testei sete alternativas, começando com uma referência simples e aumentando gradualmente a complexidade:

| Modelo | O que avaliei |
| --- | --- |
| Previsão pela média (`DummyRegressor`) | Referência inicial para comparar os demais modelos |
| Regressão linear com TV | Quanto consigo prever usando apenas TV |
| Regressão linear com TV + rádio | Se o rádio acrescenta informação |
| Regressão linear com três canais | Se incluir jornal melhora o resultado |
| Regressão com interação TV × rádio | Se a combinação dos dois canais ajuda na previsão |
| Regressão polinomial com Ridge | Se termos de segundo grau e regularização melhoram o ajuste |
| Rede neural MLP | Se um modelo não linear mais flexível traz ganho preditivo |

Para comparar os modelos, usei **RepeatedKFold com 5 divisões e 2 repetições**. O principal critério foi o menor **RMSE médio na validação**, acompanhado de MAE e R².

No Ridge e na rede neural, utilizei `GridSearchCV` dentro de cada divisão externa da validação cruzada. A expansão polinomial e a padronização também ficaram dentro dos pipelines, para que o tratamento dos dados fosse aprendido somente no treino de cada divisão.

Na MLP, comparei camadas ocultas com 5 e 10 neurônios, ativação `tanh`, otimizador `lbfgs` e diferentes valores de regularização. As entradas e a variável-alvo foram padronizadas durante o treinamento.

## Resultados

### Validação cruzada

Estes foram os RMSEs médios registrados no notebook para os modelos de maior destaque:

| Modelo | RMSE médio de validação |
| --- | ---: |
| Rede neural MLP | **1,442** |
| Ridge polinomial | 1,537 |
| Regressão linear (TV, rádio e jornal) | 1,696 |

A **MLP** apresentou o menor RMSE médio. O ajuste selecionado pelo `GridSearchCV` utilizou **10 neurônios em uma camada oculta** e **`alpha = 0,1`**.

A comparação incremental também trouxe dois resultados interessantes: acrescentar **rádio** à TV reduziu o RMSE nas dez divisões avaliadas; já acrescentar **jornal** ao modelo com TV e rádio trouxe melhora em apenas duas delas, sem ganho médio nessa amostra. Isso não significa que jornal seja irrelevante em qualquer situação.

### Avaliação no conjunto de teste

Depois de selecionar o modelo na validação, avaliei a MLP nas **40 observações que ficaram reservadas**:

| Métrica | Resultado da MLP |
| --- | ---: |
| MAE | **0,9608** |
| RMSE | **1,2578** |
| R² | **0,9488** |

Como referência, a regressão linear com os três canais teve **RMSE de 1,705** nesse mesmo teste. Também comparei valores previstos com os reais, analisei os resíduos e identifiquei os registros com maiores erros.

Existe, porém, um ponto de atenção: o RMSE de treino da MLP foi **1,000**, enquanto o RMSE médio de validação ficou em **1,442**. Essa diferença merece acompanhamento por indicar possível sobreajuste.

### Exemplo de previsão

Por fim, deixei no notebook um exemplo com os seguintes investimentos nas unidades do CSV: **TV = 150, rádio = 25 e jornal = 30**. O modelo escolhido faz uma estimativa de `Sales` para esse cenário. O exemplo demonstra como usar o modelo, mas não representa uma recomendação de distribuição de orçamento.

## Tecnologias utilizadas

- **Python:** linguagem utilizada na análise e modelagem;
- **pandas e NumPy:** manipulação dos dados e operações numéricas;
- **Plotly:** visualizações exploratórias e gráficos de avaliação;
- **Pingouin:** correlações parciais;
- **scikit-learn:** modelos de regressão, pipelines, validação cruzada, ajuste de hiperparâmetros e métricas;
- **statsmodels:** dependência utilizada pelas linhas de tendência LOWESS do Plotly;
- **Jupyter Notebook:** organização e execução do estudo.

## Como executar

Com Python instalado, clone este repositório e instale as dependências:

~~~bash
git clone https://github.com/IgorSantFerreira/Previsao-de-Vendas.git
cd Previsao-de-Vendas
python -m pip install jupyter numpy pandas plotly pingouin statsmodels scikit-learn
jupyter notebook previsao_vendas_machine_learning.ipynb
~~~

O arquivo `advertising.csv` precisa permanecer na mesma pasta do notebook. Depois, basta executar as células na ordem apresentada.

### Arquivos do repositório

~~~text
Previsao-de-Vendas/
├── advertising.csv
├── previsao_vendas_machine_learning.ipynb
└── readme.md
~~~

## Limitações e próximos passos

Apesar dos resultados, a análise tem limitações: são apenas 200 observações, o teste final tem 40 registros e a origem e as unidades da base não estão documentadas. Além disso, faltam variáveis que poderiam influenciar as vendas, como preço, sazonalidade e concorrência.

Como próximos passos, pretendo documentar melhor a procedência dos dados e avaliar a capacidade de generalização em uma base externa, com observações novas. Também faria sentido testar variáveis adicionais e acompanhar a estabilidade do modelo antes de pensar em uma aplicação real.

O principal aprendizado deste projeto foi comparar modelos de forma estruturada, separando a exploração dos dados da avaliação final e evitando escolher o modelo apenas pelo desempenho no treinamento.

---

**Notebook completo:** [`previsao_vendas_machine_learning.ipynb`](previsao_vendas_machine_learning.ipynb)