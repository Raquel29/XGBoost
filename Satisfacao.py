import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

#Leitura dos dados
dados = pd.read_csv('Base.csv',sep=None)
dados.head()

#Filtando as colunas 
dados = dados.filter(items=['Satisfacao','Fornecimento_constante','Fornec_s_Var_Tensao','agilidade_volta_energia',
'Cumprimento_prazo_servico','eficienci_resolucao_prob','facilidade_contato','aviso_manutencao','entendimento_conta','seguranca_valor_cobrado',
'facilidade_locais_pagto','gentileza_educacao_func','divulgacao_info_importantes','Pensando_Qualidade_avaliar_preco','honestidade_confiabilidade','
preocupacao_c_cliente','chance_de_indicar','mudaria_mais_Qualidade_mais_custo','Problema_Insatisfacao','teve_contato'])
dados.head()

#Exibi percentis, media, max e min dos dados
dados.describe()

#Exibe informações sobre os dados
dados.info()

#Substituir valores NA por media
dados=dados.fillna(dados.mean())

#Verifica quantidade de registros nulos
dados.isnull().sum()

#Formatando Satifacao
dados['Satisfacao']= dados['Satisfacao'].astype(str)
dados['Satisfacao']= dados['Satisfacao'].str.replace(',', '.')
dados['Satisfacao']

#convertendo para float
dados['Satisfacao']= dados['Satisfacao'].astype(float)

#Separa as variaveis 
X, y = dados.iloc[:,:-1],dados.iloc[:,-1]


#converterá o conjunto de dados em uma estrutura de dados otimizada 
data_dmatrix = xgb.DMatrix(data=X,label=y)

#dividindo em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

#Usando XGBRegressor
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)

xg_reg.fit(X_train,y_train)
preds = xg_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))

#Usando XGBClassifier
def run_xgb(x_train,x_test,y_train,y_test):
  xgb_model = XGBClassifier(random_state =44, max_depth=2)
  xgb_model.fit(x_train,y_train)
  print('Treino')
  pred = xgb_model.predict(x_train)
  print('Xgboost roc-auc: {}'.format(roc_auc_score(y_train,pred[:,],multi_class= 'ovr')))
  print('Teste')
  pred = xgb_model.predict(x_test)
  print('Xgboost roc-auc: {}'.format(roc_auc_score(y_test,pred[:,],multi_class= 'ovr')))
  
run_xgb(X_train,X_test, y_train, y_test)
