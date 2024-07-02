
# # Librerias

# 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly as px
import re
import seaborn as sns
from code_v3 import data

# 
# # Importar datos

# 
df_contract=pd.read_csv('files\\final_provider\contract.csv')
df_personal=pd.read_csv('files\\final_provider\personal.csv')
df_internet=pd.read_csv('files\\final_provider\internet.csv')
df_phone=pd.read_csv('files\\final_provider\phone.csv')

# 
# # Preprocesamiento

# 
# ## Contratos

# 
df_contract.head(10)

# 
df_contract.info()

# 
# Podemos ver que los datos no tienen faltantes al tener completas las filas en cada columna, sin embargo, el tipo de objeto de la columna TotalCharges debería ser float.

# 
#Funcion para pasar columnas al formato snake_case
def to_snake_case(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    s1 = s1.replace(' ','_')
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

# 
#Pasamos las columnas al modo snake_case
columns=df_contract.columns
new_cols=[]
for i in columns:
    i=to_snake_case(i)
    new_cols.append(i)
df_contract.columns=new_cols
print(df_contract.columns)

# 
df_contract.describe()

# 
# ### Ausentes

# 
df_contract.isna().sum()

# 
df_contract['total_charges'].value_counts()

# 
# Evidenciamos que la columna total_charges tiene un valor ' ' que no significa nada por lo cual veremos que impacto tiene. 

# 
df_contract[df_contract['total_charges']==' ']

# 
100*df_contract[df_contract['total_charges']==' ']['total_charges'].count()/df_contract.shape[0]

# 
# Estos datos representan solo el 0.1%, por lo tanto los eliminamos

# 
df_contract['total_charges'].replace(' ',np.nan,inplace=True)

# 
df_contract.dropna(inplace=True)

# 
# ### Duplicados

# 
df_contract.duplicated().sum()

# 
#Pasamos la columna total charges a float
df_contract['total_charges']=df_contract['total_charges'].astype('float')

# 
df_contract.info()

# 
# ## Personal

# 
df_personal.head(10)

# 
df_personal.info()

# 
#Pasamos las columnas al modo snake_case
columns=df_personal.columns
new_cols=[]
for i in columns:
    i=to_snake_case(i)
    new_cols.append(i)
df_personal.columns=new_cols
print(df_personal.columns)

# 
# El dataframe tiene datos basicos de cada cliente y con el metodo info se ve que está completo sin ausentes.

# 
df_personal.describe()

# 
df_personal['senior_citizen'].value_counts()

# 
# ### Ausentes

# 
df_personal.isna().sum()

# 
# ### Duplicados

# 
df_personal.duplicated().sum()

# 
# Este dataset se ve bien sin ausentes ni duplicados.

# 
# ## Internet

# 
df_internet.head(10)

# 
df_internet.info()

# 
#Pasamos las columnas al modo snake_case
columns=df_internet.columns
new_cols=[]
for i in columns:
    i=to_snake_case(i)
    new_cols.append(i)
df_internet.columns=new_cols
print(df_internet.columns)

# 
df_internet.describe()

# 
# ### Ausentes

# 
df_internet.isna().sum()

# 
# ### Duplicados

# 
df_internet.duplicated().sum()

# 
# ## Phone

# 
df_phone.head(10)

# 
df_phone.info()

# 
#Pasamos las columnas al modo snake_case
columns=df_phone.columns
new_cols=[]
for i in columns:
    i=to_snake_case(i)
    new_cols.append(i)
df_phone.columns=new_cols
print(df_phone.columns)

# 
df_phone.describe()

# 
# ### Ausentes

# 
df_phone.isna().sum()

# 
# ### Duplicados

# 
df_phone.duplicated().sum()

# 
# En conclusión de los 4 datasets, están sin ausentes o duplicados, y tienen información binaria acerca de a las características que el usuario tiene en el servicio de internet y telefonía, asi como las caracteristicas del contrato, medios de pago, envío de factura y caracteristicas de los clientes.

# Con el fin de facilitar el análisis, juntaremos todos los datasets en uno solo.

# 
data=df_contract.merge(df_personal.merge(df_phone.merge(df_internet,how='left'),how='left'),how='left')

# 
data.head(10)

# 
data.info()

# 
for item in ['multiple_lines','internet_service','online_security','online_backup',
             'device_protection','tech_support','streaming_tv','streaming_movies']:
    data[item] = data[item].fillna('No')

# Con este gráfico podemos confirmar que las personas que tienen el servicio DSL prefieren tener seguridad online que no tener. Por otro lado los que tienen fibra optica prefieren no tener seguridad online, y tienen más cargos que el servicio DSL.

# 
def cancel(data):
    if data=='No':
        return '0'
    else:
        return '1'

# 
#Creamos la columna cancel para etiquetar los datos
data['cancel']=data['end_date'].apply(cancel)
data.head(5)

# 
data['end_date'].value_counts()

# 
# ## ¿Cual es la tasa de cancelación y que caracteristicas tienen los usuarios que cancelan y los que no?

# 
print('Tasa de cancelación: ',100*data[data['cancel']=='1']['customer_id'].count()/data.shape[0])

# 
data_cancel_rate=data.groupby(['cancel'])['customer_id'].count()
data_cancel_rate.plot(kind='bar')
plt.show()

# # Código solución

# 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (f1_score,accuracy_score,roc_auc_score)
from sklearn.feature_selection import SelectFromModel
#from sklearn.pipeline import Pipeline
from scipy.stats import randint as sp_randint
from scipy.stats import uniform
import joblib
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc
import statsmodels.api as sm

# 
data.head()
data['cancel']=pd.to_numeric(data['cancel'])

# 
# ## Codificamos nuestros datos categoricos

# 
data_model=data.drop(['customer_id','begin_date','end_date','internet_service'],axis=1)

# 
def binary_categories(data):
    if data=='Yes' or data=='Male':
        return 1
    else:
        return 0

# 
binary_columns=['gender','partner', 'dependents','paperless_billing', 'online_security', 'online_backup',
       'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies',
       'multiple_lines']
for column in binary_columns:
       data_model[column]=data_model[column].apply(binary_categories)

# 
one_hot=pd.get_dummies(data_model[['type','payment_method']])
data_model = pd.concat([data_model, one_hot], axis=1).drop(columns=['type','payment_method'])

# 
data_model.dtypes

# 
# Convertir variables booleanas a numéricas (0 y 1)
bolean_list=['type_Month-to-month','type_One year','type_Two year','payment_method_Bank transfer (automatic)','payment_method_Credit card (automatic)','payment_method_Electronic check','payment_method_Mailed check']
data_model[bolean_list] = data_model[bolean_list].astype(int)

# 
# ## Análisis de correlación

# 
# Calcular la matriz de correlación
correlation_matrix = data_model.corr()

# Mostrar la matriz de correlación
print(correlation_matrix)

# 
# Crear un mapa de calor de la matriz de correlación
plt.figure(figsize=(20, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Matriz de Correlación')
plt.show()

# 
# Seleccionar las variables dependiente (y) e independiente (X)
# Supongamos que 'target' es la variable dependiente y las demás son independientes
X = data_model.drop(columns=['cancel'])
y = data_model['cancel']

# Añadir una constante a las variables independientes
X = sm.add_constant(X)

# Ajustar el modelo de regresión
model = sm.OLS(y, X).fit()

# Mostrar un resumen del modelo
print(model.summary())

# 
# Graficar las variables independientes contra la variable dependiente
for col in X.columns[1:]:  # Excluir la constante añadida
    plt.figure(figsize=(8, 6))
    sns.regplot(x=col, y='cancel', data=data_model)
    plt.title(f'Relación entre {col} y target')
    plt.xlabel(col)
    plt.ylabel('target')
    plt.show()

# 
# ## Separamos los datos de entrenamiento, validación y testeo

# 
seed=12345
data_train,data_test=train_test_split(data_model,random_state=seed,test_size=0.3)

# 
features=data_train.drop(['cancel'],axis=1)
target=data_train['cancel']
features_train,features_valid,target_train,target_valid=train_test_split(features,target,random_state=seed,test_size=0.3)

# 
features_train.head(5)

# 
# ## Escalamos los datos

# 
#Vamos a escalar las características para que nuestro modelo pueda tomar estas variables
numeric=['total_charges','monthly_charges']
scaler=StandardScaler()
scaler.fit(features_train[numeric])
features_train[numeric]=scaler.transform(features_train[numeric])
features_valid[numeric]=scaler.transform(features_valid[numeric])

# 
# ## Balanceamos los datos

# 
#Definimos la función para arreglar el sobremuestreo
def upsample(features, target, repeat):
    #Primero dividimos el conjunto de datos de entrenamiento en positivos y negativos 
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]
    #Posteriormente multiplicamos los datos de la clase que tiene menos datos, en este caso la clase 1 y unimos todos los datos
    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)
    #Por último, mesclamos todos los datos con la función shuffle y devolvemos los datos desbalanceados
    features_upsampled, target_upsampled = shuffle(
        features_upsampled, target_upsampled, random_state=seed
    )

    return features_upsampled, target_upsampled

# 
features_train_resampled, target_train_resampled = upsample(
    features_train, target_train, 2)

# 
#smote = SMOTE(random_state=seed)
#features_train_resampled, target_train_resampled = smote.fit_resample(features_train, target_train)

# 
# ## Entrenamos los modelos

# 
def eval_model(model,features_valid,target_valid):
    best_random = model.best_estimator_
    random_prediction = best_random.predict(features_valid)
    prob = best_random.predict_proba(features_valid)[:, 1]
    random_accuracy=accuracy_score(target_valid,random_prediction)
    random_f1_score=f1_score(target_valid,random_prediction)
    random_roc_auc=roc_auc_score(target_valid,prob)
    print("Accuracy:",random_accuracy)
    print('f1: ',random_f1_score)
    print('ROC_AUC: ',random_roc_auc)
    print('Best_Model: ',best_random)
    return best_random,random_accuracy,random_f1_score,random_roc_auc

# 
# ### RandomForest

# 
params_rf={'max_depth':[3,4,2,1],
         'min_samples_split':[5,10,11,12],
         'n_estimators':[10,20,30,40],
         'min_samples_leaf':[1, 2, 4],
         'bootstrap':[True, False]}

# 
model_rf=RandomForestClassifier(random_state=seed)
selector_rf = SelectFromModel(estimator=model_rf, threshold='mean')
selector_rf.fit(features_train_resampled,target_train_resampled)
selected_features_rf = features_train_resampled.columns[selector_rf.get_support()].tolist()
print("Características seleccionadas:", selected_features_rf)

# 
rs=GridSearchCV(estimator=model_rf,param_grid=params_rf,scoring='f1',cv=2)
rs.fit(features_train_resampled,target_train_resampled)

# 
results_rs=eval_model(rs,features_valid,target_valid)

# 
joblib.dump(results_rs[0],'files/models/best_random_rf.joblib')

# 
# ### LGBM Classifier

# 
param_dist = {
    'n_estimators':range(50, 201, 50),
    'max_depth': range(1, 21)    
} 

# 
model_lb=LGBMClassifier(random_state=seed)
selector_lb = SelectFromModel(estimator=model_lb, threshold='mean')
selector_lb.fit(features_train_resampled,target_train_resampled)
selected_features_lb = features_train_resampled.columns[selector_lb.get_support()].tolist()
print("Características seleccionadas:", selected_features_lb)

# 
lb=GridSearchCV(estimator=model_lb,param_grid=param_dist,scoring='f1',cv=2)
lb.fit(features_train,target_train)

# 
results_lb=eval_model(lb,features_valid,target_valid)

# 
joblib.dump(results_lb[0],'files/models/best_random_lb.joblib')

# 
# ### Catboost

# 
param_dist = {'iterations': range(50, 201, 50),
    'depth': range(1, 11)
    }

# 
model_cat=CatBoostClassifier(random_state=seed)
selector_cat = SelectFromModel(estimator=model_cat)
selector_cat.fit(features_train_resampled,target_train_resampled)
selected_features_cat = features_train_resampled.columns[selector_cat.get_support()].tolist()
print("Características seleccionadas:", selected_features_cat)

# 
cat=GridSearchCV(estimator=model_cat,param_grid=param_dist,scoring='f1',cv=2)
cat.fit(features_train_resampled,target_train_resampled)

# 
results_cat=eval_model(cat,features_valid,target_valid)

# 
joblib.dump(results_cat,'files/models/best_random_cat.joblib')

# 
# ### Logistic Regression

# 
lr=LogisticRegression(random_state=seed)
selector_lr = SelectFromModel(estimator=lr, threshold='mean')
selector_lr.fit(features_train_resampled,target_train_resampled)
selected_features_lr = features_train_resampled.columns[selector_lr.get_support()].tolist()
print("Características seleccionadas:", selected_features_lr)

# 
lr.fit(features_train[selected_features_lr],target_train)

# 
best_random_lr = lr
random_prediction = best_random_lr.predict(features_valid)
prob_lr = best_random_lr.predict_proba(features_valid)[:, 1]
random_accuracy=accuracy_score(target_valid,random_prediction)
random_f1_score=f1_score(target_valid,random_prediction)
random_roc_auc=roc_auc_score(target_valid,prob_lr)
print("Accuracy:",random_accuracy)
print('f1: ',random_f1_score)
print('ROC_AUC: ',random_roc_auc)

# 
joblib.dump(best_random_lr,'files/models/best_random_lr.joblib')

# 
# Al revisar todos los modelos, podemos ver que el que mejor métrica ROC_AUC tiene es el de Logistic_Regression, por lo cual probaremos este primero, y posteriormente probaremos Catboost

# 
# ### XG Boost

# 
param_grid = {'n_estimators': range(50, 201, 50),
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10],
    }

# 
model_xg=XGBClassifier(random_state=12345, use_label_encoder=False, eval_metric='logloss')
selector_xg = SelectFromModel(estimator=model_xg, threshold='mean')
selector_xg.fit(features_train_resampled,target_train_resampled)
selected_features_xg = features_train_resampled.columns[selector_xg.get_support()].tolist()
print("Características seleccionadas:", selected_features_xg)

# 
xg=GridSearchCV(estimator=model_xg,param_grid=param_dist,scoring='f1',cv=2)
xg.fit(features_train_resampled,target_train_resampled)

# 
results_xg=eval_model(xg,features_valid,target_valid)

# 
joblib.dump(results_xg,'files/models/best_random_xg.joblib')

# 
# ## Testeamos los datos

# 
features_test=data_test.drop(['cancel'],axis=1)
target_test=data_test['cancel']

# 
# ### Escalamos los datos

# 
features_test[numeric]=scaler.transform(features_test[numeric])

# 
# ### Catboost

# 
results_cat=eval_model(cat,features_test,target_test)

# 
# ### Logistic Regression

# 
prediction_lr=best_random_lr.predict(features_test)
prob_lr = best_random_lr.predict_proba(features_test)[:, 1]
random_accuracy=accuracy_score(target_test,prediction_lr)
random_f1_score=f1_score(target_test,prediction_lr)
random_roc_auc=roc_auc_score(target_test,prob_lr)
print("Accuracy:",random_accuracy)
print('f1: ',random_f1_score)
print('ROC_AUC: ',random_roc_auc)

# 
# ### Random Forest

# 
results_rf=eval_model(rs,features_test,target_test)

# 
# ### Light_GBM

# 
results_lb=eval_model(lb,features_test,target_test)

# 
# ### XG_boost

# 
results_xg=eval_model(xg,features_test,target_test)

# 
# # Conclusiones
# 
# 1. El modelo que mejor se ajustó a nuestros datos para clasificar las personas que cancelan fue el de CatBoost con un AUC ROC de 0.839, seguido de RandomForest con 0.835 y LogisticRegression con 0.833.
# 
# 1. Se debe tener en cuenta que para poder obtener este grado de calidad se balancearon los datos con el fin de evitar el sobre ajuste, utilizando la técnica UpSampling.
# 
# 4. Se utilizó la función SelectFromModel para cada modelo con el fin de segmentar las caracteristicas optimas para cada caso específico.


