# %% [markdown]
# # Condiciones de la asignación principal
# 
# Al operador de telecomunicaciones Interconnect le gustaría poder pronosticar su tasa de cancelación de clientes. Si se descubre que un usuario o usuaria planea irse, se le ofrecerán códigos promocionales y opciones de planes especiales. El equipo de marketing de Interconnect ha recopilado algunos de los datos personales de sus clientes, incluyendo información sobre sus planes y contratos.
# 
# ## Servicios de Interconnect
# 
# Interconnect proporciona principalmente dos tipos de servicios:
# 
# 1. Comunicación por teléfono fijo. El teléfono se puede conectar a varias líneas de manera simultánea.
# 2. Internet. La red se puede configurar a través de una línea telefónica (DSL, *línea de abonado digital*) o a través de un cable de fibra óptica.
# 
# Algunos otros servicios que ofrece la empresa incluyen:
# 
# - Seguridad en Internet: software antivirus (*ProtecciónDeDispositivo*) y un bloqueador de sitios web maliciosos (*SeguridadEnLínea*).
# - Una línea de soporte técnico (*SoporteTécnico*).
# - Almacenamiento de archivos en la nube y backup de datos (*BackupOnline*).
# - Streaming de TV (*StreamingTV*) y directorio de películas (*StreamingPelículas*)
# 
# La clientela puede elegir entre un pago mensual o firmar un contrato de 1 o 2 años. Puede utilizar varios métodos de pago y recibir una factura electrónica después de una transacción.
# 
# ### Descripción de los datos
# 
# Los datos consisten en archivos obtenidos de diferentes fuentes:
# 
# - `contract.csv` — información del contrato;
# - `personal.csv` — datos personales del cliente;
# - `internet.csv` — información sobre los servicios de Internet;
# - `phone.csv` — información sobre los servicios telefónicos.
# 
# En cada archivo, la columna `customerID` (ID de cliente) contiene un código único asignado a cada cliente. La información del contrato es válida a partir del 1 de febrero de 2020.

# %% [markdown]
# # Librerias

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly as px
import re
import seaborn as sns

# %% [markdown]
# # Importar datos

# %%
df_contract=pd.read_csv('final_provider\contract.csv')
df_personal=pd.read_csv('final_provider\personal.csv')
df_internet=pd.read_csv('final_provider\internet.csv')
df_phone=pd.read_csv('final_provider\phone.csv')

# %% [markdown]
# # Preprocesamiento

# %% [markdown]
# ## Contratos

# %%
df_contract.head(10)

# %%
df_contract.info()

# %% [markdown]
# Podemos ver que los datos no tienen faltantes al tener completas las filas en cada columna, sin embargo, el tipo de objeto de la columna TotalCharges debería ser float.

# %%
#Funcion para pasar columnas al formato snake_case
def to_snake_case(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    s1 = s1.replace(' ','_')
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

# %%
#Pasamos las columnas al modo snake_case
columns=df_contract.columns
new_cols=[]
for i in columns:
    i=to_snake_case(i)
    new_cols.append(i)
df_contract.columns=new_cols
print(df_contract.columns)

# %%
df_contract.describe()

# %% [markdown]
# ### Ausentes

# %%
df_contract.isna().sum()

# %%
df_contract['total_charges'].value_counts()

# %% [markdown]
# Evidenciamos que la columna total_charges tiene un valor ' ' que no significa nada por lo cual veremos que impacto tiene. 

# %%
df_contract[df_contract['total_charges']==' ']

# %%
100*df_contract[df_contract['total_charges']==' ']['total_charges'].count()/df_contract.shape[0]

# %% [markdown]
# Estos datos representan solo el 0.1%, por lo tanto los eliminamos

# %%
df_contract['total_charges'].replace(' ',np.nan,inplace=True)

# %%
df_contract.dropna(inplace=True)

# %% [markdown]
# ### Duplicados

# %%
df_contract.duplicated().sum()

# %%
#Pasamos la columna total charges a float
df_contract['total_charges']=df_contract['total_charges'].astype('float')

# %%
df_contract.info()

# %% [markdown]
# ## Personal

# %%
df_personal.head(10)

# %%
df_personal.info()

# %%
#Pasamos las columnas al modo snake_case
columns=df_personal.columns
new_cols=[]
for i in columns:
    i=to_snake_case(i)
    new_cols.append(i)
df_personal.columns=new_cols
print(df_personal.columns)

# %% [markdown]
# El dataframe tiene datos basicos de cada cliente y con el metodo info se ve que está completo sin ausentes.

# %%
df_personal.describe()

# %%
df_personal['senior_citizen'].value_counts()

# %% [markdown]
# ### Ausentes

# %%
df_personal.isna().sum()

# %% [markdown]
# ### Duplicados

# %%
df_personal.duplicated().sum()

# %% [markdown]
# Este dataset se ve bien sin ausentes ni duplicados.

# %% [markdown]
# ## Internet

# %%
df_internet.head(10)

# %%
df_internet.info()

# %%
#Pasamos las columnas al modo snake_case
columns=df_internet.columns
new_cols=[]
for i in columns:
    i=to_snake_case(i)
    new_cols.append(i)
df_internet.columns=new_cols
print(df_internet.columns)

# %%
df_internet.describe()

# %% [markdown]
# ### Ausentes

# %%
df_internet.isna().sum()

# %% [markdown]
# ### Duplicados

# %%
df_internet.duplicated().sum()

# %% [markdown]
# ## Phone

# %%
df_phone.head(10)

# %%
df_phone.info()

# %%
#Pasamos las columnas al modo snake_case
columns=df_phone.columns
new_cols=[]
for i in columns:
    i=to_snake_case(i)
    new_cols.append(i)
df_phone.columns=new_cols
print(df_phone.columns)

# %%
df_phone.describe()

# %% [markdown]
# ### Ausentes

# %%
df_phone.isna().sum()

# %% [markdown]
# ### Duplicados

# %%
df_phone.duplicated().sum()

# %% [markdown]
# En conclusión de los 4 datasets, están sin ausentes o duplicados, y tienen información binaria acerca de a las características que el usuario tiene en el servicio de internet y telefonía, asi como las caracteristicas del contrato, medios de pago, envío de factura y caracteristicas de los clientes.

# %% [markdown]
# # Análisis exploratorio de datos (EDA)

# %% [markdown]
# ## Analisis de contrato

# %%
df_contract.head()

# %% [markdown]
# ## ¿Que tipo de contrato es el que más escogen los clientes?

# %%
df_type=df_contract.groupby(['type'])['customer_id'].count()
df_type.plot(kind='bar')
plt.show()

# %% [markdown]
# En general podemos ver que más personas optan por el contrato mes a mes, seguido del contrato de dos años y el contrato de un año.

# %%
df_paperless=df_contract.groupby(['paperless_billing'])['customer_id'].count()
df_paperless.plot(kind='bar')
plt.show()

# %% [markdown]
# La mayoría de personas prefieren recibir su factura electrónica.

# %%
df_method=df_contract.groupby(['payment_method'])['customer_id'].count()
df_method=df_method.sort_values(ascending=False)
df_method.plot(kind='bar')
plt.show()

# %% [markdown]
# Los clientes prefieren el pago electronico, seguido por el cheque por correo, transferencia bancaria y tarjeta de crédito (autopago).

# %%
df_type=df_contract.groupby(['type'])['monthly_charges'].sum()
df_type.plot(kind='bar')
plt.show()

# %% [markdown]
# Podemos ver que los que más cargos tienen son los que tienen contrato més a més,seguido del que tiene contrato de dos años. y el que tiene contrato de un año.

# %% [markdown]
# ## Análisis Personal

# %%
df_personal.head(10)

# %%
df_gender=df_personal.groupby(['gender'])['customer_id'].count()
df_gender.plot(kind='bar')
plt.show()

# %% [markdown]
# Los clientes están balanceados, más hombres que mujeres, sin embargo, casi la misma cantidad.

# %%
df_senior=df_personal.groupby(['senior_citizen'])['customer_id'].count()
df_senior.plot(kind='bar')
plt.show()

# %% [markdown]
# La mayoría de los clientes no son adultos mayores.

# %% [markdown]
# ## Internet

# %%
df_internet.head()

# %% [markdown]
# ## ¿Cual es el servicio de internet más demandado?

# %%
df_service=df_internet.groupby(['internet_service'])['customer_id'].count()
df_service.plot(kind='bar')
plt.show()

# %% [markdown]
# La mayoría de los clientes utiliza fibra obtica.

# %%
df_security=df_internet.groupby(['online_security'])['customer_id'].count()
df_security.plot(kind='bar')
plt.show()

# %% [markdown]
# La mayoría de clientes opta por no tener seguridad.

# %%
df_backup=df_internet.groupby(['online_backup'])['customer_id'].count()
df_backup.plot(kind='bar')
plt.show()

# %% [markdown]
# La mayoría de clientes prefiere no tener respaldo de información.

# %%
df_support=df_internet.groupby(['tech_support'])['customer_id'].count()
df_support.plot(kind='bar')
plt.show()

# %% [markdown]
# La mayoría de clientes prefiere no tener soporte técnico

# %%
df_dev_prot=df_internet.groupby(['device_protection'])['customer_id'].count()
df_dev_prot.plot(kind='bar')
plt.show()

# %% [markdown]
# La mayoría de clientes prefiere no tener protección de su dispositivo.

# %%
df_streaming_tv=df_internet.groupby(['streaming_tv'])['customer_id'].count()
df_streaming_tv.plot(kind='bar')
plt.show()

# %% [markdown]
# Aunque los clientes prefieren no tener televisión, los clientes que si lo prefieren se acercan a las que no, por lo cual está balanceado este servicio.

# %%
df_streaming_movies=df_internet.groupby(['streaming_movies'])['customer_id'].count()
df_streaming_movies.plot(kind='bar')
plt.show()

# %% [markdown]
# Aunque los clientes prefieren no tener peliculas, los clientes que si lo prefieren se acercan a las que no, por lo cual está balanceado este servicio.

# %% [markdown]
# ## Phone

# %%
df_phone.head()

# %%
df_multiple=df_phone.groupby(['multiple_lines'])['customer_id'].count()
df_multiple.plot(kind='bar')
plt.show()

# %% [markdown]
# La mayoría de usuarios no tienen multimples lineas.

# %% [markdown]
# Con el fin de facilitar el análisis, juntaremos todos los datasets en uno solo.

# %%
data=df_personal.merge(df_contract.merge(df_internet.merge(df_phone,how='inner'),how='inner'),how='inner')

# %%
data.head(10)

# %% [markdown]
# ## ¿Cuales son los medios de pago más utilizados?

# %%
df_payment_method=data.groupby(['payment_method','senior_citizen'],as_index=False)['customer_id'].count()
df_payment_method.sort_values(by='customer_id',ascending=False,inplace=True)
sns.barplot(df_payment_method,y='payment_method',x='customer_id',hue='senior_citizen',orient='h')

# %% [markdown]
# Sin importar si los ususarios son adutos mayores o no, el medio de pago más usado es el cheque electronico, seguido por transferencia bancaria, tarjeta de crédito (autopago) y cheque por correo.

# %% [markdown]
# ## ¿Que tipo de factura prefieren los usuarios?

# %%
df_paperless_billing=data.groupby(['paperless_billing','senior_citizen'],as_index=False)['customer_id'].count()
df_paperless_billing.sort_values(by='customer_id',ascending=False,inplace=True)
sns.barplot(df_paperless_billing,y='paperless_billing',x='customer_id',hue='senior_citizen',orient='h')

# %% [markdown]
# Al igual que el medio de pago, la mayoría de los usuarios sin importar que sean adultos mayores prefieren no recibir su factura por correo.

# %% [markdown]
# ## ¿Que tipo de contrato prefieren las personas que tienen personas a cargo?

# %%
df_total_charges=data.groupby(['type','dependents'],as_index=False)['total_charges'].sum()
df_total_charges.sort_values(by='total_charges',ascending=False,inplace=True)
sns.barplot(df_total_charges,y='type',x='total_charges',hue='dependents',orient='h')

# %% [markdown]
# La mayoría de los clientes tienen más cargos y no tienen personas dependientes de ellos, además podemos ver que las personas que pagan mes a mes tienen más cargos

# %%
df_internet_service_1=data.groupby(['internet_service','online_security'],as_index=False)['total_charges'].sum()
df_internet_service_1.sort_values(by='total_charges',ascending=False,inplace=True)
sns.barplot(df_internet_service_1,y='internet_service',x='total_charges',hue='online_security',orient='h')

# %% [markdown]
# Podemos ver que los que tienen fibra obtica y no tiene seguridad online pagan más que los que si la tienen.

# %%
df_internet_service_1=data.groupby(['internet_service','online_security'],as_index=False)['customer_id'].count()
df_internet_service_1.sort_values(by='customer_id',ascending=False,inplace=True)
sns.barplot(df_internet_service_1,y='internet_service',x='customer_id',hue='online_security',orient='h')

# %% [markdown]
# Con este gráfico podemos confirmar que las personas que tienen el servicio DSL prefieren tener seguridad online que no tener. Por otro lado los que tienen fibra optica prefieren no tener seguridad online, y tienen más cargos que el servicio DSL.

# %%
def cancel(data):
    if data=='No':
        return '0'
    else:
        return '1'

# %%
#Creamos la columna cancel para etiquetar los datos
data['cancel']=data['end_date'].apply(cancel)
data.head(5)

# %%
data['end_date'].value_counts()

# %% [markdown]
# ## ¿Cual es la tasa de cancelación y que caracteristicas tienen los usuarios que cancelan y los que no?

# %%
print('Tasa de cancelación: ',100*data[data['cancel']=='1']['customer_id'].count()/data.shape[0])

# %%
data_cancel_rate=data.groupby(['cancel'])['customer_id'].count()
data_cancel_rate.plot(kind='bar')
plt.show()

# %% [markdown]
# Podemos ver que la tasa de cancelación es del 32% lo cual es bastante alto teniendo en cuenta solo 4 meses, veremos más a fondo que tipo de caracteristicas tiene el plan de cada uno de los que cancelaron y los que no cancelaron.

# %%
df_internet_service_2=data.groupby(['internet_service','cancel'],as_index=False)['customer_id'].count()
df_internet_service_2.sort_values(by='cancel',ascending=False,inplace=True)
sns.barplot(df_internet_service_2,y='internet_service',x='customer_id',hue='cancel',orient='h')

# %% [markdown]
# Los que cancelan y los que no cancelan siguen la misma tendencia a tener más fibra optica que DSL.

# %%
df_payment_type=data.groupby(['type','cancel'],as_index=False)['customer_id'].count()
df_payment_type.sort_values(by='cancel',ascending=False,inplace=True)
sns.barplot(df_payment_type,y='type',x='customer_id',hue='cancel',orient='h')

# %% [markdown]
# Aqui la diferencia se ve marcada, debido a que la mayoría de clientes que cancelan tienen el plan mes a mes, mientras que los que tienen planes anuales no suelen cancelar, sino que se mantienen.

# %%
df_expenses=data.groupby(['type','cancel'],as_index=False)['total_charges'].sum()
df_expenses.sort_values(by='cancel',ascending=False,inplace=True)
sns.barplot(df_expenses,y='type',x='total_charges',hue='cancel',orient='h')

# %% [markdown]
# Podemos ver que los que cancelan gastan más en los planes mes a mes, lo que tiene complementa el anterior análisis.

# %%
data.columns

# %%
df_multiline=data.groupby(['cancel','multiple_lines'],as_index=False)['customer_id'].count()
df_multiline.sort_values(by='cancel',ascending=False,inplace=True)
sns.barplot(df_multiline,y='cancel',x='customer_id',hue='multiple_lines',orient='h')

# %% [markdown]
# Las personas que tienen multiples lineas suelen cancelar más que las que no lo tienen, por lo que puede ser un servicio desatendido.

# %%
df_p_method=data.groupby(['payment_method','cancel'],as_index=False)['customer_id'].count()
df_p_method.sort_values(by='cancel',ascending=False,inplace=True)
sns.barplot(df_p_method,y='payment_method',x='customer_id',hue='cancel',orient='h')

# %%
df_multiline=data.groupby(['cancel','multiple_lines'],as_index=False)['customer_id'].count()
df_multiline.sort_values(by='cancel',ascending=False,inplace=True)
sns.barplot(df_multiline,y='cancel',x='customer_id',hue='multiple_lines',orient='h')

# %% [markdown]
# La mayoría de clientes que cancelan, tienen el método de pago de cheque electrónico.

# %%
def pop_service(data,columns):
    list=[]
    for column in data[columns]:
        percentage=100*data[data[column]=='Yes']['customer_id'].count()/data.shape[0]
        list.append(percentage)
    pop_services={'Servicio':columns,'percentage':list}    
    df_pop_services=pd.DataFrame(pop_services)
    return df_pop_services


# %%
columns=['online_security', 'online_backup', 'device_protection', 'tech_support',
       'streaming_tv', 'streaming_movies', 'multiple_lines']
df_pop_services=pop_service(data,columns)
df_pop_services.sort_values(by='percentage',ascending=False,inplace=True)
df_pop_services.plot(kind='bar',x='Servicio',y='percentage')
plt.show()

# %% [markdown]
# Podemos ver que el servicio más popular es el de las multiples líneas, seguido de las peliculas, televisión y respaldo online.

# %%
def pop_service_1(data,columns):
    list_0=[]
    list_1=[]
    for column in data[columns]:
        percentage_1=100*data[(data[column]=='Yes')&(data['cancel']=='1')]['customer_id'].count()/data.shape[0]
        percentage_0=100*data[(data[column]=='Yes')&(data['cancel']=='0')]['customer_id'].count()/data.shape[0]
        list_1.append(percentage_1)
        list_0.append(percentage_0)
    pop_services={'Servicio':columns,'percentage_0':list_0,'percentage_1':list_1}    
    df_pop_services=pd.DataFrame(pop_services)
    return df_pop_services

# %%
columns=['online_security', 'online_backup', 'device_protection', 'tech_support',
       'streaming_tv', 'streaming_movies', 'multiple_lines']
df_pop_services_1=pop_service_1(data,columns)
df_pop_services_1.sort_values(by='percentage_0',ascending=False,inplace=True)
df_pop_services_1.plot(kind='bar',x='Servicio',y=['percentage_0','percentage_1'])
plt.show()

# %% [markdown]
# Las personas que cancelaron tienen como servicios más frecuentes la multininea, las peliculas, la televisión y la protección de los dispositivos

# %% [markdown]
# De la ultima pregunta, podemos concluir que la mayoría de usuarios que cancelan tienen el plan mes a mes, pagan con cheque electronico, tienen el servicio de internet de fibra optica y los servicios que más utilizan son la multilinea, las peliculas, la televisión y la protección de los dispositivos. La tasa de cancelación es del **32%**.

# %% [markdown]
# ## Plan de trabajo
# 
# 1. Segmentar los datos
# 
# 2. Balanceo de datos
# 
# 3. Separación de datos de entrenamiento, validación y testeo
# 
# 4. Entrenamiento de modelo de clasificación
# 
# 5. Testeo del modelo de clasificación
# 
# 
# 
# Primero debemos filtrar los datos por el tipo de contrato de más frecuente entre los que van a cancelar, en este caso el de 'mes a mes', posteriormente realizamos todo el proceso de entrenamiento del modelo de entrenamiento que nos permita clasificar a los clientes que puedan cancelar. 
# 
# Para realizar el algoritmo debemos balancear los datos, debido a que el 32% de los datos son clientes que cancelan, posteriormente entrenaremos y probaremos varios algoritmos de clasificación con potenciación del gradiente para obtener una exactitud y calidad del modelo alta. Posteriormente probaremos el mejor modelo con los datos de testeo para determinar la **tasa de cancelación**. 


