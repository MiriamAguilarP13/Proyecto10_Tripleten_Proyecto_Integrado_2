# %% [markdown]
# # Proyecto Integrado II
# 
# # Descripción del proyecto  
# 
# Trabajas en una empresa emergente que vende productos alimenticios. Debes investigar el comportamiento del usuario para la aplicación de la empresa.  
# Primero, estudia el embudo de ventas. Descubre cómo los usuarios y las usuarias llegan a la etapa de compra. ¿Cuántos usuarios o usuarias realmente llegan a esta etapa? ¿Cuántos se atascan en etapas anteriores? ¿Qué etapas en particular?  
# 
# Luego, observa los resultados de un test A/A/B. Al equipo de diseño le gustaría cambiar las fuentes de toda la aplicación, pero la gerencia teme que los usuarios y las usuarias piensen que el nuevo diseño es intimidante. Por ello, deciden tomar una decisión basada en los resultados de un test A/A/B.  
# 
# Los usuarios se dividen en tres grupos: dos grupos de control obtienen las fuentes antiguas y un grupo de prueba obtiene las nuevas. Descubre qué conjunto de fuentes produce mejores resultados.

# %% [markdown]
# # Contenido
# 
# * [Objetivos](#objetivos)
# * [Diccionario de Datos](#diccionario)
# * [Inicialización](#inicio)
# * [Cargar datos](#carga_datos)
# * [Preparar los datos](#preparar_datos)
# * [Estudiar y comprobar los datos](#comprobar)
# * [Estudiar el embudo de eventos](#embudo)
# * [Estudiar los resultados del experimento](#experimento)
#     * [Comparación de grupos control 246 y 247](#grupo_246_247)
#     * [Comparación con el grupo con fuentes alteradas](#grupo_248)
#     * [Comparación de eventos del grupo 246 con el grupo 248](#246_248)
#     * [Comparación de eventos del grupo 247 con el grupo 248](#247_248)
#     * [Combinación de resultados de los grupos 246 y 247 y comparación con el grupo 248](#ctrls_248)
# * [Conclusiones Generales](#end)

# %% [markdown]
# # Objetivos <a id='objetivos'></a>  
# 
# * Obtener una comprensión general de los datos.  
# * Identificar tendencias y patrones importantes.  
# * Preparar los datos para el análisis.   
# * Analizar los resultados.  
# * Determinar si hay diferencia entre el grupo con fuentes alteradas y los grupos control.  

# %% [markdown]
# # Diccionario de Datos <a id='diccionario'></a>  
# 
# Cada entrada de registro es una acción de usuario o un evento.  
# `EventName`: nombre del evento.  
# `DeviceIDHash`: identificador de usuario unívoco.  
# `EventTimestamp`: hora del evento.  
# `ExpId`: número de experimento: 246 y 247 son los grupos de control, 248 es el grupo de prueba.  

# %% [markdown]
# # Inicialización <a id='inicio'></a>

# %%
# Cargar todas las librerías
import pandas as pd
import numpy as np
from scipy import stats as st
import math as mt
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly import graph_objects as go

# %% [markdown]
# # Cargar datos <a id='carga_datos'></a>

# %%
# se guarda el conjunto de datos en un DataFrame
logs_exp_us = pd.read_csv('files/datasets/logs_exp_us.csv', sep= '\t')

# %% [markdown]
# # Preparar los datos <a id='preparar_datos'></a>

# %%
# se imprime la información del DataFrame y las primeras 5 filas
logs_exp_us.info()

# %%
# se imprimen las primeras 5 filas
logs_exp_us.head()

# %%
# se cambia el tipo de dato de la columna 'EventTimestamp' a fecha con to_datetime() con unit='s' (segundos)
logs_exp_us['EventTimestamp'] = pd.to_datetime(logs_exp_us['EventTimestamp'], unit='s')

# %%
# se imprimen de nuevo las primeras 5 filas para ver los cambios en la columna 'EventTimestamp'
logs_exp_us.head()

# %%
# con el atributo dtypes se compruban los tipos de datos de las columnas nuevamente
logs_exp_us.dtypes

# %%
# Se cambian los nombres de todas las columnas con `rename()`
logs_exp_us = logs_exp_us.rename(columns= {'EventName':'event_name', 
                                           'DeviceIDHash': 'user_id', 
                                           'EventTimestamp':'date_hr', 
                                           'ExpId':'experiment_id'})

# %%
# se comprueban los nombres de las columnas imprimiendo las 3 primeras filas
logs_exp_us.head(3)

# %% [markdown]
# <div style="background-color: lightyellow; padding: 10px;">
# 
# <span style="color: darkblue;">  
#     
# Se buscan valores ausentes con `isna().sum()` y los valores duplicados con `duplicated().sum()`, en caso de haberlos se procesan dichos datos.
#     
# </span>
#     
# </div>

# %%
# se buscan los valores ausentes
logs_exp_us.isna().sum()

# %%
# se buscan los valores duplicados
logs_exp_us.duplicated().sum()

# %%
# Se eliminan los valores duplicados
# Ahora estos 413 valores duplicados se eliminarán con el método drop_duplicates con reset_index(drop=True) 
# para eliminar los duplicados y reiniciar los índices
logs_exp_us = logs_exp_us.drop_duplicates().reset_index(drop= True)

# %%
# se verifican nuevamente los valores duplicados
# se buscan los valores duplicados
logs_exp_us.duplicated().sum()

# %%
# se crea una columna para alamcenar solo la fecha de los eventos
logs_exp_us['date'] = logs_exp_us['date_hr'].dt.date

# se cambia el tipo de dato de la columna 'date' a fecha
logs_exp_us['date'] = pd.to_datetime(logs_exp_us['date'])
logs_exp_us.head()

# %% [markdown]
# # Estudiar y comprobar los datos <a id='comprobar'></a>

# %%
# se buscan los eventos que hay en los registros
# con value_counts() que hace recuentos de valores únicos
logs_exp_us['event_name'].value_counts()

# %% [markdown]
# <div style="background-color: lightyellow; padding: 10px;">
# 
# <span style="color: darkblue;">  
#     
# **Observaciones:**  
# Se tienen 4 diferentes tipos de eventos y cada uno tiene diferentes recuentos de valores únicos, como se observa en la celda anterior.  
#     
# </span>
#     
# </div>

# %%
# se contabilizan los usuarios únicos que hay en los regitros con nunique()
logs_exp_us['user_id'].nunique()

# %% [markdown]
# <div style="background-color: lightyellow; padding: 10px;">
# 
# <span style="color: darkblue;">  
#     
# **Observaciones:**  
# Se tienen un total de 7551 usuarios y usuarias en los registros.    
#     
# </span>
#     
# </div>

# %%
# se calculan el promedio de eventos por usuario/a a partir del DataFrame 'logs_exp_us'
# se contabilizan los eventos con count() y con mean() se calcula el promedio
# el resultado se guarda en 'average_events_per_user'
average_events_per_user = logs_exp_us.groupby('user_id')['event_name'].count().mean() 

# se calcula la mediana de los eventos por usuario/a a partir del DataFrame 'logs_exp_us'
median_events_per_user = logs_exp_us.groupby('user_id')['event_name'].count().median()

print(f'Promedio de eventos por usuario: {round(average_events_per_user)}')
print(f'La mediana de eventos por usuario: {round(median_events_per_user)}')

# %%
# se encuentra la fecha mínima y máxima de los datos
date_min = logs_exp_us['date_hr'].min()
date_max = logs_exp_us['date_hr'].max()

print(f'La fecha mínima es el {date_min}\nLa fecha máxima es el {date_max}')

# %% [markdown]
# <div style="background-color: lightyellow; padding: 10px;">
# 
# <span style="color: darkblue;">  
#     
# **Observaciones:**  
# En promedio los usuarios y usuarias tienen en promedio 32 eventos, sin embargo, este dato puede no ser tan preciso ya pueden haber valores extremos que influyan en el cálculo del promedio. Lo anterior se puede observar con el valor de la mediana que es 20, ésta métrica se ve menos afectada por la existencia de valores atípicos.  
#     
# La fecha mínima es el 2019-07-25 a las 04:43:36 y la fecha máxima es el 2019-08-07 a las 21:15:17.  
# </span>
#     
# </div>

# %%
# se traza un histograma por fecha y hora

plt.figure(figsize=(14, 10))

sns.histplot(data= logs_exp_us, x='date_hr')

# Configurar las propiedades del eje x
plt.xticks(rotation=45, ha='right')  # se ajusta la inclinación y alineación de las etiquetas
plt.xlabel('Fecha y Hora')  # se etablece el título del eje x

plt.show()

# %% [markdown]
# <div style="background-color: lightyellow; padding: 10px;">
# 
# <span style="color: darkblue;">  
#     
# **Observaciones:**  
# La fecha mínima es el 2019-07-25 a las 04:43:36, no obstante, en el histograma se observa que no se tienen datos igualmente completos para todo el periodo. Los datos comienzan a verse completos a partir del 2019-08-01, entonces el perido donde los datos se comienzan a ver completos es del 2019-08-01 a 2019-08-07, los datos del mes de agisto. Por lo tanto, se van a ignorar los datos previos al 2019-08-01.  
# 
# 
# </span>
#     
# </div>

# %%
# se filtra el DataFrame 'logs_exp_us' tomando en cuenta los datos a partir del 2019-08-01, se reinicia el índice
# el DataFrame resultante se guarda en 'df_complete_period'
df_complete_period = logs_exp_us[logs_exp_us['date_hr'] >= '2019-08-01'].reset_index(drop= True)
df_complete_period.head()

# %%
# se calcula los datos que se excluyeron más antiguos
pct_excluded = (logs_exp_us.shape[0] - df_complete_period.shape[0]) / logs_exp_us.shape[0] * 100

# se calculan los usuarios que quedaron después de filtrar el DataFrame
total_users = df_complete_period['user_id'].nunique()

print(f'Porcentaje de datos excluidos {round(pct_excluded, 1)} %')
print(f'Total de usuarios y usuarias en los registros: {total_users}')

# %% [markdown]
# <div style="background-color: lightyellow; padding: 10px;">
# 
# <span style="color: darkblue;">  
#     
# **Observaciones:**  
# El porcentaje de eventos y usuarios que se excluyeron representan únicamente el 1.2 % del total y un total de usuarios y usuarias en los registros de 7534.  
# 
# 
# </span>
#     
# </div>

# %%
# se corrobora que se tengan usuarios y usuarias de los tres grupos experimentales
df_complete_period.groupby('experiment_id')['event_name'].count()

# %% [markdown]
# # Estudiar el embudo de eventos <a id='embudo'></a>

# %%
# se buscan los eventos que hay en los registros y su frecuencia de suceso, se realiza con value_counts()
# se reincia el índice
event_frequencies = df_complete_period['event_name'].value_counts().reset_index()
# se cambia el nombre de las columnas 
event_frequencies.columns = ['event_name', 'frequency']

# se imprime el DataFrame 'event_frequencies'
event_frequencies

# %%
# se encuentra la cantidad de usuarios y usuarias que realizaron cada una de las acciones (los eventos)
# se emplea nunique() para contbilizar los usuarios/as únicos/as, reset_index() para reiniciar el índice y 
# sort_values() para ordenar los valores de mayor a menor
users_event = df_complete_period.groupby('event_name')['user_id'].nunique().reset_index()

# se cambia el nombre de las columnas 
users_event.columns = ['event_name', 'users']

# se ordena por la columna 'users' de mayor a menor
users_event = users_event.sort_values(by= 'users', ascending= False)

# se imprime el DataFrame 'users_per_event'
users_event

# %%
# se calcula la proporción de usuarios y usuarias que realizaron la acción al menos una vez
# se crea la columna 'proportion' en el DataFrame 'users_event'
# se divide la columna 'users' entre el total de usuarios que es la variable 'total_users'
users_event['proportion'] = users_event['users'] / total_users

# se reinicia el índice
users_event = users_event.reset_index(drop= True)
# se imprime el DataFrame 'users_event'
users_event

# %% [markdown]
# <div style="background-color: lightyellow; padding: 10px;">
# 
# <span style="color: darkblue;">  
#     
# **Observaciones:**  
# El orden de cómo sucedieron los eventos probablemente se da de la siguiente manera: Aparece la pantalla principal -> Tutorial -> Aparece la pantalla de ofertas -> Aparece la pantalla del carrito -> Pantalla de pago exitosa.  
# 
# 
# </span>
#     
# </div>

# %%
# se realiza un gráfico de embudo con plotly.graph_objects
fig1 = go.Figure(go.Funnel(
    y = users_event['event_name'],
    x = users_event['users']))

fig1.show()

# %% [markdown]
# <div style="background-color: lightyellow; padding: 10px;">
# 
# <span style="color: darkblue;">  
#     
# **Observaciones:**  
# Para hacer el gráfico de embudo el orden de los eventos no se tomaron en cuenta como si fueran parte dde una secuencia. El evento de tutorial es dónde se pierden más usuarios y usuarias, mientras que, el porcentaje de usuarios y usuarias que hace todo el viaje desde su primer evento hasta el pago son el 47.7%.  
# 
# 
# </span>
#     
# </div>

# %% [markdown]
# # Estudiar los resultados del experimento <a id='experimento'></a>

# %% [markdown]
# # Comparación de grupos control 246 y 247 <a id='grupo_246_247'></a>

# %%
# se calcula la cantidad de usuarios y usuarias que tengan los tres grupos experimentales
df_complete_period.groupby('experiment_id')['user_id'].nunique()

# %%
# se filtra el DataFrame 'df_complete_period' para el experimento 246 y 247, que son los dos grupos de control en el test A/A
group_246 = df_complete_period[df_complete_period['experiment_id'] == 246]
group_247 = df_complete_period[df_complete_period['experiment_id'] == 247]

# %%
# se contabilizan la cantidad de eventos por usuario/a en cada grupo de experimento
# se hace con groupby() y se cuantan los eventos de la columna 'event_name' con count()
# se reinician los índices
users_events_246 = group_246.groupby('user_id')['event_name'].count().reset_index()

# se cambian los nombres de las columnas
users_events_246.columns = ['user_id', 'count_events']

# se imprime las 5 primeras filas
users_events_246.head()

# %%
# se repiten los mismos pasos de la celda anterior pero para el grupo/experimento 247
# se hace con groupby() y se cuantan los eventos de la columna 'event_name' con count()
# se reinician los índices
users_events_247 = group_247.groupby('user_id')['event_name'].count().reset_index()

# se cambian los nombres de las columnas
users_events_247.columns = ['user_id', 'count_events']

# se imprime las 5 primeras filas
users_events_247.head()

# %%
# se establece el valor de alpha en 0.05
alpha = 0.05

# Realizar la prueba de Mann-Whitney de las dos muestras de A con la  función 'st.mannwhitneyu()'
results_A_A = st.mannwhitneyu(users_events_246['count_events'], users_events_247['count_events'])

print('El valor p es:', results_A_A.pvalue)

if results_A_A.pvalue < alpha:
    print('Se rechaza la hipótesis nula, hay diferencia entre los dos grupos')
else:
    print('No se rechaza la hipótesis nula, no hay diferencia entre los dos grupos')


# %% [markdown]
# <div style="background-color: lightyellow; padding: 10px;">
# 
# <span style="color: darkblue;">  
#     
# **Observaciones:**  
# Para los tres grupos de experimentos 246, 247 y 248, tienen una cantidad de usuarios y usuarias de 2484, 2513 y 2537, respectivamente. De acuerdo al resultado de la prueba estadística `scipy.stats.mannwhitneyu()`, no se rechaza la hipótesis nula, por tanto no hay una diferencia entre los dos grupos 246 y 247, que son los dos grupos de control en el test A/A.  
# 
# 
# </span>
#     
# </div>

# %%
# ahora se busca el evento más popular para cada grupo con groupby() y nunique()
# primero para el grupo 246
group_246.groupby('event_name')['user_id'].nunique().sort_values(ascending= False)

# %%
# ahora se busca el evento más popular para el grupo 247
group_247.groupby('event_name')['user_id'].nunique().sort_values(ascending= False)

# %% [markdown]
# <div style="background-color: lightyellow; padding: 10px;">
# 
# <span style="color: darkblue;">  
#     
# **Observaciones:**  
# Para ambos grupos el evento más popular es **Main Screen Appear**.
# 
# 
# </span>
#     
# </div>

# %%
# se calcula la cantidad de usuarios para el evento 'MainScreenAppear' del grupo 246
users_MainScreenAppear_246 = group_246[group_246['event_name'] == 'MainScreenAppear']['user_id'].nunique()

# se calcula la cantidad de usuarios totales para el grupo 246
total_users_246 = group_246['user_id'].nunique()

print(f'Usuarios para el evento Main Screen Appear del grupo 246: {users_MainScreenAppear_246}')
print(f'Usuarios totales para el grupo 246: {total_users_246}')
print(f'Proporción para el evento más popular Main Screen Appear en el grupo 246: {round(users_MainScreenAppear_246 / total_users_246, 3)}')

# %%
# se calcula la cantidad de usuarios para el evento 'MainScreenAppear' del grupo 247
users_MainScreenAppear_247 = group_247[group_247['event_name'] == 'MainScreenAppear']['user_id'].nunique()

# se calcula la cantidad de usuarios totales para el grupo 246
total_users_247 = group_247['user_id'].nunique()

print(f'Usuarios para el evento Main Screen Appear del grupo 247: {users_MainScreenAppear_247}')
print(f'Usuarios totales para el grupo 247: {total_users_247}')
print(f'Proporción para el evento más popular Main Screen Appear en el grupo 247: {round(users_MainScreenAppear_247 / total_users_247, 3)}')

# %% [markdown]
# <div style="background-color: lightyellow; padding: 10px;">
# 
# <span style="color: darkblue;">  
#     
# ****  
# La proporción de usuarios y usuarias que realizaron la acción/evento de Main Screen Appear son para el grupo 246 de 0.986 y para el grupo 247 0.985.  
#  
# Se crea una función que calcule la proporción para cada uno de los eventos en cada grupo y con un bucle `for` se itera sobre cada uno de los eventos del DataFrame de interés.   
# 
# 
# </span>
#     
# </div>

# %%
# se crea la función proportion()
def proportion(df, event):
    users_events = df[df['event_name'] == event]['user_id'].nunique()
    total_users = df['user_id'].nunique()
    proportion = round(users_events / total_users, 3)
    return proportion

# %%
# con un bucle for se calcula la proporción para cada evento en el grupo 246
# se crea una lista con los nombres de los eventos de interés ordenados en de mayor a menor de acuerdo a su popularidad

events = ['MainScreenAppear', 'OffersScreenAppear', 'CartScreenAppear', 'PaymentScreenSuccessful', 'Tutorial']

for event in events:
    result_proportion = proportion(group_246, event) # se emplea la función proportion()
    print(f'La proporción del evento {event} es: {result_proportion}')

# %%
# con un bucle for se calcula la proporción para cada evento en el grupo 247
for event in events:
    result_proportion = proportion(group_247, event) # se emplea la función proportion()
    print(f'La proporción del evento {event} es: {result_proportion}')

# %% [markdown]
# <div style="background-color: lightyellow; padding: 10px;">
# 
# <span style="color: darkblue;">  
#     
# ****  
# Se comprueba si la diferencia entre las proporciones de los grupos es estadísticamente significativa.
# 
# 
# </span>
#     
# </div>

# %%
# se crea una función para calcular la proporción por evento
def proportion_by_event(df_group):
    '''
    Función que calcula la proporción para todos los eventos de interés y retorna una tupla con los resultados.
    '''
    # Lista de los eventos de interés
    events_of_interest = ['MainScreenAppear', 'OffersScreenAppear', 'CartScreenAppear', 'PaymentScreenSuccessful', 'Tutorial']
    
    # se crea un diccionario, cada clave del diccionario es un evento de interés, y el valor asociado 
    # es la proporción calculada utilizando la función 'proportion' 
    proportions_dict = {event: proportion(df_group, event) for event in events_of_interest}
    
    # La función retorna una tupla que contiene las proporciones calculadas para cada evento en el 
    # mismo orden en que aparecen en la lista 'events_of_interest'
    return tuple(proportions_dict[event] for event in events_of_interest)


# %% [markdown]
# <div style="background-color: lightyellow; padding: 10px;">
# 
# <span style="color: darkblue;">  
#     
# ****  
# Por simplicidad el nombre de los eventos de interés se abrevian de la siguiente manera:
# 
# | Nombre del Evento        | Abreviación |
# |--------------------------|--------------|
# | MainScreenAppear         |     msa      |
# | OffersScreenAppear       |     osa      |
# | CartScreenAppear         |     csa      |
# | PaymentScreenSuccessful  |     pss      |
# | Tutorial                 |     tut      |
# 
# </span>
#     
# </div>

# %%
# Llamada a la función 'proportion_by_event' con el grupo 246
p_msa_246, p_osa_246, p_csa_246, p_pss_246, p_tut_246 = proportion_by_event(group_246)

# %%
# Llamada a la función 'proportion_by_event' con el grupo 247
p_msa_247, p_osa_247, p_csa_247, p_pss_247, p_tut_247 = proportion_by_event(group_247)

# %%
# se crea la función 'proportion_combined'
def proportion_combined(df_group_1, df_group_2):
    '''
    Función que calcula las proporciones combinadas de usuarios que realizaron eventos específicos en dos grupos dados
    '''
    
    events_of_interest = ['MainScreenAppear', 'OffersScreenAppear', 'CartScreenAppear', 'PaymentScreenSuccessful', 'Tutorial']

    proportions_combined = {}

    for event in events_of_interest:
        # usuarios/as combinados de ambos grupos por evento
        users_comb_event = df_group_1[df_group_1['event_name'] == event]['user_id'].nunique() + df_group_2[df_group_2['event_name'] == event]['user_id'].nunique()
        # usuarios/as totales de ambos grupos
        total_users_comb = df_group_1['user_id'].nunique() + df_group_2['user_id'].nunique()
        # el resultado se almacena en el diccionario 'proportions_combined'
        proportions_combined[event] = round(users_comb_event / total_users_comb, 3)

    return proportions_combined

# %%
# se llama a la función 'proportion_combined' con los grupos 246 y 247
proportions_combined = proportion_combined(group_246, group_247)

# %%
# se imprimen los valores de las propociones comibinadas con un bucle for
for key, value in proportions_combined.items():
    print(f'{key}: {value}')

# %%
# se guardan las proporciones combinadas de cada evento en una variable diferente
p_msa_comb= proportions_combined['MainScreenAppear']
p_osa_comb= proportions_combined['OffersScreenAppear']
p_csa_comb= proportions_combined['CartScreenAppear']
p_pss_comb= proportions_combined['PaymentScreenSuccessful']
p_tut_comb= proportions_combined['Tutorial']

# se gurdan los valores de las proporciones en una lista
p_list_comb_246_247 = [p_msa_comb, p_osa_comb, p_csa_comb, p_pss_comb, p_tut_comb]
p_list_comb_246_247

# %%
# se calcula la diferencia entre las proporciones de los dos grupos 246 y 247
difference_msa = p_msa_246 - p_msa_247
difference_osa = p_osa_246 - p_osa_247
difference_csa = p_csa_246 - p_csa_247
difference_pss = p_pss_246 - p_pss_247
difference_tut = p_tut_246 - p_tut_247 

# se guardan los valores de las diferencias en una lista
diff_list_246_247 = [difference_msa, difference_osa, difference_csa, difference_pss, difference_tut]
print(diff_list_246_247)

# %% [markdown]
# <div style="background-color: lightyellow; padding: 10px;">
# 
# <span style="color: darkblue;">  
#     
# *******
# Se crea una función para probar la hipótesis de que las proporciones son iguales o no. 
# 
# </span>
#     
# </div>

# %%
# se cre una función para calcular el valor p y probar la hipótesis
def test_proportions_difference(difference, p_event_combined, df_group1, df_group2, event, alpha= 0.05): # alpha se establece en 0.05 dentro de la función
    '''
     Función para probar la hipótesis de que las proporciones son iguales o no.
     difference: la diferencia entre las proporciones de los datasets
     p_event_combined: proporción de éxito en los dataset unidos
     df_group1: dataset de interés 1
     df_group2: dataset de interés 2
     event: evento/acción de interés
     alpha: valor de alfa para la prueba
    '''
    
    # se calcula la estadística en desviaciones estándar de la distribución normal estándar
    z_value = difference / mt.sqrt(p_event_combined * (1 - p_event_combined) * (1/df_group1[df_group1['event_name'] == event]['user_id'].nunique() + 1/df_group2[df_group2['event_name'] == event]['user_id'].nunique()))
    
    # se establece la distribución normal estándar (media 0, desviación estándar 1)
    distr = st.norm(0, 1)
    
    p_value = (1 - distr.cdf(abs(z_value))) * 2
    
    #result = {'p_value': p_value, 'reject_null': p_value < alpha}
    
    return p_value

# %%
# se crea un diccionario combinando las listas 'diff_list_246_247' y 'p_list_comb_246_247' utilizando zip()
# se emplea list() para conevertir el resultado de zip en una lista. 
# Cada elemento de la lista es una tupla que contiene un elemento de diff_list_246_247 y un elemento correspondiente de p_list_comb_246_247

results_diff_comb = list(zip(diff_list_246_247, p_list_comb_246_247))
print(results_diff_comb)

# %% [markdown]
# <div style="background-color: lightyellow; padding: 10px;">
# 
# <span style="color: darkblue;">  
#     
# *******
# La variable `results_diff_comb` es una lista que contiene pares de valores (como tuplas), cada uno de los pares de valores corresponde a uno de los eventos en el siguiente orden: 'MainScreenAppear', 'OffersScreenAppear', 'CartScreenAppear', 'PaymentScreenSuccessful', 'Tutorial'. Para guardar y asignar cada uno de estos pares de valores a su correspondiente evento/acción se hará creando un diccionario, donde las claves del diccionario serán los nombres de los eventos.  
# 
# </span>
#     
# </div>

# %%
# se crea el diccionario vacio
dict_events_diff_comb = {}

# se crea un diccionario de mapeo evento a índice
events_index_map = {
    'MainScreenAppear': 0,
    'OffersScreenAppear': 1,
    'CartScreenAppear': 2,
    'PaymentScreenSuccessful': 3,
    'Tutorial': 4
}

# se usa el diccionario de mapeo para asignar los valores de 'results_diff_comb' a cada evento
for event in events:
    dict_events_diff_comb[event] = results_diff_comb[events_index_map[event]]

print(dict_events_diff_comb)

# %% [markdown]
# <div style="background-color: lightyellow; padding: 10px;">
# 
# <span style="color: darkblue;">  
#     
# *******
# Ya se tienen la lista de los eventos de interés y el diccionario con los resultados de cada evento, ahora se llama a la función `test_proportions_difference` para hacer la prueba y determinar si hay o no diferencias entre las proporciones.   
# 
# </span>
#     
# </div>

# %%
# se establece el valor de alpha
alpha= 0.05

# con un bucle for se recorre la lista de los eventos de interés
for event in events:
    
    # se almacena el resultado de la prueba en results
    results = test_proportions_difference(dict_events_diff_comb[event][0], dict_events_diff_comb[event][1], group_246, group_247, event, alpha)
    
    print(f'Resultados de la prueba para el evento: {event}')        
    print('p-value: ', results)

    if results < alpha:
        print("Rechazar la hipótesis nula: hay una diferencia significativa entre las proporciones")
    else:
        print("No se pudo rechazar la hipótesis nula: no hay razón para pensar que las proporciones son diferentes")
    print()

# %% [markdown]
# <div style="background-color: lightyellow; padding: 10px;">
# 
# <span style="color: darkblue;">  
#     
# **Observaciones:**  
# Con base en los resultados de la prueba en ninguno de los eventos no hay una diferencia en las proporciones. Por lo tanto, se puede decir que los grupos se dividieron correctamente.  
# 
# </span>
#     
# </div>

# %% [markdown]
# # Comparación con el grupo con fuentes alteradas <a id='grupo_248'></a>

# %% [markdown]
# <div style="background-color: lightyellow; padding: 10px;">
# 
# <span style="color: darkblue;">  
#     
# ****  
# Se comparan los grupos control 246 y 247 con el grupo con fuentes alteradas 248. Los resultados se comparan con los de cada uno de los grupos de control para cada evento de forma aislada.
# 
# </span>
#     
# </div>

# %%
# se filtra el DataFrame 'df_complete_period' para el experimento 248, que es el grupo con fuentes alteradas
group_248 = df_complete_period[df_complete_period['experiment_id'] == 248]

# %%
# se contabilizan la cantidad de eventos por usuario/a en el grupo 248
# se hace con groupby() y se cuentan los eventos de la columna 'event_name' con count()
# se reinician los índices
users_events_248 = group_248.groupby('user_id')['event_name'].count().reset_index()

# se cambian los nombres de las columnas
users_events_248.columns = ['user_id', 'count_events']

# se imprime las 5 primeras filas
users_events_248.head()

# %%
# se compara el grupo control 246 con el grupo con fuentes alteradas 248
# se establece el valor de alpha en 0.05
alpha = 0.05

# Realizar la prueba de Mann-Whitney de las muestras de del grupo 246 y el grupo 248 con la  función 'st.mannwhitneyu()'
results_246_248 = st.mannwhitneyu(users_events_246['count_events'], users_events_248['count_events'])

print('El valor p es:', results_246_248.pvalue)

if results_246_248.pvalue < alpha:
    print('Se rechaza la hipótesis nula, hay diferencia entre los dos grupos')
else:
    print('No se rechaza la hipótesis nula, no hay diferencia entre los dos grupos')

# %%
# se compara el grupo control 247 con el grupo con fuentes alteradas 248
# se establece el valor de alpha en 0.05
alpha = 0.05

# Realizar la prueba de Mann-Whitney de las muestras de del grupo 246 y el grupo 248 con la  función 'st.mannwhitneyu()'
results_247_248 = st.mannwhitneyu(users_events_247['count_events'], users_events_248['count_events'])

print('El valor p es:', results_247_248.pvalue)

if results_246_248.pvalue < alpha:
    print('Se rechaza la hipótesis nula, hay diferencia entre los dos grupos')
else:
    print('No se rechaza la hipótesis nula, no hay diferencia entre los dos grupos')

# %%
# ahora se busca el evento más popular para el grupo 248
group_248.groupby('event_name')['user_id'].nunique().sort_values(ascending= False)

# %% [markdown]
# <div style="background-color: lightyellow; padding: 10px;">
# 
# <span style="color: darkblue;">  
#     
# **Observaciones:**  
# De acuerdo al resultado de la prueba estadística `scipy.stats.mannwhitneyu()`, no se rechaza la hipótesis nula, por tanto no hay una diferencia entre los dos grupos 246 y 248, tampoco hay una diferencia entre el grupo 247 y 248. Por lo tanto, no hay diferencia entre el grupo con fuentes alteradas y los grupos control. El evento más popular para el grupo 248 fue MainScreenAppear, el mismo evento/acción que para los grupos control.  
# 
# 
# </span>
#     
# </div>

# %% [markdown]
# <div style="background-color: lightyellow; padding: 10px;">
# 
# <span style="color: darkblue;">  
#     
# ****  
# Ahora para el grupo con fuentes alteradas se comparan los resultados con los de cada uno de los grupos de control para cada evento de forma aislada.
# 
# 
# </span>
#     
# </div>

# %% [markdown]
# # Comparación de eventos del grupo 246 con el grupo 248 <a id='246_248'></a>

# %%
# Llamada a la función 'proportion_by_event' con el grupo 248
p_msa_248, p_osa_248, p_csa_248, p_pss_248, p_tut_248 = proportion_by_event(group_248)

# %%
# se llama a la función 'proportion_combined' con los grupos 246 y 248
proportions_combined_246_248 = proportion_combined(group_246, group_248)
# se imprimen los resultados
proportions_combined_246_248

# %%
# se guardan las proporciones combinadas de cada evento en una variable diferente
p_msa_comb_246_248= proportions_combined_246_248['MainScreenAppear']
p_osa_comb_246_248= proportions_combined_246_248['OffersScreenAppear']
p_csa_comb_246_248= proportions_combined_246_248['CartScreenAppear']
p_pss_comb_246_248= proportions_combined_246_248['PaymentScreenSuccessful']
p_tut_comb_246_248= proportions_combined_246_248['Tutorial']

# se gurdan los valores de las proporciones en una lista
p_list_comb_246_248 = [p_msa_comb_246_248, p_osa_comb_246_248, p_csa_comb_246_248, p_pss_comb_246_248, p_tut_comb_246_248]
p_list_comb_246_248

# %%
# se calcula la diferencia entre las proporciones de los dos grupos 246 y 248
difference_msa_246_248 = p_msa_246 - p_msa_248
difference_osa_246_248 = p_osa_246 - p_osa_248
difference_csa_246_248 = p_csa_246 - p_csa_248
difference_pss_246_248 = p_pss_246 - p_pss_248
difference_tut_246_248 = p_tut_246 - p_tut_248

# se guardan los valores de las diferencias en una lista
diff_list_246_248 = [difference_msa_246_248, 
                     difference_osa_246_248, 
                     difference_csa_246_248, 
                     difference_pss_246_248, 
                     difference_tut_246_248]
print(diff_list_246_248)

# %%
# se crea un diccionario combinando las listas 'diff_list_246_248' y 'p_list_comb_246_248' utilizando zip()
# se emplea list() para conevertir el resultado de zip en una lista. 
# Cada elemento de la lista es una tupla que contiene un elemento de diff_list_246_248 y un elemento correspondiente de p_list_comb_246_248

results_diff_comb_246_248 = list(zip(diff_list_246_248, p_list_comb_246_248))
print(results_diff_comb_246_248)

# %% [markdown]
# <div style="background-color: lightyellow; padding: 10px;">
# 
# <span style="color: darkblue;">  
#     
# *******
# La variable `results_diff_comb` es una lista que contiene pares de valores (como tuplas), cada uno de los pares de valores corresponde a uno de los eventos en el siguiente orden: 'MainScreenAppear', 'OffersScreenAppear', 'CartScreenAppear', 'PaymentScreenSuccessful', 'Tutorial'. Para guardar y asignar cada uno de estos pares de valores a su correspondiente evento/acción se hará creando un diccionario, donde las claves del diccionario serán los nombres de los eventos.  
# 
# </span>
#     
# </div>

# %%
# se crea el diccionario vacio
dict_events_diff_comb_246_248 = {}

# se crea un diccionario de mapeo evento a índice
events_index_map = {
    'MainScreenAppear': 0,
    'OffersScreenAppear': 1,
    'CartScreenAppear': 2,
    'PaymentScreenSuccessful': 3,
    'Tutorial': 4
}

# se usa el diccionario de mapeo para asignar los valores de 'results_diff_comb_246_248' a cada evento
for event in events:
    dict_events_diff_comb_246_248[event] = results_diff_comb_246_248[events_index_map[event]]

print(dict_events_diff_comb_246_248)

# %%
# se llama a la función 'test_proportions_difference' 
# se establece el valor de alpha
alpha= 0.05

# con un bucle for se recorre la lista de los eventos de interés
for event in events:
    
    # se almacena el resultado de la prueba en results
    results = test_proportions_difference(dict_events_diff_comb_246_248[event][0], dict_events_diff_comb_246_248[event][1], group_246, group_248, event, alpha)
    
    print(f'Resultados de la prueba para el evento: {event}')        
    print('p-value: ', results)

    if results < alpha:
        print("Rechazar la hipótesis nula: hay una diferencia significativa entre las proporciones")
    else:
        print("No se pudo rechazar la hipótesis nula: no hay razón para pensar que las proporciones son diferentes")
    print()

# %% [markdown]
# # Comparación de eventos del grupo 247 con el grupo 248 <a id='247_248'></a>

# %%
# se llama a la función 'proportion_combined' con los grupos 247 y 248
proportions_combined_247_248 = proportion_combined(group_247, group_248)
# se imprimen los resultados
proportions_combined_247_248

# %%
# se guardan las proporciones combinadas de cada evento en una variable diferente
p_msa_comb_247_248= proportions_combined_247_248['MainScreenAppear']
p_osa_comb_247_248= proportions_combined_247_248['OffersScreenAppear']
p_csa_comb_247_248= proportions_combined_247_248['CartScreenAppear']
p_pss_comb_247_248= proportions_combined_247_248['PaymentScreenSuccessful']
p_tut_comb_247_248= proportions_combined_247_248['Tutorial']

# se gurdan los valores de las proporciones en una lista
p_list_comb_247_248 = [p_msa_comb_247_248, 
                       p_osa_comb_247_248, 
                       p_csa_comb_247_248, 
                       p_pss_comb_247_248, 
                       p_tut_comb_247_248]
p_list_comb_247_248

# %%
# se calcula la diferencia entre las proporciones de los dos grupos 247 y 248
difference_msa_247_248 = p_msa_247 - p_msa_248
difference_osa_247_248 = p_osa_247 - p_osa_248
difference_csa_247_248 = p_csa_247 - p_csa_248
difference_pss_247_248 = p_pss_247 - p_pss_248
difference_tut_247_248 = p_tut_247 - p_tut_248

# se guardan los valores de las diferencias en una lista
diff_list_247_248 = [difference_msa_247_248, 
                     difference_osa_247_248, 
                     difference_csa_247_248, 
                     difference_pss_247_248, 
                     difference_tut_247_248]
print(diff_list_247_248)

# %%
# se crea un diccionario combinando las listas 'diff_list_247_248' y 'p_list_comb_247_248' utilizando zip()
# se emplea list() para conevertir el resultado de zip en una lista. 
# Cada elemento de la lista es una tupla que contiene un elemento de diff_list_247_248 y un elemento correspondiente de p_list_comb_247_248

results_diff_comb_247_248 = list(zip(diff_list_247_248, p_list_comb_247_248))
print(results_diff_comb_247_248)

# %%
# se crea el diccionario vacio
dict_events_diff_comb_247_248 = {}

# se crea un diccionario de mapeo evento a índice
events_index_map = {
    'MainScreenAppear': 0,
    'OffersScreenAppear': 1,
    'CartScreenAppear': 2,
    'PaymentScreenSuccessful': 3,
    'Tutorial': 4
}

# se usa el diccionario de mapeo para asignar los valores de 'results_diff_comb_246_248' a cada evento
for event in events:
    dict_events_diff_comb_247_248[event] = results_diff_comb_247_248[events_index_map[event]]

print(dict_events_diff_comb_247_248)

# %%
# se llama a la función 'test_proportions_difference' 
# se establece el valor de alpha
alpha= 0.05

# con un bucle for se recorre la lista de los eventos de interés
for event in events:
    
    # se almacena el resultado de la prueba en results
    results = test_proportions_difference(dict_events_diff_comb_247_248[event][0], dict_events_diff_comb_247_248[event][1], group_247, group_248, event, alpha)
    
    print(f'Resultados de la prueba para el evento: {event}')        
    print('p-value: ', results)

    if results < alpha:
        print("Rechazar la hipótesis nula: hay una diferencia significativa entre las proporciones")
    else:
        print("No se pudo rechazar la hipótesis nula: no hay razón para pensar que las proporciones son diferentes")
    print()

# %% [markdown]
# <div style="background-color: lightyellow; padding: 10px;">
# 
# <span style="color: darkblue;">  
#     
# **Observaciones:**  
# De acuerdo al resultado de la prueba estadística para saber si las proporciones son iguales o no; no se rechaza la hipótesis nula, por tanto no hay una diferencia entre los dos grupos 246 y 248 entre las proporciones para cada evento, tampoco hay una diferencia entre las proporciones entre el grupo 247 y 248. Por lo tanto, no hay diferencia entre el grupo con fuentes alteradas y los grupos control.  
# 
# 
# </span>
#     
# </div>

# %% [markdown]
# # Combinación de resultados de los grupos 246 y 247 y comparación con el grupo 248 <a id='ctrls_248'></a>

# %%
# se combinan los resultados de los grupos 246 y 247
# se filtra el DataFrame 'df_complete_period' para el experimento 246 y 247, ambos son grupos control
groups_ctrls = df_complete_period[(df_complete_period['experiment_id'] == 246) | (df_complete_period['experiment_id'] == 247)]
groups_ctrls.head()

# %%
# se contabilizan la cantidad de eventos por usuario/ade de los grupos control
# se hace con groupby() y se cuentan los eventos de la columna 'event_name' con count()
# se reinician los índices
users_events_ctrls = groups_ctrls.groupby('user_id')['event_name'].count().reset_index()

# se cambian los nombres de las columnas
users_events_ctrls.columns = ['user_id', 'count_events']

# se imprime las 5 primeras filas
users_events_ctrls.head()

# %%
# se compara los grupos control con el grupo con fuentes alteradas 248
# se establece el valor de alpha en 0.05
alpha = 0.05

# Realizar la prueba de Mann-Whitney de las muestras de los grupo control y el grupo 248 con la  función 'st.mannwhitneyu()'
results_ctrls_248 = st.mannwhitneyu(users_events_ctrls['count_events'], users_events_248['count_events'])

print('El valor p es:', results_ctrls_248.pvalue)

if results_ctrls_248.pvalue < alpha:
    print('Se rechaza la hipótesis nula, hay diferencia entre los dos grupos')
else:
    print('No se rechaza la hipótesis nula, no hay diferencia entre los dos grupos')

# %% [markdown]
# <div style="background-color: lightyellow; padding: 10px;">
# 
# <span style="color: darkblue;">  
#     
# **Observaciones:**  
# De acuerdo al resultado de la prueba estadística `scipy.stats.mannwhitneyu()`, no se rechaza la hipótesis nula, por tanto no hay una diferencia entre los resultados de los grupos control combinados y el grupo 248 con fuentes alteradas.  
# 
# 
# </span>
#     
# </div>

# %% [markdown]
# <div style="background-color: lightyellow; padding: 10px;">
# 
# <span style="color: darkblue;">  
#     
# ****  
# Ahora para el grupo con fuentes alteradas se comparan los resultados con los resultados combinados de los grupos control para cada evento.
# 
# 
# </span>
#     
# </div>

# %%
# Llamada a la función 'proportion_by_event' con el grupo control combinado
p_msa_ctrls, p_osa_ctrls, p_csa_ctrls, p_pss_ctrls, p_tut_ctrls = proportion_by_event(groups_ctrls)

# %%
# se llama a la función 'proportion_combined' con los grupos control combinados y 248
proportions_combined_ctrls_248 = proportion_combined(groups_ctrls, group_248)
# se imprimen los resultados
proportions_combined_ctrls_248

# %%
# se guardan las proporciones combinadas de cada evento en una variable diferente
p_msa_comb_ctrls_248= proportions_combined_ctrls_248['MainScreenAppear']
p_osa_comb_ctrls_248= proportions_combined_ctrls_248['OffersScreenAppear']
p_csa_comb_ctrls_248= proportions_combined_ctrls_248['CartScreenAppear']
p_pss_comb_ctrls_248= proportions_combined_ctrls_248['PaymentScreenSuccessful']
p_tut_comb_ctrls_248= proportions_combined_ctrls_248['Tutorial']

# se gurdan los valores de las proporciones en una lista
p_list_comb_ctrls_248 = [p_msa_comb_ctrls_248, 
                       p_osa_comb_ctrls_248, 
                       p_csa_comb_ctrls_248, 
                       p_pss_comb_ctrls_248, 
                       p_tut_comb_ctrls_248]
p_list_comb_ctrls_248

# %%
# se calcula la diferencia entre las proporciones de los dos grupos control y 248
difference_msa_ctrls_248 = p_msa_ctrls - p_msa_248
difference_osa_ctrls_248 = p_osa_ctrls - p_osa_248
difference_csa_ctrls_248 = p_csa_ctrls - p_csa_248
difference_pss_ctrls_248 = p_pss_ctrls - p_pss_248
difference_tut_ctrls_248 = p_tut_ctrls - p_tut_248

# se guardan los valores de las diferencias en una lista
diff_list_ctrls_248 = [difference_msa_ctrls_248, 
                     difference_osa_ctrls_248, 
                     difference_csa_ctrls_248, 
                     difference_pss_ctrls_248, 
                     difference_tut_ctrls_248]
print(diff_list_ctrls_248)

# %%
# se crea un diccionario combinando las listas 'diff_list_ctrls_248' y 'p_list_comb_ctrls_248' utilizando zip()
# se emplea list() para conevertir el resultado de zip en una lista. 
# Cada elemento de la lista es una tupla que contiene un elemento de diff_list_ctrls_248 y un elemento correspondiente de p_list_comb_ctrls_248

results_diff_comb_ctrls_248 = list(zip(diff_list_ctrls_248, p_list_comb_ctrls_248))
print(results_diff_comb_ctrls_248)

# %%
# se crea el diccionario vacio
dict_events_diff_comb_ctrls_248 = {}

# se crea un diccionario de mapeo evento a índice
events_index_map = {
    'MainScreenAppear': 0,
    'OffersScreenAppear': 1,
    'CartScreenAppear': 2,
    'PaymentScreenSuccessful': 3,
    'Tutorial': 4
}

# se usa el diccionario de mapeo para asignar los valores de 'results_diff_comb_ctrls_248' a cada evento
for event in events:
    dict_events_diff_comb_ctrls_248[event] = results_diff_comb_ctrls_248[events_index_map[event]]

print(dict_events_diff_comb_ctrls_248)

# %%
# se llama a la función 'test_proportions_difference' 
# se establece el valor de alpha
alpha= 0.05

# con un bucle for se recorre la lista de los eventos de interés
for event in events:
    
    # se almacena el resultado de la prueba en results
    results = test_proportions_difference(dict_events_diff_comb_ctrls_248[event][0], dict_events_diff_comb_ctrls_248[event][1], groups_ctrls, group_248, event, alpha)
    
    print(f'Resultados de la prueba para el evento: {event}')        
    print('p-value: ', results)

    if results < alpha:
        print("Rechazar la hipótesis nula: hay una diferencia significativa entre las proporciones")
    else:
        print("No se pudo rechazar la hipótesis nula: no hay razón para pensar que las proporciones son diferentes")
    print()

# %% [markdown]
# <div style="background-color: lightyellow; padding: 10px;">
# 
# <span style="color: darkblue;">  
#     
# **Observaciones:**  
# De acuerdo al resultado de la prueba estadística para saber si las proporciones son iguales o no; no se rechaza la hipótesis nula en cada uno de los eventos, por tanto no hay una diferencia entre los resultados combinados de los dos grupos control y el grupo 248 con fuentes alteradas entre las proporciones.  
# 
# 
# </span>
#     
# </div>

# %% [markdown]
# # Conclusiones Generales <a id='end'></a>

# %% [markdown]
# <div style="background-color: lightyellow; padding: 10px;">
# 
# <span style="color: darkblue;">  
#       
# **1.** Dado que la mediana es menor que el promedio indica la presencia de valores atípicos en la distribución de los eventos por usuario y usuaria. La mediana es una medida de tendencia central menos sensible a los valores extremos, esto sugiere que la distribución tiene un sesgo.  
#     
# **2.** Se observó que los datos no estaban completos para todo el periodo. La decisión de ignorar los datos anteriores al 2019-08-01 se apoya en la observación de que los datos comienzan a estar completos a partir de esa fecha.  
#     
# **3.** Se excluyeron los datos previos al 2019-08-01 representó un pequeño porcentaje del total de eventos y usuarios (1.2%).  
#     
# **4.** Se propone una secuencia probable de eventos, destacando el orden de ocurrencia, Aparece la pantalla principal -> Tutorial -> Aparece la pantalla de ofertas -> Aparece la pantalla del carrito -> Pantalla de pago exitosa.  
#     
# **5.** Se observa que el evento de tutorial tiene una pérdida significativa de usuarios del 88.7 % desde el evento MainScreenAppear. Además, menos del 50% de los usuarios completan todo el viaje desde la aparición de la pantalla principal hasta el pago.  
#     
# **6.** En ambos grupos control y el grupo con fuentes alteradas, el evento más popular es "Main Screen Appear". Esta consistencia indica que la manipulación en el grupo 248 no ha afectado considerablemente la popularidad de este evento.  
#     
# **7.** De acuerdo a la prueba estadística de Wilcoxon-Mann-Whitney, no hay una diferencia significativa para rechazar la hipótesis nula entre los grupos 246 y 247, 246 y 248 ni entre los grupos 247 y 248. Lo cual sugiere que no hay diferencias significativas entre los grupos control y el grupo con fuentes alteradas.  
#     
# **8.** Las pruebas estadísticas no mustran diferencias significativas en las proporciones de cada evento entre los dos grupos control 246 y 247, tampoco hay diferencias entre el grupo 246 y el grupo 248 ni entre el grupo 247 y 248. Esto indica que, en términos relativos, la frecuencia de eventos se mantiene similar entre los grupos, no afectan las fuentes alteradas.  
#     
# **9.** Las pruebas no revelan diferencias significativas entre los resultados combinados de los dos grupos control y el grupo 248 con fuentes alteradas en términos de eventos y proporciones.  
#     
# **10.** Las fuentes alteradas en el grupo 248 no ha tenido un impacto significativo en comparación con los grupos control. Los análisis estadísticos respaldan la conclusión de que no hay diferencias significativas en términos de eventos y proporciones.  
#     
# **11.** Aproximadamente se hicieron 23 pruebas de hipótesis. Se estableció un valor de alpha de 0.05, si se hubiera establecido un nivel de significancia de 0.1, este es más tolerante que uno de 0.05, lo cual implica una mayor disposición a aceptar resultados significativos incluso si existe una mayor posibilidad de que sean falsos.  
# 
# 
# </span>
#     
# </div>

# %%



