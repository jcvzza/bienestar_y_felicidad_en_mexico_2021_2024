# %% [markdown]
# # Agrupamiento de datos

# %% [markdown]
# ### Carga de librerias

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# %% [markdown]
# ### Carga de datos

# %%
df_biare = pd.read_csv("../Limpieza de datos (Practica 1)/biare_limpio_2021_2024.csv")

# %% [markdown]
# ### Separación y preparación de datos 

# %% [markdown]
# Utilizaremos las preguntas de cb_P5, ya que son las que hacen referencia directa a la satisfaccion con la vida

# %%
#hacemos una lista con las columnas que queremos usar para hacer el clustering las preguntas cb_P5 desde la 1 hasta la 13
dominios = [f'cb_P5_{i}' for i in range(1, 13)]

#se crea un nuevo dataframe con los datos de las columnas seleccionadas y se eliminan las filas con valores nulos
datos_km = df_biare[dominios].dropna()

#vemos el promedio y la desviación estándar de los datos para poder ir pensando en el modelo
print(datos_km.describe().loc[['mean', 'std']].round(2))

# %% [markdown]
# la media mas baja que es la de cb_P%_7 corresponde a la satisfacción con la seguridad ciudadana, la cual tambien es la de mayor dispersión.
# 
# También las preguntas referentes a la satisfaccion con el pais, la ciudad y el vecindario tienen desviacones muy altas.

# %% [markdown]
# ### Estandarizamos los datos

# %% [markdown]
# Estandarizamos los datos debido a la gran desviacion estandar que hay en las preguntas

# %%
#usamos standardscaler para escalar los datos
escalador_km = StandardScaler()

#se guarda la transformación de los datos en una nueva variable (media y desviacion estandar)
datos_esc = escalador_km.fit_transform(datos_km)

# %% [markdown]
# ### Metodo del codo y metodo de la silueta

# %%
# valores de k (grupos) que puede tomar nuestro k means
valores_k = range(2, 9)
#resultados del metodo del codo
inercias = []
#resultados del metodo de la silueta
siluetas = []

for k in valores_k:
    #creamos un modelo de kmeans con k clusters y 10 inicios aleatorios y el randomstate se asegura de siempre tener el mismo resultado
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    #se entrena el modelo con los datos
    etiquetas_km = km.fit_predict(datos_esc)
    #se mide que tan compactos son los grupos para cada k y se agrega a la lista
    inercias.append(km.inertia_)
    #se ve el promedio de que tan separados estan los grupos y se agrega a la lista
    siluetas.append(silhouette_score(datos_esc, etiquetas_km))

#se guardan los resultados
resultados_k = pd.DataFrame({
    'k': list(valores_k),
    'inercia': np.round(inercias),
    'silueta': np.round(siluetas, 3)
})
print(resultados_k)

# %%
plt.figure(figsize=(7, 4))
plt.plot(list(valores_k), inercias, marker='o')
plt.xlabel('k (número de grupos)')
plt.ylabel('Inercia')
plt.title('Método del codo')
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(7, 4))
plt.plot(list(valores_k), siluetas, marker='o', color='darkorange')
plt.xlabel('k (número de grupos)')
plt.ylabel('Coeficiente de silueta')
plt.title('Método de la silueta')
plt.tight_layout()
plt.show()

# %% [markdown]
# A pesar de que k = 2 seria la mejor inercia y el mejor coeficiente de silueta, tener solo dos grupos me daria solo dos descripciones como alta y baja, y como pudimos ver en las practicas anteriores  la gran mayoria de nuestros datos quedarian en alta, lo que sesgaria mucho nuestros resultados, por lo que entonces usaremo 3 y 4 grupos

# %% [markdown]
# ### Resultados de kmeans con k=3 y k=4

# %%
#usamos kmeans con 3 clusters, 10 inicios aleatorios y un random state para que siempre nos de el mismo resultado
km3 = KMeans(n_clusters=3, n_init=10, random_state=42)
#entrenamos el modelo con los datos
km3.fit(datos_esc)

#vemos cuantas personas hay en cada grupo
print(pd.Series(km3.labels_).value_counts().sort_index())
#vemos el promedio de cada variable para cada grupo
perfil_k3 = datos_km.groupby(km3.labels_).mean().round(2).T 
print(perfil_k3)

# %% [markdown]
# Al dividirlo en 3 grupos, vemos que principalmente, suponiendo que divide alto, medio y bajo, la gran mayoría quedan en alta y media, como habiamos visto antes, el promedio de la satisfaccion con la seguridad es la variable mas baja, en absolutamente todos los niveles algo que llama mucho la atencion es que en el bajo, practicamente todas las preguntas con excepcion de la de seguridad quedan cercanas a 6

# %% [markdown]
# Grupo 1: Alta satisfacción con la vida
# 
# Grupo 2: Media satisfacción con la vida
# 
# Grupo 3: Baja satisfacción con la vida

# %%
#usamos kmeans con 4 clusters, 10 inicios aleatorios y un random state para que siempre nos de el mismo resultado
km4 = KMeans(n_clusters=4, n_init=10, random_state=42)
#entrenamos el modelo con los datos
km4.fit(datos_esc)

#vemos cuantas personas hay en cada grupo
print(pd.Series(km4.labels_).value_counts().sort_index())
#vemos el promedio de cada variable para cada grupo
perfil_k4 = datos_km.groupby(km4.labels_).mean().round(2).T
print(perfil_k4)

# %% [markdown]
# Al dividirlo en 4 grupos, vemos que el grupo 1, 2 y 3 son las personas con satisfaccion alta, media y baja respectivamente, el que cambia realmente es el grupo 4, que como se puede ver el grupo 3 tiene 5.96 de satisfacción con la vida, pero el grupo 4 tiene 8.46 de satisfacción con la vida, pero podemos ver que su satisfaccion con la seguridad ciudadana es la mas baja de todos los grupos 2.60, tambien las preguntas que tan satisfecho esta con su ciudad y que tan satisfecho esta con su país, son las más bajas de todos los grupos, por lo que podemos decir que el grupo 4 son personas con alta satisfacción con la vida, pero muy insatisfechos con el lugar donde viven

# %% [markdown]
# Grupo 1: Alta satisfacción con la vida
# 
# Grupo 2: Media satisfacción con la vida
# 
# Grupo 3: Baja satisfacción con la vida
# 
# Grupo 4: Alta satisfacción con la vida, pero insatisfcción con el lugar en el que viven.


