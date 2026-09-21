# %% [markdown]
# # Pruebas Estadísticas

# %% [markdown]
# ### 1. Carga de librerias a utilizar

# %%
#!pip install pandas numpy scipy scikit-posthocs

# %% [markdown]
# ### 2. Cargamos las librerías a utilizar

# %%
import numpy as np
import pandas as pd
from scipy import stats

# %% [markdown]
# ### 3. Carga de datos

# %%
df_biare = pd.read_csv("../Limpieza de datos (Practica 1)/biare_limpio_2021_2024.csv")

df_biare['N_ENT'] = df_biare['N_ENT'].astype(str)
df_biare['N_REN'] = df_biare['N_REN'].astype(str)

# %% [markdown]
# ### 4. Supuesto de normalidad

# %% [markdown]
# Utilizamos el supuesto de normalidad para decidir si usamos ANOVA O Kruskal-wallis, usando la prueba de Shapiro-Wilk, sabemos si los resultados de la pregunta 1 de BIARE tiene una distribución normal.

# %% [markdown]
# Si el p-valor de la prueba es mayor a 0.05, entonces tiene una distribución normal, si es menor a 0.05 entonces no tiene una distribución normal.

# %% [markdown]
# El resultado estadístico entre más cercano este a 1, más normal es la distribución.

# %%
# Shapiro-Wilk sobre una muestra de 500 personas, de la pregunta cb_P1
muestra = df_biare['cb_P1'].dropna().sample(500, random_state=42) #quitamos los nulos si hay y tomamos una muestra de 500 personas permitiendo repeticion con random_state
stat, p_valor = stats.shapiro(muestra)

print(f"Estadístico: {stat:.4f}, p-valor: {p_valor:.10f}")

# %% [markdown]
# Como el p-valor es practicamente 0, eso quiere decir que p-valor < 0.05, por lo que no es una distribución normal, utilizaremos Kruskal-Wallis

# %% [markdown]
# ### 5. Realizamos la prueba de Kruskal-Wallis por año

# %% [markdown]
# Realizamos la prueba de Kruskal-Wallis, para ver si hay alguna diferencía entre las respuestas de la satisfacción con la vida entre los años

# %% [markdown]
# Si el p-valor es menor a 0.05 entonces sí hay una diferencia significativa entre al menos un año, si el p-valor es mayor a 0.05, entonces no hay una diferencia significativa entre los años.

# %%
#cargamos los anios de cuestionarios que aarecen en la muestra
a2021 = df_biare[df_biare['ANIO'] == 2021]['cb_P1'].dropna()
a2022 = df_biare[df_biare['ANIO'] == 2022]['cb_P1'].dropna()
a2023 = df_biare[df_biare['ANIO'] == 2023]['cb_P1'].dropna()
a2024 = df_biare[df_biare['ANIO'] == 2024]['cb_P1'].dropna()

#calculamos el estadistico y el p-valor de la prueba de Kruskal-Wallis
stat, p_valor = stats.kruskal(a2021, a2022, a2023, a2024)

#mostramos el estadistico y el p-valor de la prueba de Kruskal-Wallis
print(f"Estadístico H: {stat:.4f}")
print(f"p-valor: {p_valor:.10f}")

# %% [markdown]
# Como el p-valor es 0, entonces el p-valor < 0.05, por lo tanto si hay una diferencia entre al menos un año con respecto a la satisfacción con la vida

# %% [markdown]
# ##### Prueba de Dunn

# %% [markdown]
# Realizamos una prueba de Dunn, para saber entre cuales años hay exactamente una diferencia significante

# %%
#importamos la libreria scikit_posthocs para realizar la prueba post-hoc de Dunn
import scikit_posthocs as sp

#realizamos la prueba post-hoc de Dunn con ajuste de Bonferroni
resultado_dunn = sp.posthoc_dunn(df_biare, val_col='cb_P1', group_col='ANIO', p_adjust='bonferroni')
print(resultado_dunn)

# %% [markdown]
# Podemos observar que 2024 es el que difiere de los demas años, siendo el único que al compararlo con los demás años, da un resultado cercano a 0 en su p-valor, por lo que nuestro hallazgo es que 2024, difiere de los demás años.

# %% [markdown]
# ### 5. Realizamos la prueba de Kruskal-Wallis por género

# %% [markdown]
# Como lo hicimos anteriormente realizamos la prueba de Kruskal-Wallis, pero en esta ocasión la realizamos para comparar la satisfacción con la vida por género.

# %% [markdown]
# Si el p-valor de la prueba de Kruskal-Wallis es mayor a 0.05, entonces no hay diferencia entre hombres y mujeres, por el contrario, sí el p-valor es menor a 0.05, entonces si hay diferencia entre hombres y mujeres.

# %%
#dividimos los datos por hombres y mujeres
grupo_hombres = df_biare[df_biare['cs_SEX'] == 'Hombre']['cb_P1'].dropna()
grupo_mujeres = df_biare[df_biare['cs_SEX'] == 'Mujer']['cb_P1'].dropna()

#calculamos el estadistico y el p-valor de la prueba de Kruskal-Wallis para hombres y mujeres
stat, p_valor = stats.kruskal(grupo_hombres, grupo_mujeres)

print(f"Estadístico H: {stat:.4f}")
print(f"p-valor: {p_valor:.10f}")

# %% [markdown]
# Como el p-valor es menor a 0.05, entonces si hay una diferencia con la satisfacción con la vida entre hombres y mujeres

# %%
#mostramos el promedio que cada genero
print("Promedio de satisfacción por género:")
print(df_biare.groupby('cs_SEX')['cb_P1'].mean())

print("\nMediana de satisfacción por género:")
print(df_biare.groupby('cs_SEX')['cb_P1'].median())

# %% [markdown]
# Al ver el promedio y la mediana por género, vemos que aunque en el promedio hay diferencia, no es una diferencia tan significativa, por otro lado al verificar la mediana de cada género, vemos una diferencia de un número entre hombres y mujeres, siendo los hombres los de mayor satisfacción con la vida con una calificación de 9, mientras que las mujeres tienen una califiación de 8, aunque realmente no es una diferenciatan grande.

# %% [markdown]
# ### 6. Realizamos la prueba de Kruskal-Wallis por estado (seguridad ciudadana)

# %% [markdown]
# Como lo hicimos con las pruebas pasadas, realizamos la prueba de Kruskal-Wallis para ver si hay una diferencia entre la percepción de la seguridad ciudadana es diferente entre estados

# %% [markdown]
# Si el p-valor de la prueba de Kruskal-Wallis es mayor a 0.05, entonces no hay diferencia entre estados, por el contrario, sí el p-valor es menor a 0.05, entonces si hay diferencia entre estados.

# %%
#agrupamos por estado en cuanto a la percepcion de la seguridad ciudadana por estado
grupos_estado = [df_biare[df_biare['ENT'] == estado]['cb_P5_7'].dropna()
                  for estado in df_biare['ENT'].unique()]

#calculamos el estadistico y el p-valor de la prueba de Kruskal-Wallis para los estados
stat, p_valor = stats.kruskal(*grupos_estado)

print(f"Estadístico H: {stat:.4f}")
print(f"p-valor: {p_valor:.10f}")

# %% [markdown]
# Como el p-valor es menor a 0.05, entonces si hay una diferencia con la percepción de la seguridad ciudadana entre estados

# %% [markdown]
# ### Prueba de Dunn

# %% [markdown]
# Realizamos una prueba de Dunn, para saber entre cuales estados hay exactamente una diferencia significante

# %%
#realizamos la prueba post-hoc de Dunn con ajuste de Bonferroni
resultado_dunn_estado = sp.posthoc_dunn(df_biare, val_col='cb_P5_7', group_col='ENT', p_adjust='bonferroni')

# Convertimos la tabla a una fila por cada conexion de estados para poder filtrar
dunn_largo = resultado_dunn_estado.stack().reset_index()
dunn_largo.columns = ['Estado_A', 'Estado_B', 'p_valor']

# Quitamos las comparaciones de un estado consigo mismo, y los pares duplicados
dunn_largo = dunn_largo[dunn_largo['Estado_A'] < dunn_largo['Estado_B']]

# Filtramos solo los pares con un p-valor < 0.05
significativos = dunn_largo[dunn_largo['p_valor'] < 0.05].sort_values('p_valor')

print(significativos.head(20))

# %% [markdown]
# Los primeros 3 resultados casualmente son comparados con Veracruz, por lo que podemos decir que San Luis, Guerrero y Guanajuato pueden ser de los que peor seguridad ciudadana tienen en comparación a Veracruz que puede tener una de las seguridades ciudadanas mas altas.

# %% [markdown]
# Sacamos el promedio de la seguridad ciudadana de los estados para poder confirmar nuestro hallazgo

# %%
print(df_biare.groupby('ENT')['cb_P5_7'].mean().sort_values())

# %% [markdown]
# Zacatecas, Guerrero y San Luis Potosí reportan la seguridad ciudadana más baja, mientras que Yucatán, Veracruz y Coahuila reportan la seguridad ciudadana más alta, mostrando una diferencia de hasta poco más de 3 puntos.

# %% [markdown]
# ### 7. Realizamos la prueba de Kruskal-Wallis por nivel de educación

# %% [markdown]
# Como lo hicimos con las pruebas pasadas, realizamos la prueba de Kruskal-Wallis para ver si hay una diferencia entre la satisfacción con la vida por nivel de educación terminado

# %% [markdown]
# Si el p-valor de la prueba de Kruskal-Wallis es mayor a 0.05, entonces no hay diferencia entre hombres y mujeres, por el contrario, sí el p-valor es menor a 0.05, entonces si hay diferencia entre el nivel educativo alcanzado.

# %%
#agruapmos por nivel educativo y realizamos la prueba de Kruskal-Wallis
grupos_educacion = [df_biare[df_biare['cs_I_NIV'] == nivel]['cb_P1'].dropna()
                     for nivel in df_biare['cs_I_NIV'].unique()]

stat, p_valor = stats.kruskal(*grupos_educacion)

print(f"Estadístico H: {stat:.4f}")
print(f"p-valor: {p_valor:.10f}")

# %% [markdown]
# Como el p-valor es menor a 0.05, entonces confirmamos que sí hay diferencia entre la satisfacción con la vida, dependiendo del nivel educativo alcanzado.

# %% [markdown]
# ### Prueba de Dunn

# %% [markdown]
# Realizamos una prueba de Dunn, para saber entre cuales niveles educativos hay exactamente una diferencia significante

# %%
#realizamos la prueba post-hoc de Dunn con ajuste de Bonferroni
resultado_dunn_educ = sp.posthoc_dunn(df_biare, val_col='cb_P1', group_col='cs_I_NIV', p_adjust='bonferroni')

# Convertimos la tabla a una fila por cada conexion de niveles para poder filtrar
dunn_largo_educ = resultado_dunn_educ.stack().reset_index()
dunn_largo_educ.columns = ['Nivel_A', 'Nivel_B', 'p_valor']
# Quitamos las comparaciones de un nivel consigo mismo, y los pares duplicados
dunn_largo_educ = dunn_largo_educ[dunn_largo_educ['Nivel_A'] < dunn_largo_educ['Nivel_B']]

#filtramos por solo los valores significativos
significativos_educ = dunn_largo_educ[dunn_largo_educ['p_valor'] < 0.05].sort_values('p_valor')

print(significativos_educ)

# %% [markdown]
# La diferencia entre los niveles de educación alcanzados, se podría dividir en dos bloques, quienes tienen como máximo primaria y secundaria reportan significativamente menos satisfacción que quienes alcanzaron educación superior


