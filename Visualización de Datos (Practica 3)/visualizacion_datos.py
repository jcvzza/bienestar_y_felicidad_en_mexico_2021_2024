# %% [markdown]
# # Visualización de datos

# %% [markdown]
# ### Importación de librerías

# %%
#importamos las librerías necesarias para hacer nuestras graficas
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %% [markdown]
# ### Carga de datos

# %%
#cargamos los datos y los hacemos un dataframe
df_biare = pd.read_csv("../Limpieza de datos (Practica 1)/biare_limpio_2021_2024.csv")

#debido a que el csv no guarda e tipo de dato, nosotros se lo damos
df_biare['N_ENT'] = df_biare['N_ENT'].astype(str)
df_biare['N_REN'] = df_biare['N_REN'].astype(str)

# %% [markdown]
# # Gráficas descriptivas

# %% [markdown]
# #### Histogramas sobre las preguntas de satisfacción con la vida en general

# %%
preguntas = ['cb_P1', 'cb_P2', 'cb_P3_1', 'cb_P4_1', 'cb_P5_1']
titulos = ['Satisfacción actual', 'Satisfacción hace un año',
           'Me siento bien conmigo mismo', 'Buen humor (ayer)', 'Satisfacción con nivel de vida']

fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for ax, pregunta, titulo in zip(axes, preguntas, titulos):
    ax.hist(df_biare[pregunta].dropna(), bins=11, color='steelblue', edgecolor='white')
    ax.set_title(titulo)
    ax.set_xlabel('Escala 0-10')

plt.tight_layout()
plt.show()

# %% [markdown]
# Como se puede observar la gran mayoría de las preguntas sigue la misma tendencia, en la que la mayor parte de las encuestas se situa en la escala entre el 8 y el 10.

# %% [markdown]
# #### box plot de satisfacción con la vida por año

# %%
plt.figure(figsize=(8, 5))
sns.boxplot(data=df_biare, x='ANIO', y='cb_P1', hue='ANIO', palette='Blues', legend=False,
            medianprops={'color': 'red', 'linewidth': 2})
plt.title('Satisfacción con la vida por año')
plt.xlabel('Año')
plt.ylabel('Satisfacción (0-10)')
plt.show()

# %% [markdown]
# Lo único que destaca entre años, es que en 2022, el 50% de los datos se encuentran entre el 8 y el 9, y no entre el 10 y el 8 como comunmente sucede.

# %% [markdown]
# #### Heatmap en comparación de la satisfaccióm con la vida y la edad

# %%
df_biare['grupo_edad'] = pd.cut(df_biare['cs_EDA'],
                                  bins=[0, 25, 40, 60, 100],
                                  labels=['18-25', '26-40', '41-60', '61+'])

plt.figure(figsize=(10, 6))
tabla_cruzada_edad = pd.crosstab(df_biare['grupo_edad'], df_biare['cb_P1'])
sns.heatmap(tabla_cruzada_edad, cmap='Oranges', annot=False)
plt.title('Grupo de edad vs Satisfacción con la vida (mapa de calor)')
plt.xlabel('Satisfacción con la vida (0-10)')
plt.ylabel('Grupo de edad')
plt.show()

# %% [markdown]
# Podemos ver que la distribución de satisfacción con la vida, suele ser la misma entre las edades, siendo los de 26 a 40 y 41 a 60 los que más datos tienen.

# %% [markdown]
# #### Pie chart sobre la distribución de edades en la muestra

# %%
conteo_edad = df_biare['grupo_edad'].value_counts().sort_index()
total = conteo_edad.sum()

def mostrar_pct_y_cantidad(pct):
    cantidad = int(round(pct / 100 * total))
    return f'{pct:.1f}%\n({cantidad})'

plt.figure(figsize=(7, 7))
plt.pie(conteo_edad, labels=conteo_edad.index, autopct=mostrar_pct_y_cantidad,
        colors=['#FFD59E', '#FFA45C', '#FF7F41', '#D9534F'])
plt.title('Composición de la muestra por grupo de edad')
plt.show()

# %% [markdown]
# Como vimos anteriormente, los dos rangos de edad con más participación en la muestra son de 41-60 con el 36% y de 26-40 con el 29.1%

# %% [markdown]
# #### Pie chart sobre la distribución de géneros en la muestra

# %%
conteo_sexo = df_biare['cs_SEX'].value_counts()

plt.figure(figsize=(6, 6))
plt.pie(conteo_sexo, labels=conteo_sexo.index, autopct='%1.1f%%', colors=['lightcoral', 'lightskyblue'])
plt.title('Composición de la muestra por sexo')
plt.show()

# %% [markdown]
# Podemos ver que a pesar de que hay más mujeres que hombres, estamos muy cerca de una equidad 50 y 50

# %% [markdown]
# #### Gráfico de barras con el promedio de satisfacción con la vida por estado

# %%
promedio_estado = df_biare.groupby('ENT')['cb_P1'].mean().sort_values()

plt.figure(figsize=(8, 10))
plt.barh(promedio_estado.index, promedio_estado.values, color='mediumseagreen')
plt.title('Satisfacción promedio con la vida por estado')
plt.xlabel('Satisfacción promedio (0-10)')
plt.show()

# %% [markdown]
# Aguascalientes y Durango cuentan con el mayor promedio de satisfacción de vida, mientras que Oaxaca y Tlaxcala cuentan con el peor promedio de satisfacción de vida.

# %% [markdown]
# #### Boxplot de satisfacción con la vida por nivel económico

# %%
# Creamos rangos de ingreso usando cuartiles (4 grupos con cantidad similar de personas en cada uno)
df_biare['nivel_economico'] = pd.qcut(df_biare['cs_ING'], q=4,
                                        labels=['Bajo', 'Medio-bajo', 'Medio-alto', 'Alto'])

plt.figure(figsize=(8, 5))
sns.boxplot(data=df_biare, x='nivel_economico', y='cb_P1',
            order=['Bajo', 'Medio-bajo', 'Medio-alto', 'Alto'], palette='Greens')
plt.title('Satisfacción con la vida por nivel económico')
plt.xlabel('Nivel económico (por ingreso)')
plt.ylabel('Satisfacción (0-10)')
plt.show()

# %% [markdown]
# Mientras que del nivel medio bajo al alto, la distribución de la satisfacción con la vida es prácticamente la misma, entre 8 y 10, donde el 50 se encuentra por debajo de 9 y el otro 50 por encima, en el caso del nivel economico bajo, la distribución es mas amplia, y el 50% de la muestra se encuentra entre el 7 y el 10, donde la mitad esta por debajo del 8 y la otra mitad por encima de 8.

# %% [markdown]
# ##### Histogramas sobre las preguntas de la satisfacción con dominios especificos 

# %%
dominios = ['cb_P5_1', 'cb_P5_2', 'cb_P5_4', 'cb_P5_8', 'cb_P5_7']
titulos = ['Nivel de vida', 'Salud', 'Relaciones personales',
           'Actividad principal', 'Seguridad ciudadana']

fig, axes = plt.subplots(1, 5, figsize=(22, 4))

for ax, dominio, titulo in zip(axes, dominios, titulos):
    ax.hist(df_biare[dominio].dropna(), bins=11, color='mediumpurple', edgecolor='white')
    ax.set_title(titulo)
    ax.set_xlabel('Escala 0-10')
    ax.set_xlim(0, 10)

plt.tight_layout()
plt.show()

# %% [markdown]
# Aunque la gran mayoría de la satisfacción con los dominios especificos sigue la misma distribución, la seguridad ciudadana es la que más distribución tiene, por lo que es una metrica que debera evaluarse

# %% [markdown]
# ##### Heatmap con la comparación entre la seguridad ciudadana y la satisfacción con la vida

# %%
import numpy as np

tabla_cruzada = pd.crosstab(df_biare['cb_P5_7'], df_biare['cb_P1'])

plt.figure(figsize=(9, 7))
sns.heatmap(tabla_cruzada, cmap='Purples', annot=False)
plt.title('Seguridad ciudadana vs Satisfacción con la vida (mapa de calor)')
plt.xlabel('Satisfacción con la vida (0-10)')
plt.ylabel('Satisfacción con seguridad ciudadana (0-10)')
plt.gca().invert_yaxis()
plt.show()

# %% [markdown]
# Como podemos ver, la satisfacción con la seguridad ciudadana suele ser baja, mostrando una gran concentración de datos en el número 5 con la satisfacción de la seguridad, que curiosamente se encuentra con un 8 y 10 de satisfacción con la vida, por lo que tomando como 5 un indiferente, a pesar de que la satisfacción con la seguridad no es la mejor, la satisfacción con la vida sigue siendo alta.


