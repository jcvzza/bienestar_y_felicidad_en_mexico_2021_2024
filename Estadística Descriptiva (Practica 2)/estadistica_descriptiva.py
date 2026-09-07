# %% [markdown]
# # Estadística descriptiva

# %% [markdown]
# ### Importar librerias

# %%
import pandas as pd

# %% [markdown]
# ### Carga de datos

# %%
#cargamos el conjunto de datos ya limpio
df_biare = pd.read_csv("../Limpieza de datos (Practica 1)/biare_limpio_2021_2024.csv")

# %% [markdown]
# ### Aplicamos estadística descriptiva

# %% [markdown]
# #### Preguntras sobre bienestar autorreportado: Satisfacción con la vida en general

# %%
# mostramos las estadísticas descriptivas de las preguntas sobre satisfacción con la vida en general
df_biare[["cb_P1", "cb_P2"]].describe()

# %% [markdown]
# #### Preguntras sobre bienestar autorreportado: Eudemonia

# %%
# mostramos las estadísticas descriptivas de las preguntas sobre eudemonia
df_biare[["cb_P3_1", "cb_P3_2", "cb_P3_3", "cb_P3_4", "cb_P3_5", "cb_P3_6", "cb_P3_7", "cb_P3_8", "cb_P3_9", "cb_P3_10", "cb_P3_11"]].describe()

# %% [markdown]
# #### Preguntras sobre bienestar autorreportado: Balance animico

# %%
# mostramos las estadísticas descriptivas de las preguntas sobre balance animico
df_biare[["cb_P4_1", "cb_P4_2", "cb_P4_3", "cb_P4_4", "cb_P4_5", "cb_P4_6", "cb_P4_7", "cb_P4_8", "cb_P4_9", "cb_P4_10"]].describe()

# %% [markdown]
# #### Preguntras sobre bienestar autorreportado: Satisfacción con dominios especificos

# %%
# mostramos las estadísticas descriptivas de las preguntas sobre satisfaccion con dominios especificos
df_biare[["cb_P5_1", "cb_P5_2", "cb_P5_3", "cb_P5_4", "cb_P5_5", "cb_P5_6", "cb_P5_7", "cb_P5_8", "cb_P5_9", "cb_P5_10", "cb_P5_11", "cb_P5_12"]].describe()

# %% [markdown]
# #### Preguntras sociodemograficas

# %%
# mostramos las estadísticas descriptivas de las preguntas sociodemograficas
df_biare[["cs_EDA", "cs_ING"]].describe()

# %% [markdown]
# #### Preguntas sobre vivienda

# %%
# mostramos las estadísticas descriptivas de las preguntas sobre vivienda
df_biare[["viv_P1", "viv_P3","viv_P4"]].describe()

# %% [markdown]
# ### Estadistica descriptiva de las variables categoricas

# %% [markdown]
# #### Género

# %%
# mostramos las estadísticas descriptivas del genero
print(df_biare['cs_SEX'].value_counts())

# %% [markdown]
# #### Nivel de instrucción

# %%
#mostramos las estadísticas descriptivas del nivel educativo
print(df_biare['cs_I_NIV'].value_counts())

# %% [markdown]
# #### Estado conyugal

# %%
#mostramos las estadísticas descriptivas del estado conyugal
print(df_biare['cs_E_CON'].value_counts())

# %% [markdown]
# #### Condición de actividad

# %%
#mostramos las estadísticas descriptivas de la condición de actividad
print(df_biare['cs_C_ACT'].value_counts())

# %% [markdown]
# #### Estados

# %%
#mostramos las estadísticas descriptivas de los estados de méxico
print(df_biare['ENT'].value_counts())

# %% [markdown]
# ### Agrupación de respuestas

# %% [markdown]
# #### Satisfacción de vida por año

# %%
#mostramos el promedio de satisfacción con la vida por año
print("Satisfacción con la vida por año")
print(df_biare.groupby('ANIO')['cb_P1'].mean())

# %% [markdown]
# #### Satisfacción con la vida por género

# %%
#mostramos el promedio de satisfacción con la vida por género
print("\nSatisfacción con la vida por género")
print(df_biare.groupby('cs_SEX')['cb_P1'].mean())

# %% [markdown]
# #### Satiasfacción con la vida por estado

# %%
#mostramos el promedio de satisfacción con la vida por estado en México
print("\nSatisfacción con la vida por estado")
print(df_biare.groupby('ENT')['cb_P1'].mean().sort_values(ascending=False))

# %% [markdown]
# #### Satisfacción con la vida por nivel educativo

# %%
#mostramos el promedio de satisfacción con la vida por nivel educativo
print("\nSatisfacción con la vida por nivel educativo")
print(df_biare.groupby('cs_I_NIV')['cb_P1'].mean().sort_values(ascending=False))

# %% [markdown]
# #### Satisfacción con la vida por nivel de actividad

# %%
#mostramos el promedio de satisfacción con la vida por nivel de actividad
print("\nSatisfacción con la vida por condición de actividad")
print(df_biare.groupby('cs_C_ACT')['cb_P1'].mean().sort_values(ascending=False))

# %% [markdown]
# #### Ingreso promedio por nivel educativo

# %%
#mostramos el promedio de ingreso por nivel educativo
print("\nIngreso promedio por nivel educativo")
print(df_biare.groupby('cs_I_NIV')['cs_ING'].mean().sort_values(ascending=False))

# %% [markdown]
# #### Satisfacción con la seguridad del estado

# %%
#mostramos el promedio de satisfacción con la seguridad por estado
print("\nSatisfacción con la seguridad por estado")
print(df_biare.groupby('ENT')['cb_P5_7'].mean().sort_values())

# %% [markdown]
# #### Autoestima general por género

# %%
#mostramos el promedio de la autoestima general por género
print("\nAutoestima general por género")
print(df_biare.groupby('cs_SEX')['cb_P3_1'].mean())


