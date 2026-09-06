# %% [markdown]
# # Limpieza de datos

# %%
import pandas as pd
import numpy as np

# %% [markdown]
# ### Carga de datos

# %%
# datos de BIARE de 2021
df_cb_21 = pd.read_csv("../Dataset/biare_cb_1021.csv")
# datos de BIARE de 2022
df_cb_22 = pd.read_csv("../Dataset/biare_cb_1022.csv")
# datos de BIARE de 2023
df_cb_23 = pd.read_csv("../Dataset/biare_cb_1023.csv")
# datos de BIARE de 2024
df_cb_24 = pd.read_csv("../Dataset/biare_cb_1124.csv")

# %%
# datos sociodemograficos de 2021
df_cs_21 = pd.read_csv("../Dataset/biare_cs_1021.csv")
# datos sociodemograficos de 2022
df_cs_22 = pd.read_csv("../Dataset/biare_cs_1022.csv")
# datos sociodemograficos de 2023
df_cs_23 = pd.read_csv("../Dataset/biare_cs_1023.csv")
# datos sociodemograficos de 2024
df_cs_24 = pd.read_csv("../Dataset/biare_cs_1124.csv")

# %%
# datos de vivienda de 2021
df_viv_21 = pd.read_csv("../Dataset/biare_viv_1021.csv")
# datos de vivienda de 2022
df_viv_22 = pd.read_csv("../Dataset/biare_viv_1022.csv")
# datos de vivienda de 2023
df_viv_23 = pd.read_csv("../Dataset/biare_viv_1023.csv")
# datos de vivienda de 2024
df_viv_24 = pd.read_csv("../Dataset/biare_viv_1124.csv")

# %% [markdown]
# ### Analizamos el contenido de las tablas para encontrar anomalias

# %%
df_cb_21.info()

# %%
df_cb_22.info()

# %%
df_cb_23.info()

# %%
df_cb_24.info()

# %%
df_cs_21.info()

# %%
df_cs_22.info()

# %%
df_cs_23.info()

# %%
df_cs_24.info()

# %%
df_viv_21.info()

# %%
df_viv_22.info()

# %%
df_viv_23.info()

# %%
df_viv_24.info()

# %% [markdown]
# ##### Se ha encontrado una discrepancia entre la cantidad de columnas en la tabla de viviendas del 2024 en comparación con las demás, teniendo 2 columnas sobrantes, pero eso se solucionara al limpiar las tablas despues del merge

# %%
# al ver que varias columnas tienen el mismo nombre y algunas no son primary key
#entonces procedemos a renombrarlas para en el caso que estn en otra tabla, se puedan identificar
#identificamos las columnas que son primary key y no se deben renombrar
llaves = ['PER', 'N_ENT', 'FOL', 'ENT', 'CON', 'V_SEL', 'N_HOG', 'H_MUD', 'N_REN']

#renombramos las columnas del dataframe que no son primary key, agregando un identificador
def identificar_columnas(df, ident):
    nuevas_columnas = {
        col: f"{ident}_{col}" if col not in llaves else col
        for col in df.columns
    }
    return df.rename(columns=nuevas_columnas)

#lo agregamos a cada una de las tablas que creamos
df_cb_21 = identificar_columnas(df_cb_21, 'cb')
df_cs_21 = identificar_columnas(df_cs_21, 'cs')
df_viv_21 = identificar_columnas(df_viv_21, 'viv')

df_cb_22 = identificar_columnas(df_cb_22, 'cb')
df_cs_22 = identificar_columnas(df_cs_22, 'cs')
df_viv_22 = identificar_columnas(df_viv_22, 'viv')

df_cb_23 = identificar_columnas(df_cb_23, 'cb')
df_cs_23 = identificar_columnas(df_cs_23, 'cs')
df_viv_23 = identificar_columnas(df_viv_23, 'viv')

df_cb_24 = identificar_columnas(df_cb_24, 'cb')
df_cs_24 = identificar_columnas(df_cs_24, 'cs')
df_viv_24 = identificar_columnas(df_viv_24, 'viv')

# %% [markdown]
# ### Unión de las tablas por año

# %%
#declaramos las columnas que funcionaran como primary key para el merge
key_persona = ['N_ENT', 'FOL', 'ENT', 'CON', 'V_SEL', 'N_HOG', 'H_MUD', 'N_REN']
key_vivienda = ['N_ENT', 'FOL', 'ENT', 'CON', 'V_SEL', 'N_HOG', 'H_MUD']

#realizamos el merge entre la tabla de BIARE y la tabla de sociodemografico para el año 2021
df_2021 = df_cb_21.merge(df_cs_21, on=key_persona, how='left')

#a nuestro dataframe de 2021 le agregamos la tabla de vivienda
df_2021 = df_2021.merge(df_viv_21, on=key_vivienda, how='left')

#por ultimo agregamos una nueva columna que nos indique el año de la encuesta
df_2021['ANIO'] = 2021

# %%
#realizamos el merge entre la tabla de BIARE y la tabla de sociodemografico para el año 2022
df_2022 = df_cb_22.merge(df_cs_22, on=key_persona, how='left')

#a nuestro dataframe de 2022 le agregamos la tabla de vivienda
df_2022 = df_2022.merge(df_viv_22, on=key_vivienda, how='left')

#por ultimo agregamos una nueva columna que nos indique el año de la encuesta
df_2022['ANIO'] = 2022

# %%
#realizamos el merge entre la tabla de BIARE y la tabla de sociodemografico para el año 2023
df_2023 = df_cb_23.merge(df_cs_23, on=key_persona, how='left')

#a nuestro dataframe de 2023 le agregamos la tabla de vivienda
df_2023 = df_2023.merge(df_viv_23, on=key_vivienda, how='left')

#por ultimo agregamos una nueva columna que nos indique el año de la encuesta
df_2023['ANIO'] = 2023

# %%
#realizamos el merge entre la tabla de BIARE y la tabla de sociodemografico para el año 2024
df_2024 = df_cb_24.merge(df_cs_24, on=key_persona, how='left')

#a nuestro dataframe de 2024 le agregamos la tabla de vivienda
df_2024 = df_2024.merge(df_viv_24, on=key_vivienda, how='left')

#por ultimo agregamos una nueva columna que nos indique el año de la encuesta
df_2024['ANIO'] = 2024

# %%
# verificamos que se haya realizado correctamente el merge de las tablas
print(df_2021.shape)
print(df_2022.shape)
print(df_2023.shape)
print(df_2024.shape)

# %%
#juntamos todas las tablas en un solo dataframe
df_biare = pd.concat([df_2021, df_2022, df_2023, df_2024], ignore_index=True)

#verificamos que si se pasaron todas las columnas
df_biare.shape

# %% [markdown]
# ### Limpieza de los datos

# %%
#verificamos las columnas del nuevo dataset
nulos = df_biare.isna().sum().sort_values(ascending=False)
print(nulos[nulos > 0])

# %%
#eliminamos todas las columnas que no nos serviran para nuestro analisis
eliminar_columnas = [
    'cb_TIPO', 'cb_FAC_MOD', 'cs_PAR', 'cs_C_RES', 'cs_F_NAC', 'cs_A_ESC',
    'cs_I_PER', 'cs_I_ANIO', 'cs_TIPO', 'cs_ELEGIDO', 'cs_FACTOR', 'cs_FAC_MOD',
    'cs_R_DEF_MOD', 'cs_NOM', 'viv_FCH_DEF_MO', 'viv_FCH_PRE_MO', 
    'viv_CTA_SUP', 'viv_CTA_ENT', 'viv_RFC_ENT', 'viv_RFC_SUP', 'viv_TIPO', 
    'viv_FACTOR', 'viv_HORA', 'viv_AGEB', 'viv_UPM', 'viv_AREA',
    'FOL', 'CON', 'viv_N_PRO_VIV', 'V_SEL', 'N_HOG', 'H_MUD',
    'viv_R_PRE', 'viv_R_DEF', 'viv_E_OBS', 'viv_R_INF', 'viv_R_ELE', 'viv_TEL',
    'viv_R_HAB', 'viv_R_NUE', 'viv_R_AUS', 'viv_DIASEM', 'viv_R_PRE_MOD',
    'viv_R_DEF_MOD', 'viv_CD', 'viv_MPIO', 'PER_x', 'PER_y', 'PER'
]


# %%
df_biare = df_biare.drop(columns=eliminar_columnas)
print(df_biare.shape)
print(df_biare.columns.tolist())

# %%
#volvemos a verificar los nulos
nulos = df_biare.isna().sum().sort_values(ascending=False)
print(nulos[nulos > 0])

# %%
#como los datos númericos no podemos alterarlos, ya que para realizar el analisis estadistico
#si lo llenamos con 0 influira y si le ponemos la leyenda no aplica, no se podran realizar los analisis

#verificamos los tipos de datos de las columnas
df_biare.info()

# %%
#verificamos los datos de la columna cs_SEX, ya que aparece como object
df_biare['cs_SEX'].unique()

#como el sexo aparece como 1 y 2, procedemos a cambiarlo por Hombre y Mujer, respectivamente
df_biare['cs_SEX'] = pd.to_numeric(df_biare['cs_SEX'], errors='coerce')
df_biare['cs_SEX'] = df_biare['cs_SEX'].map({1: 'Hombre', 2: 'Mujer'})

#verificamos que se hagan los cambios correctamente
df_biare['cs_SEX'].unique()

# %%
#verificamos los datos de la columna cs_EDA, ya que aparece como object
df_biare['cs_EDA'].unique()

#vemos que algunas edades aparecen como str y otras como int, por lo que procedemos a convertirlas primero a numero y luego a int
df_biare['cs_EDA'] = pd.to_numeric(df_biare['cs_EDA'], errors='coerce')
df_biare['cs_EDA'] = df_biare['cs_EDA'].astype('Int64')

#verificamos que ningun valor sea menor a 15 ni tampoco mayor a 120
df_biare['cs_EDA'].min()
df_biare['cs_EDA'].max()

#verificamos que se hagan los cambios correctamente
df_biare['cs_EDA'].unique()

# %%
#verificamos los datos de la columna cs_ALF, ya que aparece como object
df_biare['cs_ALF'].unique()

#como vemos que algunos valores aparecen como str y otros como int, los corvertimos a numerico
df_biare['cs_ALF'] = pd.to_numeric(df_biare['cs_ALF'], errors='coerce')

#especificamos el significado de cada uno de los numeros
df_biare['cs_ALF'] = df_biare['cs_ALF'].map({1: 'Sabe leer y escribir', 2: 'No sabe leer y escribir', 9: 'No especificado'})

#verificamos que se hayan realizado los cambios
df_biare['cs_ALF'].unique()

# %%
#verificamos los datos de la columna cs_I_NIV, ya que aparece como object
df_biare['cs_I_NIV'].unique()

#como vemos que algunos valores aparecen como str y otros como int, los corvertimos a numerico
df_biare['cs_I_NIV'] = pd.to_numeric(df_biare['cs_I_NIV'], errors='coerce')

#especificamos el significado de cada uno de los numeros
df_biare['cs_I_NIV'] = df_biare['cs_I_NIV'].map({0: 'Ninguno', 1: 'Preescolar', 2: 'Primaria',
    3: 'Secundaria', 4: 'Preparatoria o bachillerato', 5: 'Normal', 6: 'Carrera técnica',
    7: 'Profesional', 8: 'Maestría', 9: 'Doctorado', 99: 'No sabe'
})

#verificamos que se hayan realizado los cambios
df_biare['cs_I_NIV'].unique()

# %%
#verificamos los datos de la columna cs_E_CON, ya que aparece como object
df_biare['cs_E_CON'].unique()

#como vemos que algunos valores aparecen como str y otros como int, los corvertimos a numerico
df_biare['cs_E_CON'] = pd.to_numeric(df_biare['cs_E_CON'], errors='coerce')

#especificamos el significado de cada uno de los numeros
df_biare['cs_E_CON'] = df_biare['cs_E_CON'].map({1: 'Unión libre', 2: 'Separado(a)',
    3: 'Divorciado(a)', 4: 'Viudo(a)', 5: 'Casado(a)', 6: 'Soltero(a)',
    9: 'No especificado'
})

#verificamos que se hayan realizado los cambios
df_biare['cs_E_CON'].unique()

# %%
#verificamos los datos de la columna cs_C_ACT, ya que aparece como object
df_biare['cs_C_ACT'].unique()

#como vemos que algunos valores aparecen como str y otros como int, los corvertimos a numerico
df_biare['cs_C_ACT'] = pd.to_numeric(df_biare['cs_C_ACT'], errors='coerce')

#especificamos el significado de cada uno de los numeros
df_biare['cs_C_ACT'] = df_biare['cs_C_ACT'].map({
    1: 'Trabajó por lo menos una hora para obtener ingresos',
    2: 'Trabajó sin pago ayudando a un familiar',
    3: 'No trabajó, pero sí tiene trabajo',
    4: 'Buscó trabajo',
    5: 'Espera respuesta a una solicitud de trabajo',
    6: 'Estudiante',
    7: 'Se dedica a los quehaceres del hogar',
    8: 'Jubilado(a) o pensionado(a)',
    9: 'Incapacitado(a) permanentemente para trabajar',
    10: 'Otra situación',
    99: 'No especificado'
})

#verificamos que se hayan realizado los cambios
df_biare['cs_C_ACT'].unique()

# %%
#verificamos los datos de la columna cs_OCU, ya que aparece como object
df_biare['cs_OCU'].unique()

#como vemos que algunos valores aparecen como str y otros como int, los corvertimos a numerico
df_biare['cs_OCU'] = pd.to_numeric(df_biare['cs_OCU'], errors='coerce')

#especificamos el significado de cada uno de los numeros
df_biare['cs_OCU'] = df_biare['cs_OCU'].map({
    1: 'Profesionistas y técnicos',
    2: 'Funcionarios de los sectores público y privado',
    3: 'Personal administrativo',
    4: 'Comerciantes, vendedores y similares',
    5: 'Trabajadores en servicios personales y conductores de vehículos',
    6: 'Trabajadores en labores agropecuarias',
    7: 'Trabajadores industriales',
    9: 'No especificado'
})

#llenamos los nulos como 'no trabaja'
df_biare['cs_OCU'] = df_biare['cs_OCU'].fillna('No trabaja')

#verificamos que se hayan realizado los cambios
df_biare['cs_OCU'].unique()

# %%
#verificamos los datos de la columna cs_A_ECO, ya que aparece como object
df_biare['cs_A_ECO'].unique()

#como vemos que algunos valores aparecen como str y otros como int, los corvertimos a numerico
df_biare['cs_A_ECO'] = pd.to_numeric(df_biare['cs_A_ECO'], errors='coerce')

#especificamos el significado de cada uno de los numeros
df_biare['cs_A_ECO'] = df_biare['cs_A_ECO'].map({1: 'Agropecuaria', 2: 'Industria',
    3: 'Construcción', 4: 'Comercio', 5: 'Servicios', 6: 'Comunicaciones y transportes',
    7: 'Administración pública y defensa', 9: 'No especificado'
})

#llenamos los nulos como 'no trabaja'
df_biare['cs_A_ECO'] = df_biare['cs_A_ECO'].fillna('No trabaja')

#verificamos que se hayan realizado los cambios
df_biare['cs_A_ECO'].unique()

# %%
#verificamos los datos de la columna cs_POS, ya que aparece como object
df_biare['cs_POS'].unique()

#como vemos que algunos valores aparecen como str y otros como int, los corvertimos a numerico
df_biare['cs_POS'] = pd.to_numeric(df_biare['cs_POS'], errors='coerce')

#especificamos el significado de cada uno de los numeros
df_biare['cs_POS'] = df_biare['cs_POS'].map({
    1: 'Patrón(a)',
    2: 'Trabajador(a) por su cuenta',
    3: 'Trabajador(a) a sueldo fijo, salario o jornal',
    4: 'Trabajador(a) a destajo',
    5: 'Trabajador(a) a comisión o porcentaje',
    6: 'Trabajador(a) sin pago',
    9: 'No sabe'
})

#llenamos los nulos como 'no trabaja'
df_biare['cs_POS'] = df_biare['cs_POS'].fillna('No trabaja')

#verificamos que se hayan realizado los cambios
df_biare['cs_POS'].unique()

# %%
#verificamos los datos de la columna cs_ING, ya que aparece como object
df_biare['cs_ING'].unique()

#PRIMERO convertimos a numérico (esto ya arregla los '006000' y el ' ' -> NaN)
df_biare['cs_ING'] = pd.to_numeric(df_biare['cs_ING'], errors='coerce')

#YA CONVERTIDO a número, ahora sí el 999999 (no sabe/no responde) se reemplaza correctamente
df_biare['cs_ING'] = df_biare['cs_ING'].replace(999999, np.nan)

#confirmamos que ya no quede ningún 999999
print('Casos con 999999 restantes:', (df_biare['cs_ING'] == 999999).sum())

#verificamos que se hayan realizado los cambios
df_biare['cs_ING'].unique()

# %%
#verificamos los datos de la columna viv_P1, ya que aparece como object
df_biare['viv_P1'].unique()

#como vemos que algunos valores aparecen como str y otros como int, los corvertimos a numerico y luego a int
df_biare['viv_P1'] = pd.to_numeric(df_biare['viv_P1'], errors='coerce')
df_biare['viv_P1'] = df_biare['viv_P1'].astype('Int64')

#verificamos que se hayan realizado los cambios
df_biare['viv_P1'].unique()

# %%
#verificamos los datos de la columna viv_P2, ya que aparece como object
df_biare['viv_P2'].unique()

#como vemos que algunos valores aparecen como str y otros como int, los corvertimos a numerico
df_biare['viv_P2'] = pd.to_numeric(df_biare['viv_P2'], errors='coerce')

#especificamos el significado de cada uno de los numeros
df_biare['viv_P2'] = df_biare['viv_P2'].map({1: 'Sí', 2: 'No'})

#llenamos los nulos como 'No aplica'
df_biare['viv_P2'] = df_biare['viv_P2'].fillna('No aplica')

#verificamos que se hayan realizado los cambios
df_biare['viv_P2'].unique()

# %%
#verificamos los datos de la columna viv_P3, ya que aparece como object
df_biare['viv_P3'].unique()

#como vemos que algunos valores aparecen como str y otros como int, los corvertimos a numerico
df_biare['viv_P3'] = pd.to_numeric(df_biare['viv_P3'], errors='coerce')

#llenamos los nulos como 1
df_biare['viv_P3'] = df_biare['viv_P3'].fillna(1)

#cambiamos el tipo de dato a int
df_biare['viv_P3'] = df_biare['viv_P3'].astype('Int64')

#verificamos que se hayan realizado los cambios
df_biare['viv_P3'].unique()

# %%
#verificamos los datos de la columna viv_P4, ya que aparece como object
df_biare['viv_P4'].unique()

#como vemos que algunos valores aparecen como str y otros como int, los corvertimos a numerico
df_biare['viv_P4'] = pd.to_numeric(df_biare['viv_P4'], errors='coerce')

#llenamos los nulos de p4 con el valor de su p1
df_biare['viv_P4'] = df_biare['viv_P4'].fillna(df_biare['viv_P1'])

#cambiamos el tipo de dato a int
df_biare['viv_P4'] = df_biare['viv_P4'].astype('Int64')

#verificamos que se hayan realizado los cambios
df_biare['viv_P4'].unique()

# %%
df_biare.info()

# %%
#convertimos N_ENT y N_REN a str ya que son identificadores
df_biare['N_ENT'] = df_biare['N_ENT'].astype(str)
df_biare['N_REN'] = df_biare['N_REN'].astype(str)

#como vemos que algunos valores de ENT aparecen como str y otros como int, los corvertimos a numerico
df_biare['ENT'] = pd.to_numeric(df_biare['ENT'], errors='coerce')

#definimos los valores de cada numero
ent_dict = {
    1: 'Aguascalientes', 2: 'Baja California', 3: 'Baja California Sur',
    4: 'Campeche', 5: 'Coahuila', 6: 'Colima', 7: 'Chiapas', 8: 'Chihuahua',
    9: 'Ciudad de México', 10: 'Durango', 11: 'Guanajuato', 12: 'Guerrero',
    13: 'Hidalgo', 14: 'Jalisco', 15: 'México', 16: 'Michoacán', 17: 'Morelos',
    18: 'Nayarit', 19: 'Nuevo León', 20: 'Oaxaca', 21: 'Puebla', 22: 'Querétaro',
    23: 'Quintana Roo', 24: 'San Luis Potosí', 25: 'Sinaloa', 26: 'Sonora',
    27: 'Tabasco', 28: 'Tamaulipas', 29: 'Tlaxcala', 30: 'Veracruz',
    31: 'Yucatán',32: 'Zacatecas'
}

#convertimos los numeros en estados
df_biare['ENT'] = df_biare['ENT'].map(ent_dict)

df_biare['ENT'].unique()

# %%
#convertimos las columnas de fechas en tipo de dato datetime
df_biare['viv_FCH_PRE'] = pd.to_datetime(df_biare['viv_FCH_PRE'], format='%d/%m/%Y', errors='coerce')
df_biare['viv_FCH_DEF'] = pd.to_datetime(df_biare['viv_FCH_DEF'], format='%d/%m/%Y', errors='coerce')

df_biare[['viv_FCH_PRE', 'viv_FCH_DEF']].info()


# %%
#guardamos en csv nuestro dataset limpio
df_biare.to_csv('../Limpieza de datos (Practica 1)/biare_limpio_2021_2024.csv', index=False, encoding='utf-8-sig')


