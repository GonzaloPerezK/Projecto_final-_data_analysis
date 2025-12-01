# TODO: Aquí debes escribir tu código

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# ----------------------------------------------------
# Configuración básica de la página
# ----------------------------------------------------
st.set_page_config(
    page_title="California Housing - Análisis Interactivo",
    page_icon="🏠",
    layout="wide",
)

st.title("🏠 Análisis Interactivo de Precios de Vivienda en California")
st.write(
    """
Esta aplicación utiliza el *dataset* **California Housing** de `scikit-learn` para 
explorar la relación entre características sociodemográficas y el valor mediano 
de las viviendas (**MedHouseVal**).

En la barra lateral podés ajustar filtros y observar cómo cambian los datos,
las estadísticas descriptivas y los gráficos.
"""
)

# ----------------------------------------------------
# Fase 1: Carga y vista general de los datos
# ----------------------------------------------------


@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Carga el dataset de California desde scikit-learn y lo devuelve como un DataFrame.
    """
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()
    # El DataFrame ya incluye las columnas de características y la columna objetivo `MedHouseVal`
    return df


df_california = load_data()

st.subheader("Vista general del dataset")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**Primeras 5 filas del dataset:**")
    st.dataframe(df_california.head())

with col_b:
    st.markdown("**Información general y tipos de datos:**")
    st.write(df_california.dtypes)

st.markdown("**Valores faltantes por columna:**")
st.write(df_california.isna().sum())

st.markdown("---")

# ----------------------------------------------------
# Fase 2: Análisis descriptivo interactivo (widgets)
# ----------------------------------------------------

st.sidebar.header("Filtros de exploración")

st.sidebar.write(
    """
Ajustá los filtros para explorar cómo cambian los datos 
y las métricas descriptivas del valor de la vivienda.
"""
)

# Slider para filtrar por HouseAge
age_min = int(df_california["HouseAge"].min())
age_max = int(df_california["HouseAge"].max())

age_range = st.sidebar.slider(
    "Rango de Edad Mediana de la Casa (HouseAge)",
    min_value=age_min,
    max_value=age_max,
    value=(age_min, age_max),
    step=1,
)

# Filtro por latitud mínima (vecindario aproximado)
lat_min_global = float(df_california["Latitude"].min())
lat_max_global = float(df_california["Latitude"].max())

st.sidebar.markdown("### Filtro por vecindario (Latitud mínima)")
min_latitude = st.sidebar.number_input(
    "Latitud mínima",
    min_value=lat_min_global,
    max_value=lat_max_global,
    value=lat_min_global,
    step=0.5,
    help="Se filtrarán las viviendas cuya latitud sea mayor o igual a este valor.",
)

# Aplicar filtros
filtered_df = df_california[
    (df_california["HouseAge"] >= age_range[0])
    & (df_california["HouseAge"] <= age_range[1])
    & (df_california["Latitude"] >= min_latitude)
]

st.subheader("Datos filtrados")

st.write(
    f"Filas resultantes después de aplicar los filtros: **{filtered_df.shape[0]}**"
)
st.dataframe(filtered_df.head())

# Resumen descriptivo de MedHouseVal
st.subheader("Resumen descriptivo de MedHouseVal (datos filtrados)")

if not filtered_df.empty:
    mediana_valor = filtered_df["MedHouseVal"].median()
    rango_valor = filtered_df["MedHouseVal"].max() - filtered_df["MedHouseVal"].min()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mediana de MedHouseVal", f"{mediana_valor:.3f}")
    with col2:
        st.metric("Rango (máx - mín) de MedHouseVal", f"{rango_valor:.3f}")
else:
    st.warning(
        "No hay datos con los filtros seleccionados. Ajustá los filtros en la barra lateral."
    )

st.markdown("---")

# ----------------------------------------------------
# Fase 3: Visualización dinámica
# ----------------------------------------------------

# 3.1 Histograma de MedHouseVal
st.subheader("Distribución del valor mediano de la vivienda (MedHouseVal)")

if not filtered_df.empty:
    fig_hist, ax_hist = plt.subplots()
    ax_hist.hist(filtered_df["MedHouseVal"], bins=30)
    ax_hist.set_xlabel("MedHouseVal (cientos de miles de USD)")
    ax_hist.set_ylabel("Frecuencia")
    ax_hist.set_title("Histograma de MedHouseVal (datos filtrados)")
    st.pyplot(fig_hist)
else:
    st.info(
        "No se puede mostrar el histograma porque no hay datos con los filtros actuales."
    )

# 3.2 Scatter plot MedInc vs MedHouseVal
st.subheader("Relación entre ingresos y valor de la vivienda")

if not filtered_df.empty:
    fig_scatter, ax_scatter = plt.subplots()
    ax_scatter.scatter(
        filtered_df["MedInc"],
        filtered_df["MedHouseVal"],
        alpha=0.3,
    )
    ax_scatter.set_xlabel("MedInc (decenas de miles de USD)")
    ax_scatter.set_ylabel("MedHouseVal (cientos de miles de USD)")
    ax_scatter.set_title("MedInc vs MedHouseVal (datos filtrados)")
    st.pyplot(fig_scatter)
else:
    st.info(
        "No se puede mostrar el gráfico de dispersión porque no hay datos con los filtros actuales."
    )

st.markdown("---")

# 3.3 Mapa geográfico (extra)
st.subheader("Mapa geográfico de las viviendas filtradas (opcional)")

if not filtered_df.empty:
    st.caption("Cada punto representa un grupo de bloques según Latitude y Longitude.")
    map_df = filtered_df[["Latitude", "Longitude"]].rename(
        columns={"Latitude": "lat", "Longitude": "lon"}
    )
    st.map(map_df)
else:
    st.info("No se puede mostrar el mapa porque no hay datos con los filtros actuales.")

##