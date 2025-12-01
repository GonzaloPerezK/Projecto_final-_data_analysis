import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

#Configuración de la página

st.set_page_config(
    page_title="California Housing – Análisis Interactivo",
    page_icon="🏡",
    layout="wide",
)

#Estilo de Matplotlib

plt.style.use("seaborn-v0_8")

PRIMARY_COLOR = "#2274A5"
SECONDARY_COLOR = "#F75C03"

#Carga del dataset

@st.cache_data
def load_data() -> pd.DataFrame:
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()
    return df


df_california = load_data()


#Titulos
st.title("🏡 Análisis Interactivo de Precios de Vivienda en California")

st.write(
    "Aplicación interactiva basada en el dataset California Housing "
    "de scikit-learn (censo de 1990)."
)

st.divider()

#Vista general del dataset

st.subheader("Vista general del dataset")

col_left, col_right = st.columns((2, 1))

with col_left:
    st.write("Primeras 5 filas del dataset:")
    st.dataframe(df_california.head())

with col_right:
    st.write("Información general y tipos de datos:")
    st.write(df_california.dtypes.to_frame("dtype"))

st.write("Valores faltantes por columna:")
st.dataframe(df_california.isna().sum().to_frame("n_nulos"))

st.divider()

#Filtros interactivos

st.sidebar.header("Filtros de exploración")
st.sidebar.write(
    "Útiliza filtros para enfocar el análisis en un subconjunto del dataset."
)

age_min = int(df_california["HouseAge"].min())
age_max = int(df_california["HouseAge"].max())

age_range = st.sidebar.slider(
    "Rango de Edad Mediana de la Casa (HouseAge)",
    min_value=age_min,
    max_value=age_max,
    value=(age_min, age_max),
    step=1,
)

#Filtro por latitud mínima

lat_min_global = float(df_california["Latitude"].min())
lat_max_global = float(df_california["Latitude"].max())

st.sidebar.write("Filtro por vecindario (Latitud mínima)")
min_latitude = st.sidebar.number_input(
    "Latitud mínima",
    min_value=lat_min_global,
    max_value=lat_max_global,
    value=lat_min_global,
    step=0.5,
    help="Se incluyen solo los grupos de bloques con Latitude mayor o igual a este valor.",
)

#Número de bins del histograma

n_bins = st.sidebar.slider(
    "Número de bins del histograma",
    min_value=10,
    max_value=60,
    value=30,
    step=5,
)

#Aplicar filtros

filtered_df = df_california[
    (df_california["HouseAge"] >= age_range[0])
    & (df_california["HouseAge"] <= age_range[1])
    & (df_california["Latitude"] >= min_latitude)
]

#Indicadores globales vs filtrados

st.subheader("Indicadores clave de MedHouseVal")

median_global = df_california["MedHouseVal"].median()
range_global = df_california["MedHouseVal"].max() - df_california["MedHouseVal"].min()

if not filtered_df.empty:
    median_filtered = filtered_df["MedHouseVal"].median()
    range_filtered = filtered_df["MedHouseVal"].max() - filtered_df["MedHouseVal"].min()
else:
    median_filtered = np.nan
    range_filtered = np.nan

col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

with col_kpi1:
    st.metric("Mediana global MedHouseVal", f"{median_global:.3f}")

with col_kpi2:
    st.metric(
        "Mediana filtrada MedHouseVal",
        f"{median_filtered:.3f}" if not np.isnan(median_filtered) else "—",
    )

with col_kpi3:
    st.metric("Rango global (máx - mín)", f"{range_global:.3f}")

with col_kpi4:
    st.metric(
        "Rango filtrado (máx - mín)",
        f"{range_filtered:.3f}" if not np.isnan(range_filtered) else "—",
    )

st.caption(
    "Los indicadores filtrados se calculan con las observaciones que cumplen los filtros de la barra lateral."
)

st.divider()


#Datos filtrados + resumen descriptivo solicitado

st.subheader("Datos filtrados")

st.write(
    f"Filas resultantes después de aplicar los filtros: {filtered_df.shape[0]}"
)
st.dataframe(filtered_df.head())

st.subheader("Resumen descriptivo de MedHouseVal (datos filtrados)")

if not filtered_df.empty:
    mediana_valor = filtered_df["MedHouseVal"].median()
    rango_valor = filtered_df["MedHouseVal"].max() - filtered_df["MedHouseVal"].min()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mediana de MedHouseVal (filtrado)", f"{mediana_valor:.3f}")
    with col2:
        st.metric("Rango (máx - mín) de MedHouseVal (filtrado)", f"{rango_valor:.3f}")
else:
    st.warning(
        "No hay datos con los filtros seleccionados. Ajustá los filtros en la barra lateral."
    )

st.divider()

#Visualizaciones dinámicas

st.subheader("Distribución de MedHouseVal")

if not filtered_df.empty:
    fig_hist, ax_hist = plt.subplots(figsize=(7, 4))
    ax_hist.hist(
        filtered_df["MedHouseVal"],
        bins=n_bins,
        color=PRIMARY_COLOR,
        edgecolor="white",
        alpha=0.9,
    )
    ax_hist.set_xlabel("MedHouseVal (cientos de miles de USD)")
    ax_hist.set_ylabel("Frecuencia")
    ax_hist.set_title("Histograma de MedHouseVal (datos filtrados)")
    st.pyplot(fig_hist)
else:
    st.info(
        "No se puede mostrar el histograma porque no hay datos con los filtros actuales."
    )

#Scatter plot MedInc vs MedHouseVal

st.subheader("Relación entre MedInc y MedHouseVal")

if not filtered_df.empty:
    fig_scatter, ax_scatter = plt.subplots(figsize=(7, 4))
    ax_scatter.scatter(
        filtered_df["MedInc"],
        filtered_df["MedHouseVal"],
        alpha=0.35,
        s=15,
        color=SECONDARY_COLOR,
        edgecolors="none",
    )
    ax_scatter.set_xlabel("MedInc (decenas de miles de USD)")
    ax_scatter.set_ylabel("MedHouseVal (cientos de miles de USD)")
    ax_scatter.set_title("MedInc vs MedHouseVal (datos filtrados)")
    st.pyplot(fig_scatter)
else:
    st.info(
        "No se puede mostrar el gráfico de dispersión porque no hay datos con los filtros actuales."
    )

# Matriz de correlación (extra analítico)
st.subheader("Matriz de correlación (subconjunto filtrado)")

variables_correlacion = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "MedHouseVal",
]

if not filtered_df.empty:
    corr = filtered_df[variables_correlacion].corr()

    fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
    im = ax_corr.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

    ax_corr.set_xticks(range(len(variables_correlacion)))
    ax_corr.set_yticks(range(len(variables_correlacion)))
    ax_corr.set_xticklabels(variables_correlacion, rotation=45, ha="right")
    ax_corr.set_yticklabels(variables_correlacion)

    for i in range(len(variables_correlacion)):
        for j in range(len(variables_correlacion)):
            ax_corr.text(
                j,
                i,
                f"{corr.values[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=7,
            )

    fig_corr.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)
    ax_corr.set_title("Correlación entre variables seleccionadas")
    st.pyplot(fig_corr)
else:
    st.info(
        "No se puede calcular la matriz de correlación porque no hay datos con los filtros actuales."
    )

st.divider()

# Mapa geográfico

st.subheader("Mapa geográfico de las viviendas filtradas")

if not filtered_df.empty:
    st.caption("Cada punto representa un grupo de bloques según Latitude y Longitude.")
    map_df = filtered_df[["Latitude", "Longitude"]].rename(
        columns={"Latitude": "lat", "Longitude": "lon"}
    )
    st.map(map_df)
else:
    st.info("No se puede mostrar el mapa porque no hay datos con los filtros actuales.")