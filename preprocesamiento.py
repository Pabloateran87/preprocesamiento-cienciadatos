import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def cargar_dataset(ruta):
    return pd.read_csv(ruta)

def eliminar_duplicados(df):
    return df.drop_duplicates()

def manejar_valores_nulos(df):
    for columna in df.columns:
        if df[columna].dtype in ["int64", "float64"]:
            df[columna] = df[columna].fillna(df[columna].mean())
        else:
            df[columna] = df[columna].fillna(df[columna].mode()[0])
    return df

def codificar_categoricas(df):
    return pd.get_dummies(df, drop_first=True)

def normalizar_datos(df):
    columnas_numericas = df.select_dtypes(include=["int64", "float64"]).columns
    scaler = MinMaxScaler()
    df[columnas_numericas] = scaler.fit_transform(df[columnas_numericas])
    return df

def preprocesamiento_completo(ruta):
    df = cargar_dataset(ruta)
    df = eliminar_duplicados(df)
    df = manejar_valores_nulos(df)
    df = codificar_categoricas(df)
    df = normalizar_datos(df)
    return df

if __name__ == "__main__":
    print("Modulo de preprocesamiento ejecutado correctamente.")
    print("Workflow actualizado correctamente")
