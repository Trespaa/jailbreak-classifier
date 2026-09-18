"""
data_prep.py

Cleans, balances and divides the dataset downloaded by download_data.py
en train / validation / test.

Execute:
    python src/data_prep.py
"""

import pandas as pd                                     # tabla de datos con filtrado/limpieza por columna, en vez de bucles a mano
from pathlib import Path
from sklearn.model_selection import train_test_split    # Es la función que reparte tu tabla de datos en trozos separados — el train/validation/test
#rutas de datos
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"       #ruta para los datos de download_data.py
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" #nueva ruta para los datos procesados
OUT_DIR.mkdir(parents=True, exist_ok=True)                              #creamos el directorio de los datos procesados

#constantes de datos
RANDOM_STATE = 42    # vale cualquier numero, el 42 por un meme
MIN_CHARS = 15       # numero minimo de caracteres para que la info sea fiable

def load_and_label(): # es la función que cargará y nombrará y clasificará la información de los csv (coma separated values)
    jb = pd.read_csv(RAW_DIR / "jailbreak_prompts_2023_12_25.csv") # interpreta los documentos como una tabla usando las comas para separar columnas
    reg = pd.read_csv(RAW_DIR / "regular_prompts_2023_12_25.csv")

    jb = jb[["prompt"]].copy() # coge solo la columna prompt que es la que da la info sobre jailbreak o reg.
    jb["label"] = 1            # asignamos el numero 1 a los jailbreak para clasificarlos

    reg = reg[["prompt"]].copy()
    reg["label"] = 0

    df = pd.concat([jb, reg], ignore_index=True) #junta ambas tablas mezclando origenes pero cada una con su etiqueta pegada
                                                 # ignore_index=True es casi obligatorio para unir tablas que venían de sitios distintos, sin eso quizás bugs raros más adelante cuando alguna otra función de pandas intente usar esos índices duplicados para algo.
    return df                                    # Cierra la función devolviendo la tabla ya unida y etiquetada

def clean(df: pd.DataFrame) -> pd.DataFrame:     # El : pd.DataFrame  — dice "espero que df sea una tabla de pandas", sino no hay nada que limpiar
    df = df.dropna(subset=["prompt"]).copy()            # "drop NA" (elimina valores vacíos). subset=["prompt"] le dice: solo fijate en prompt
    df["prompt"] = df["prompt"].astype(str).str.strip() # .astype(str): fuerza a que el valor sea texto y .str.strip(): quita espacios en blanco sobrantes al principio y al final del texto
    df = df[df["prompt"].str.len() >= MIN_CHARS] #df["prompt"].str.len() calcula la longitud (número de caracteres). Compara con los minimos, te quedas solo con las filas donde el resultado fue True
    df = df.drop_duplicates(subset="prompt")     # quita los repetidos, que solo inflan porcentaje de aprendizaje
    return df

def balance(df: pd.DataFrame) -> pd.DataFrame: # balance() recorta la clase mayoritaria para que ambas tengan el mismo número de ejemplos de jb y reg
    n_min = df["label"].value_counts().min()   # cuenta cuantas veces sale cada valor
    parts = [
        group.sample(n_min, random_state=RANDOM_STATE) # group.sample(n_min, ...) coge, de cada sub-tabla, una muestra aleatoria de exactamente n_minima filas
        for _, group in df.groupby("label")    # df.groupby("label"): agrupa la tabla en "sub-tablas" separadas, una por cada valor distinto de label
    ]                                          # for _, group in ...:  es una list comprehension (una forma compacta de escribir un bucle for que construye una lista, todo en una línea.  _ significa "esta variable existe porque la sintaxis lo exige, pero la voy a ignorar"
    balanced = pd.concat(parts).sample(  # parts es una lista con 2 tablas dentro, cada una con 1,363 filas, concat las junta
        frac=1, random_state=RANDOM_STATE   # mezcla todas de forma aleatoria desd sample().
    ).reset_index(drop=True)    #  renumera los índices de 0 en adelante, limpio, sin arrastrar los índices viejos, los viejos los desecha
    return balanced

def split_dataset(df: pd.DataFrame):
    """Create deterministic stratified train/validation/test splits (70/15/15)."""
    holdout_size = round(len(df) * 0.15)
    train_val, test = train_test_split(
        df, test_size=holdout_size, stratify=df["label"], random_state=RANDOM_STATE
    )
    train, val = train_test_split(
        train_val, test_size=holdout_size,
        stratify=train_val["label"], random_state=RANDOM_STATE
    )
    return train, val, test


def main():
    df = load_and_label()
    print(f"Raw combined: {len(df)} rows")                # guarda el resultado en df, e imprime cuántas filas hay. len(df) en un DataFrame de pandas te da el número de filas

    df = clean(df)
    print(f"After cleaning/dedup: {len(df)} rows "         # Le pasa el resultado anterior a clean(), y reasigna df con la versión limpia. df['label'].value_counts() (ya la conoces, de balance()) cuenta cuántas filas hay de cada clase. .to_dict() convierte ese resultado a un diccionario normal de Python ({0: 13106, 1: 1363})
          f"({df['label'].value_counts().to_dict()})")

    df = balance(df)
    print(f"After balancing: {len(df)} rows "              # ahora con balance(). Después de esto, df debería tener las dos clases exactamente igualadas (recuerda: 2,726 filas totales, 1,363 de cada una)
          f"({df['label'].value_counts().to_dict()})")

    train, val, test = split_dataset(df)

    for name, split in [("train", train), ("val", val), ("test", test)]:
        split.to_csv(OUT_DIR / f"{name}.csv", index=False)
        print(f"{name}: {len(split)} rows -> {OUT_DIR / (name + '.csv')}") # En cada vuelta, name toma el texto y split toma la tabla correspondiente. split.to_csv(ruta, index=False) guarda la tabla como archivo CSV en disco. Index=false porq es numeracion sin importancia de pandas


if __name__ == "__main__":
    main()
