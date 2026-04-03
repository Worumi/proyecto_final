import pandas as pd
import numpy as np

def data_preprocessing(df: pd.DataFrame) -> pd.DataFrame:

    # Se transforma el dataset
    df["buy_price_log"] = np.log1p(df["buy_price"])
    
    # Eliminación de outliers
    q1_buy_price = df["buy_price_log"].quantile(0.25)
    q3_buy_price = df["buy_price_log"].quantile(0.75)
    IQR_buy_price = q3_buy_price - q1_buy_price
    lower_bound = q1_buy_price - 1.5 * IQR_buy_price
    upper_bound = q3_buy_price + 1.5 * IQR_buy_price

    # Se Elimina los outliers del target
    df.drop(df[(df["buy_price_log"] < lower_bound) | (df["buy_price_log"] > upper_bound)].index,inplace=True)

    # Se eliminan las columnas "buy_price_by_area", "Precio_por_m2"
    columns_to_drop = ["buy_price_by_area", "Precio_por_m2"]
    df.drop(columns=columns_to_drop, inplace=True)

    # Se eleminan valores negativos en las columnas relevantes
    columns_to_check = ["sq_mt_built", "n_rooms", "n_bathrooms", "rent_price", "buy_price"]
    
    for column in columns_to_check:
        df[column].astype(np.float32)
        df.drop(df.loc[(df[column]) <= 0].index,inplace=True)

    return df