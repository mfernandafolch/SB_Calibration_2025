"""
BATCH SUAVIZADO DE DATOS EXPERIMENTALES CON SAVITZKY–GOLAY
========================================================

Este script:
- Recorre múltiples archivos Excel en una carpeta
- Extrae datos de sensores desde la hoja "Prov Sensores"
- Calcula temperatura promedio (RAW)
- Aplica suavizado Savitzky–Golay robusto (maneja NaNs)
- Retorna los datos en una estructura eficiente (dict de DataFrames)
- (Opcional) guarda un Excel suavizado por archivo

Pensado para:
- estimación de parámetros
- pipelines de modelos dinámicos
- uso posterior en CasADi / SciPy / Pyomo
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter
from typing import Optional, Dict, Union


# =========================================================
# 1) FUNCIÓN DE SUAVIZADO ROBUSTO (MANEJA NaNs)
# =========================================================
def savgol_smooth_with_nans(
    x: np.ndarray,
    window_length: int,
    polyorder: int
) -> np.ndarray:
    """
    Aplica Savitzky–Golay a una serie que puede contener NaNs.
    El suavizado se realiza por tramos continuos válidos.
    """
    x = np.asarray(x, dtype=float)
    y = np.full_like(x, np.nan)

    valid = ~np.isnan(x)
    if not np.any(valid):
        return y

    idx = np.where(valid)[0]

    # Si hay pocos datos, no se suaviza
    if len(idx) < max(window_length, polyorder + 2):
        y[valid] = x[valid]
        return y

    # Identificar tramos continuos
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(idx) - 1]

    for s, e in zip(starts, ends):
        seg_idx = idx[s:e + 1]
        seg = x[seg_idx]

        wl = window_length
        if len(seg) < wl:
            wl = len(seg) if len(seg) % 2 == 1 else len(seg) - 1

        if wl < polyorder + 2 or wl < 3:
            y[seg_idx] = seg
        else:
            y[seg_idx] = savgol_filter(
                seg,
                window_length=wl,
                polyorder=polyorder
            )

    return y


# =========================================================
# 2) PROCESAMIENTO DE UN ARCHIVO INDIVIDUAL
# =========================================================
def process_single_excel(
    file_path: Path,
    sheet_name: str,
    col_time: str,
    col_density: str,
    col_temp_mosto: str,
    col_temp_sombrero: str,
    sg_params_density: dict,
    sg_params_temp: dict
) -> pd.DataFrame:
    """
    Lee un Excel, suaviza los datos y retorna un DataFrame completo.
    """

    df_raw = pd.read_excel(file_path, sheet_name=sheet_name)

    # Nos quedamos solo con columnas relevantes
    df = df_raw[
        [col_time, col_density, col_temp_mosto, col_temp_sombrero]
    ].copy()

    # Conversión robusta a numérico
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Orden temporal
    df = df.dropna(subset=[col_time]).sort_values(col_time).reset_index(drop=True)

    # Temperatura promedio RAW
    df["temp_promedio_raw"] = df[
        [col_temp_mosto, col_temp_sombrero]
    ].mean(axis=1)

    # Suavizado
    df["densidad_smooth"] = savgol_smooth_with_nans(
        df[col_density].values,
        **sg_params_density
    )

    for col in [col_temp_mosto, col_temp_sombrero]:
        df[f"{col}_smooth"] = savgol_smooth_with_nans(
            df[col].values,
            **sg_params_temp
        )

    df["temp_promedio_smooth"] = savgol_smooth_with_nans(
        df["temp_promedio_raw"].values,
        **sg_params_temp
    )

    return df


# =========================================================
# 3) PROCESAMIENTO BATCH DE UNA CARPETA
# =========================================================
def process_folder_of_excels(
    folder_path: Path,
    output_folder: Optional[Path] = None
) -> Dict[str, pd.DataFrame]:
    """
    Procesa todos los Excel de una carpeta.
    Retorna un diccionario: {nombre_archivo: DataFrame}
    """

    # Configuración fija del proyecto
    sheet_name = "Prov Sensores"

    col_time = "indice_tiempo_dias"
    col_density = "densidad"
    col_temp_mosto = "temp_mosto"
    col_temp_sombrero = "temp_sombrero"

    # Parámetros Savitzky–Golay
    sg_params_density = {
        "window_length": 21,
        "polyorder": 3
    }

    sg_params_temp = {
        "window_length": 31,
        "polyorder": 2
    }

    results = {}

    excel_files = sorted(folder_path.glob("*.xlsx"))

    if not excel_files:
        raise FileNotFoundError("No se encontraron archivos Excel en la carpeta.")

    if output_folder is not None:
        output_folder.mkdir(parents=True, exist_ok=True)

    for file in excel_files:
        print(f"Procesando: {file.name}")

        df_smooth = process_single_excel(
            file_path=file,
            sheet_name=sheet_name,
            col_time=col_time,
            col_density=col_density,
            col_temp_mosto=col_temp_mosto,
            col_temp_sombrero=col_temp_sombrero,
            sg_params_density=sg_params_density,
            sg_params_temp=sg_params_temp
        )

        results[file.stem] = df_smooth

        # Guardado opcional
        if output_folder is not None:
            out_file = output_folder / f"{file.stem}_smooth.xlsx"
            df_smooth.to_excel(out_file, index=False)

    return results


# =========================================================
# 4) FUNCIÓN MAIN (PUNTO DE ENTRADA)
# =========================================================
def main(data_folder: Optional[Union[Path, str]] = None):
    """
    Punto de entrada del script.

    Parámetros
    - data_folder: Path o str opcional que apunta a la carpeta `data`.
      Si no se provee, se asume la carpeta `data` al mismo nivel que este archivo.
    """

    # Determinar carpeta `data` base
    if data_folder is None:
        base_data = Path(__file__).parent / "data"
    else:
        base_data = Path(data_folder)

    input_folder = base_data / "raw_excels"
    # output_folder = base_data / "smoothed_excels"
    output_folder = None

    # Si la carpeta no existe o está vacía, intentar localizar archivos .xlsx en el repo
    if not input_folder.exists() or not any(input_folder.glob("*.xlsx")):
        print(f"Ruta esperada de entrada: {input_folder!s}")
        print("No se encontraron archivos Excel en la ruta esperada. Buscando .xlsx en el proyecto...")

        repo_root = Path(__file__).parent
        found = list(repo_root.rglob("*.xlsx"))
        if found:
            # Usar la carpeta del primer archivo encontrado
            discovered_folder = found[0].parent
            print(f"Se encontraron {len(found)} archivo(s) .xlsx. Usando carpeta: {discovered_folder}")
            input_folder = discovered_folder
        else:
            print("No se encontraron archivos .xlsx en el proyecto.")
            print("Comprueba la ruta --data-folder o coloca los Excel en 'data/raw_excels' junto al script.")
            return {}

    results = process_folder_of_excels(
        folder_path=input_folder,
        output_folder=output_folder
    )

    print(f"\nProcesamiento terminado.")
    print(f"Archivos procesados: {len(results)}")

    return results


# =========================================================
# 5) EJECUCIÓN
# =========================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Suavizado batch de archivos Excel."
    )
    parser.add_argument(
        "-d",
        "--data-folder",
        help="Ruta a la carpeta 'data' (por defecto: carpeta 'data' junto al script)",
        default=None,
    )

    args = parser.parse_args()

    results = main(data_folder=args.data_folder)