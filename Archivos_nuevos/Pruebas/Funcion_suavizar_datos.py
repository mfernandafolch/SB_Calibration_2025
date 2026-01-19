"""
BATCH SUAVIZADO DE DATOS EXPERIMENTALES CON SPLINES
=================================================

Este script:
- Recorre múltiples archivos Excel en una carpeta
- Extrae datos desde la hoja "Prov Sensores"
- Estima ruido efectivo (método de residuos)
- Ajusta smoothing splines
- Submuestrea explícitamente los datos (ej. 10%)
- Retorna los sets finales en una estructura eficiente
- Genera un gráfico por archivo (original vs puntos submuestreados)

Pensado para:
- estimación de parámetros
- pipelines dinámicos
- uso posterior en CasADi / SciPy / Pyomo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from typing import Dict, Optional, Union


# =========================================================
# 1) ESTIMACIÓN DE RUIDO (MÉTODO DE RESIDUOS)
# =========================================================
def estimate_noise_residual(y, window, poly):
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(y)
    yv = y[mask]

    if len(yv) < window:
        return np.nan

    y_smooth = savgol_filter(yv, window, poly)
    return np.std(yv - y_smooth, ddof=1)


# =========================================================
# 2) PROCESAR UN ARCHIVO INDIVIDUAL (MASTER UNITARIO)
# =========================================================
def process_single_excel_spline(
    file_path: Path,
    sheet_name: str,
    col_time: str,
    col_density: str,
    col_temp_mosto: str,
    col_temp_sombrero: str,
    noise_params: dict,
    spline_params: dict,
    fraction_keep: float,
    make_plot: bool = True
) -> Dict[str, np.ndarray]:
    """
    Procesa un Excel individual y retorna los sets finales reducidos.
    """

    # --- Lectura ---
    df_raw = pd.read_excel(file_path, sheet_name=sheet_name)

    df = df_raw[
        [col_time, col_density, col_temp_mosto, col_temp_sombrero]
    ].copy()

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=[col_time]).sort_values(col_time).reset_index(drop=True)

    df["temp_promedio_raw"] = df[
        [col_temp_mosto, col_temp_sombrero]
    ].mean(axis=1)

    t = df[col_time].to_numpy()

    # --- Estimar ruido ---
    sigma_temp = estimate_noise_residual(
        df[col_temp_mosto],
        noise_params["temp"]["window"],
        noise_params["temp"]["poly"]
    )

    sigma_dens = estimate_noise_residual(
        df[col_density],
        noise_params["dens"]["window"],
        noise_params["dens"]["poly"]
    )

    N_temp = np.sum(~df[col_temp_mosto].isna())
    N_dens = np.sum(~df[col_density].isna())

    s_temp = spline_params["alpha_temp"] * N_temp * sigma_temp**2
    s_dens = spline_params["alpha_dens"] * N_dens * sigma_dens**2

    # --- Ajuste splines ---
    def fit_spline(y, s):
        mask = ~np.isnan(y)
        return UnivariateSpline(t[mask], y[mask], s=s, k=2)

    spline_dens = fit_spline(df[col_density].to_numpy(), s_dens)
    spline_tm   = fit_spline(df[col_temp_mosto].to_numpy(), s_temp)
    spline_ts   = fit_spline(df[col_temp_sombrero].to_numpy(), s_temp)
    spline_tp   = fit_spline(df["temp_promedio_raw"].to_numpy(), s_temp)

    # --- Submuestreo (respeta inicio densidad) ---
    t_start_density = df.loc[df[col_density].notna(), col_time].iloc[0]

    N_original = len(t)
    N_reduced = max(3, int(np.ceil(fraction_keep * N_original)))

    t_red = np.linspace(t_start_density, t.max(), N_reduced)

    dens_red = spline_dens(t_red)
    tm_red   = spline_tm(t_red)
    ts_red   = spline_ts(t_red)
    tp_red   = spline_tp(t_red)

    # --- Gráfico (formato gráfico 2) ---
    if make_plot:
        fig, ax1 = plt.subplots(figsize=(11, 5))
        ax2 = ax1.twinx()

        ax1.plot(t, df[col_density], "--", color="tab:blue", alpha=0.5, label="densidad raw")
        ax1.plot(t_red, dens_red, ".", color="tab:blue", ms=6, label="densidad reducida")

        ax2.plot(t, df[col_temp_mosto], "--", color="tab:red", alpha=0.5, label="temp_mosto raw")
        ax2.plot(t_red, tm_red, ".", color="tab:red", ms=6, label="temp_mosto reducida")

        ax2.plot(t, df[col_temp_sombrero], "--", color="tab:green", alpha=0.5, label="temp_sombrero raw")
        ax2.plot(t_red, ts_red, ".", color="tab:green", ms=6, label="temp_sombrero reducida")

        ax2.plot(t, df["temp_promedio_raw"], "--", color="tab:orange", alpha=0.5, label="temp_promedio raw")
        ax2.plot(t_red, tp_red, ".", color="tab:orange", ms=6, label="temp_promedio reducida")

        ax1.set_xlabel("Tiempo (días)")
        ax1.set_ylabel("Densidad")
        ax2.set_ylabel("Temperatura (°C)")
        ax1.grid(True, alpha=0.3)

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="lower left", ncol=3)

        plt.title(f"{file_path.stem} — datos originales vs submuestreados")
        plt.tight_layout()
        plt.show()

    return {
        "t": t_red,
        "densidad": dens_red,
        "temp_mosto": tm_red,
        "temp_sombrero": ts_red,
        "temp_promedio": tp_red,
        "N_original": N_original,
        "N_reduced": N_reduced
    }


# =========================================================
# 3) PROCESAMIENTO BATCH DE UNA CARPETA
# =========================================================
def process_folder_of_excels_spline(
    folder_path: Path,
    make_plots: bool = True
) -> Dict[str, Dict[str, np.ndarray]]:

    sheet_name = "Prov Sensores"

    col_time = "indice_tiempo_dias"
    col_density = "densidad"
    col_temp_mosto = "temp_mosto"
    col_temp_sombrero = "temp_sombrero"

    noise_params = {
        "temp": {"window": 81, "poly": 1},
        "dens": {"window": 51, "poly": 1}
    }

    spline_params = {
        "alpha_temp": 2.0,
        "alpha_dens": 2.0
    }

    fraction_keep = 0.10

    results = {}

    excel_files = sorted(folder_path.glob("*.xlsx"))
    if not excel_files:
        raise FileNotFoundError("No se encontraron archivos Excel en la carpeta.")

    for file in excel_files:
        print(f"Procesando: {file.name}")

        results[file.stem] = process_single_excel_spline(
            file_path=file,
            sheet_name=sheet_name,
            col_time=col_time,
            col_density=col_density,
            col_temp_mosto=col_temp_mosto,
            col_temp_sombrero=col_temp_sombrero,
            noise_params=noise_params,
            spline_params=spline_params,
            fraction_keep=fraction_keep,
            make_plot=make_plots
        )

    return results


# =========================================================
# 4) FUNCIÓN MAIN (MISMA LÓGICA QUE TU BATCH ORIGINAL)
# =========================================================
def main(data_folder: Optional[Union[Path, str]] = None):

    if data_folder is None:
        base_data = Path(__file__).parent / "Data"
    else:
        base_data = Path(data_folder)

    input_folder = base_data

    if not input_folder.exists():
        print(f"No existe la carpeta: {input_folder}")
        return {}

    results = process_folder_of_excels_spline(
        folder_path=input_folder,
        make_plots=True
    )

    print("\nProcesamiento terminado.")
    print(f"Archivos procesados: {len(results)}")

    return results


# =========================================================
# 5) EJECUCIÓN
# =========================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Suavizado batch con splines y submuestreo."
    )
    parser.add_argument(
        "-d",
        "--data-folder",
        help="Ruta a la carpeta con archivos Excel",
        default=None,
    )

    args = parser.parse_args()

    results = main(data_folder=args.data_folder)
    # 'results' es un diccionario con los datos procesados por archivo