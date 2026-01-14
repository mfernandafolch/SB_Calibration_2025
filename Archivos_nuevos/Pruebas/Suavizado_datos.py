"""
SUAVIZADO DE DATOS EXPERIMENTALES (Savitzky–Golay) PARA ESTIMACIÓN DE PARÁMETROS
-------------------------------------------------------------------------------

Qué hace este script:
1) Lee un Excel desde la hoja: "Prov Sensores"
2) Extrae columnas:
   - tiempo (días):      "indice_tiempo_dias"  (col D en tu archivo)
   - densidad:           "densidad"            (col H)
   - temp_mosto:         "temp_mosto"          (col E)
   - temp_sombrero:      "temp_sombrero"       (col F)

3) Calcula temp_promedio con DATOS ORIGINALES:
      temp_promedio_raw = (temp_mosto + temp_sombrero)/2
   (Si uno de los dos está vacío en un tiempo, el promedio queda NaN.)

4) Suaviza (Savitzky–Golay) estas series:
   - densidad
   - temp_mosto
   - temp_sombrero
   - temp_promedio_raw

5) Dibuja 3 gráficos (doble eje):
   - Gráfico 1: datos crudos
   - Gráfico 2: datos suavizados
   - Gráfico 3: comparación (raw punteado, smooth continuo)

Notas importantes:
- Savitzky–Golay necesita ventanas impares y suficientes puntos.
- Tus datos tienen huecos (NaNs). Este script suaviza por TRAMOS continuos.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from pathlib import Path

# =========================================================
# 0) CARGA DEL ARCHIVO EXCEL DESDE LA CARPETA "data"
# =========================================================

# Ruta base del proyecto (donde está este script)
BASE_DIR = Path(__file__).resolve().parent

# Carpeta data (al mismo nivel que el código)
DATA_DIR = BASE_DIR / "Data"

# Nombre del archivo Excel (solo el nombre, no la ruta completa)
excel_filename = "Data ME 25 Q. AGUA estanque 233.xlsx"

# Ruta completa al Excel
excel_path = DATA_DIR / excel_filename


# OPCIÓN B (si estás en Jupyter y quieres que te aparezca un selector):
# Descomenta estas líneas:
# from tkinter import Tk
# from tkinter.filedialog import askopenfilename
# Tk().withdraw()
# excel_path = askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls")])

sheet_name = "Prov Sensores"

# Columnas por NOMBRE (asumiendo que en el Excel están tal cual)
col_t  = "indice_tiempo_dias"
col_rho = "densidad"
col_tm  = "temp_mosto"
col_ts  = "temp_sombrero"

# =========================================================
# 1) PARÁMETROS DEL SUAVIZADO (AJUSTABLES)
# =========================================================
# window_length debe ser IMPAR y >= polyorder+2
# Si los datos están muy ruidosos: aumenta window_length
# Si te está "aplanando" demasiado: reduce window_length
sg_window = 61
sg_poly = 2

# =========================================================
# 2) LECTURA Y LIMPIEZA DE DATOS
# =========================================================
# Verificar que el archivo existe
if not excel_path.exists():
    print(f"❌ Error: No se encontró el archivo en: {excel_path}")
    print(f"\nArchivos disponibles en {DATA_DIR}:")
    for f in DATA_DIR.glob("*.xlsx"):
        print(f"  - {f.name}")
    raise FileNotFoundError(f"El archivo {excel_filename} no existe en {DATA_DIR}")

df_raw = pd.read_excel(str(excel_path), sheet_name=sheet_name)

# Nos quedamos solo con columnas relevantes
df = df_raw[[col_t, col_rho, col_tm, col_ts]].copy()

# Aseguramos numéricos (si vienen como texto, se convierten; lo inválido queda NaN)
for c in [col_t, col_rho, col_tm, col_ts]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Ordenamos por tiempo y removemos filas sin tiempo
df = df.dropna(subset=[col_t]).sort_values(col_t).reset_index(drop=True)

# =========================================================
# 3) TEMP PROMEDIO (RAW) Y FUNCIÓN DE SUAVIZADO CON NaNs
# =========================================================
# Promedio con datos originales (RAW)
# Si te interesa "promedio de lo disponible" aunque falte uno, usa:
# df["temp_promedio_raw"] = df[[col_tm, col_ts]].mean(axis=1, skipna=True)
df["temp_promedio_raw"] = df[[col_tm, col_ts]].mean(axis=1)

def savgol_smooth_with_nans(x, window_length=21, polyorder=3):
    """
    Suaviza una serie con Savitzky–Golay incluso si tiene NaNs.
    Estrategia:
    - Identifica TRAMOS continuos (sin NaNs).
    - Aplica Savitzky–Golay en cada tramo por separado.
    - Mantiene NaNs donde no hay datos.
    """
    x = np.asarray(x, dtype=float)
    y = np.full_like(x, np.nan)

    valid = ~np.isnan(x)
    if not np.any(valid):
        return y

    idx = np.where(valid)[0]

    # Si hay pocos datos válidos, devolvemos tal cual (solo donde hay datos)
    if len(idx) < max(window_length, polyorder + 2):
        y[valid] = x[valid]
        return y

    # Encontrar cortes entre índices válidos (cuando la diferencia > 1)
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(idx) - 1]

    for s, e in zip(starts, ends):
        seg_idx = idx[s:e+1]
        seg = x[seg_idx]

        # Ajuste de ventana si el tramo es corto (mantener impar)
        wl = window_length
        if len(seg) < wl:
            wl = len(seg) if len(seg) % 2 == 1 else len(seg) - 1

        # Si no alcanza para filtrar, copiamos sin suavizar
        if wl < polyorder + 2 or wl < 3:
            y[seg_idx] = seg
        else:
            y[seg_idx] = savgol_filter(seg, window_length=wl, polyorder=polyorder)

    return y

# Aplicamos suavizado
df["densidad_smooth"] = savgol_smooth_with_nans(df[col_rho], sg_window, sg_poly)
df["temp_mosto_smooth"] = savgol_smooth_with_nans(df[col_tm], sg_window, sg_poly)
df["temp_sombrero_smooth"] = savgol_smooth_with_nans(df[col_ts], sg_window, sg_poly)
df["temp_promedio_smooth"] = savgol_smooth_with_nans(df["temp_promedio_raw"], sg_window, sg_poly)

# =========================================================
# 4) FUNCIÓN DE PLOTEO (DOBLE EJE)
# =========================================================
def plot_dual_axis(title, t, dens, tm, ts, tp,
                   overlay=None,
                   raw_linestyle="--",
                   smooth_linestyle="-"):
    """
    Grafica densidad (eje izq) y temperaturas (eje der),
    usando colores distintos y consistentes.
    """

    # --- definición explícita de colores ---
    color_dens = "tab:blue"
    color_tm = "tab:red"
    color_ts = "tab:green"
    color_tp = "tab:orange"

    fig, ax1 = plt.subplots(figsize=(12, 5))

    # ===== eje izquierdo: densidad =====
    ax1.plot(
        t, dens,
        color=color_dens,
        linestyle=smooth_linestyle,
        linewidth=2.2,
        label="densidad"
    )
    ax1.set_xlabel("Tiempo (días)")
    ax1.set_ylabel("Densidad")
    ax1.grid(True, alpha=0.3)

    # ===== eje derecho: temperaturas =====
    ax2 = ax1.twinx()
    ax2.plot(
        t, tm,
        color=color_tm,
        linestyle=smooth_linestyle,
        linewidth=2,
        label="temp_mosto"
    )
    ax2.plot(
        t, ts,
        color=color_ts,
        linestyle=smooth_linestyle,
        linewidth=2,
        label="temp_sombrero"
    )
    ax2.plot(
        t, tp,
        color=color_tp,
        linestyle=smooth_linestyle,
        linewidth=2,
        label="temp_promedio"
    )
    ax2.set_ylabel("Temperatura (°C)")

    # ===== overlay RAW (punteado) =====
    if overlay is not None:
        dens_raw, tm_raw, ts_raw, tp_raw = overlay

        ax1.plot(
            t, dens_raw,
            color=color_dens,
            linestyle=raw_linestyle,
            linewidth=1.3,
            alpha=0.75,
            label="densidad (raw)"
        )
        ax2.plot(
            t, tm_raw,
            color=color_tm,
            linestyle=raw_linestyle,
            linewidth=1.3,
            alpha=0.75,
            label="temp_mosto (raw)"
        )
        ax2.plot(
            t, ts_raw,
            color=color_ts,
            linestyle=raw_linestyle,
            linewidth=1.3,
            alpha=0.75,
            label="temp_sombrero (raw)"
        )
        ax2.plot(
            t, tp_raw,
            color=color_tp,
            linestyle=raw_linestyle,
            linewidth=1.3,
            alpha=0.75,
            label="temp_promedio (raw)"
        )

    # ===== leyenda combinada =====
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower left", ncol=2)

    plt.title(title)
    plt.tight_layout()
    plt.show()

# =========================================================
# 5) GENERAR LOS 3 GRÁFICOS
# =========================================================
t = df[col_t].to_numpy()

# (1) NO suavizados
plot_dual_axis(
    "1) Datos NO suavizados (raw)",
    t=t,
    dens=df[col_rho].to_numpy(),
    tm=df[col_tm].to_numpy(),
    ts=df[col_ts].to_numpy(),
    tp=df["temp_promedio_raw"].to_numpy(),
    overlay=None,
    smooth_linestyle="-"
)

# (2) Suavizados
plot_dual_axis(
    "2) Datos SUAVIZADOS (Savitzky–Golay)",
    t=t,
    dens=df["densidad_smooth"].to_numpy(),
    tm=df["temp_mosto_smooth"].to_numpy(),
    ts=df["temp_sombrero_smooth"].to_numpy(),
    tp=df["temp_promedio_smooth"].to_numpy(),
    overlay=None,
    smooth_linestyle="-"
)

# (3) Comparación: smooth vs raw (raw punteado)
plot_dual_axis(
    "3) Comparación: RAW (punteado) vs SUAVIZADO (continuo)",
    t=t,
    dens=df["densidad_smooth"].to_numpy(),
    tm=df["temp_mosto_smooth"].to_numpy(),
    ts=df["temp_sombrero_smooth"].to_numpy(),
    tp=df["temp_promedio_smooth"].to_numpy(),
    overlay=(
        df[col_rho].to_numpy(),
        df[col_tm].to_numpy(),
        df[col_ts].to_numpy(),
        df["temp_promedio_raw"].to_numpy()
    ),
    raw_linestyle="--",
    smooth_linestyle="-"
)

# =========================================================
# 6) (OPCIONAL) GUARDAR RESULTADOS A EXCEL
# =========================================================
# Si quieres que te deje un Excel nuevo con columnas raw + suavizadas:
# out_path = "datos_suavizados_savgol.xlsx"
# df.to_excel(out_path, index=False)
# print(f"Guardado en: {out_path}")
