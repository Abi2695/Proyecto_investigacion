# Proyecto_investigacion
El repositorio es la aplicación del modelo random forest para predecir paras de producción.
# Random Forest para anticipar **PARO MAQUINA** (líneas 6 y 7) — README reproducible

Este proyecto/notebook (`Random_Forest_AF.ipynb`) entrena y evalúa modelos **Random Forest** para **predecir si ocurrirá un paro de máquina** (`PARO MAQUINA` ∈ {0,1}) a partir de variables operacionales de producción.

Incluye:

- Carga de datos desde un archivo Excel.
- Limpieza/transformación de variables de fecha y hora (con tratamiento robusto de formatos de Excel).
- Análisis descriptivo y exploratorio (EDA).
- Entrenamiento de Random Forest **sin** y **con** balanceo de datos (SMOTE).
- Ajuste de umbral de decisión usando *precision-recall curve*.
- Guardado del pipeline final (preprocesamiento + SMOTE + modelo) y del umbral óptimo.

---

## 1) Requisitos

### 1.1. Software

- Python **3.9+** (recomendado 3.10 o 3.11).
- Jupyter Notebook / JupyterLab **o** Google Colab.

### 1.2. Librerías

El notebook utiliza principalmente:

- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn`
- `imbalanced-learn` (para SMOTE)
- `joblib`

Instalación recomendada (local):

```bash
pip install -U pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib openpyxl
```

> Nota: `openpyxl` es el motor usual para leer `.xlsx` con `pandas.read_excel()`.

### 1.3. Archivo de datos

El notebook espera un Excel con columnas (al menos) como las siguientes:

- `FECHA`
- `OF`, `LINEA`, `TURNO`
- `CODIGO`, `PRODUCTO`, `SUPERVISOR`
- `INICIO_PROCESO`, `FIN_PROCESO`
- `TIEMPO ALM/MER`
- `DURACION (HORAS)` (se compara contra duración calculada)
- `UNIDADES PRODUCIDAS`, `PRODUCCION X HORA`
- `PARO MAQUINA` (target)

En esta conversación, el archivo se encuentra en:

- `Produccion linea 6 y 7 mayo a nov 2025.xlsx`

---

## 2) Cómo ejecutar (dos opciones)

### Opción A — Google Colab (igual que el notebook)

1. Abrir `Random_Forest_AF (1).ipynb` en Colab.
2. Montar Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

3. Definir `ruta` apuntando al Excel en Drive, por ejemplo:

```python
ruta = "/content/drive/MyDrive/Tesis Maestria UDLA/Base de datos/Produccion linea 6 y 7 mayo a nov 2025.xlsx"
df = pd.read_excel(ruta)
```

4. Ejecutar todas las celdas en orden.

**Archivos generados (Drive):**

- `/content/drive/MyDrive/Tesis Maestria UDLA/modelo_final_rf_smote.joblib`
- `/content/drive/MyDrive/Tesis Maestria UDLA/umbral_optimo.txt`

### Opción B — Ejecución local (Jupyter)

1. Clonar/copiar el notebook y el Excel a una carpeta.
2. Crear entorno (opcional pero recomendado):

```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

3. Instalar dependencias:

```bash
pip install -U pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib openpyxl
```

4. En el notebook, **reemplazar** la parte de Google Drive por una ruta local, por ejemplo:

```python
ruta = "./Produccion linea 6 y 7 mayo a nov 2025.xlsx"
df = pd.read_excel(ruta)
```

5. Cambiar las rutas donde se guardan modelo/umbral (si no usas Drive), por ejemplo:

```python
modelfinal_filename = "./modelo_final_rf_smote.joblib"
umbral_filename = "./umbral_optimo.txt"
```

---

## 3) Reproducibilidad (importante)

El notebook fija `random_state=42` en:

- `RandomForestClassifier(..., random_state=42)`
- `SMOTE(random_state=42)`

y usa partición **temporal** (por fecha) en lugar de un split aleatorio:

- 80% más antiguo → `train`
- 20% más reciente → `test`

Para que los resultados sean lo más reproducibles posible:

- Mantén las mismas versiones de librerías.
- No cambies el orden de las filas antes del `sort_values("FECHA")`.
- Ejecuta las celdas en orden.

### Guardar versiones (recomendado)

En local, puedes generar un archivo de versiones:

```bash
pip freeze > requirements.txt
```

Y luego reproducir el entorno con:

```bash
pip install -r requirements.txt
```

---

## 4) Paso a paso del notebook

A continuación se describe lo que hace cada bloque (equivale a las secciones del notebook).

### 4.1 Importación de librerías

Se importan `pandas`, `numpy`, `matplotlib` y, más adelante, `seaborn`, `sklearn`, `imblearn` y `joblib`.

### 4.2 Carga del archivo Excel

- Montaje de Drive (si aplica).
- Lectura con `pd.read_excel(ruta)`.
- Normalización de nombres de columnas:
  - `strip()`
  - reemplazo de espacios múltiples por uno.

Se imprimen:

- Dimensión del dataset (`df.shape`)
- Lista de columnas
- Tipos de datos
- Primeras filas

### 4.3 Limpieza y transformación de fechas/horas

1) **`FECHA`** se convierte a `datetime`:

```python
df2["FECHA"] = pd.to_datetime(df2["FECHA"], errors="coerce")
```

2) Se crea una función robusta `time_to_seconds()` para convertir `INICIO_PROCESO` y `FIN_PROCESO` a **segundos desde medianoche**, soportando:

- Horas guardadas como número en Excel (fracción del día).
- Texto o datetime (por ejemplo `1900-01-01 00:05:00`).
- Texto tipo `HH:MM:SS` usando `to_timedelta`.

3) Se construyen timestamps completos:

- `inicio_dt = FECHA + inicio_sec`
- `fin_dt = FECHA + fin_sec`

4) Se corrige el caso en que la producción **cruza medianoche** (cuando `fin_sec < inicio_sec`):

- `fin_dt = fin_dt + 1 día`

5) Se calcula una duración confiable:

- `duracion_calc_horas = (fin_dt - inicio_dt)` en horas.

Además, el notebook muestra una muestra comparativa entre `DURACION (HORAS)` y `duracion_calc_horas`.

### 4.4 Verificación del balance de clases

Se revisa la distribución de `PARO MAQUINA`:

- Conteo y porcentaje.
- Tasa de paro por `LINEA` y por `TURNO`.

Esto ayuda a justificar (o no) el uso de técnicas de balanceo como **SMOTE**.

### 4.5 Análisis descriptivo y exploratorio (EDA)

Se crean variables adicionales para explorar patrones:

- `dia_semana` (0=lunes, ..., 6=domingo)
- `mes`
- `inicio_hora`, `inicio_min`
- `alm_mer_min`: `TIEMPO ALM/MER` convertido a minutos
- `duracion_horas_final` (equivalente a `duracion_calc_horas`)

Luego se realiza:

- `describe()` sobre variables numéricas + conteo de valores faltantes.
- Gráfico de distribución de `PARO MAQUINA`.
- Boxplots de variables numéricas vs. `PARO MAQUINA` (si existen en el dataset).
- Matriz de correlación **Spearman** para variables numéricas.
- Tendencia temporal: tasa de paro mensual.

### 4.6 Preparación de datos para el modelo

Se construye un dataset `df_model` con **features permitidas** para anticipar el paro (evita variables que pudieran causar fuga de información).

Features usadas:

- `OF`, `LINEA`, `TURNO`
- `CODIGO`, `PRODUCTO`, `SUPERVISOR`
- `inicio_hora`, `inicio_min`
- `dia_semana`, `mes`
- `alm_mer_min`

`FECHA` se conserva solo para **ordenar temporalmente**, pero se excluye del entrenamiento.

### 4.7 Split temporal Train/Test

- Se ordena por `FECHA`.
- Se define un corte al 80% del dataset:

```python
cut = int(len(df_model) * 0.8)
train_df = df_model.iloc[:cut]
test_df  = df_model.iloc[cut:]
```

Se reporta el rango de fechas en train/test y la distribución de la clase en cada partición.

### 4.8 Pipeline de preprocesamiento

Se separan columnas por tipo:

- Numéricas → imputación por mediana (`SimpleImputer(strategy="median")`).
- Categóricas → imputación por moda + one-hot encoding (`OneHotEncoder(handle_unknown="ignore")`).

Esto se implementa con `ColumnTransformer` y se integra a un `Pipeline`.

### 4.9 Random Forest sin SMOTE (3 corridas)

Se evalúan 3 configuraciones (con `class_weight="balanced"`):

1. `RF_noSMOTE_1_baseline`
2. `RF_noSMOTE_2_regularizado`
3. `RF_noSMOTE_3_prof_moderada`

Métricas que imprime el notebook:

- ROC-AUC
- Matriz de confusión (umbral 0.5)
- `classification_report` (precision/recall/f1)

Además, se construye un resumen (DataFrame) ordenado por ROC-AUC.

### 4.10 Random Forest con SMOTE (3 corridas)

Se repiten 3 configuraciones, pero ahora con un pipeline `imblearn`:

`preprocess -> SMOTE -> model`

Experimentos:

1. `RF_SMOTE_1_baseline`
2. `RF_SMOTE_2_regularizado`
3. `RF_SMOTE_3_prof_moderada`

Se imprimen las mismas métricas y un resumen por ROC-AUC.

### 4.11 Ajuste del umbral de decisión

Con el **mejor modelo SMOTE** (en el notebook: `RF_SMOTE_1_baseline`), se calculan:

- `precision_recall_curve(y_test, proba)`
- F1 para cada umbral
- Umbral que maximiza F1

Luego se vuelve a evaluar el modelo con ese umbral:

- Matriz de confusión
- Reporte de clasificación

> Importante: Si tu objetivo principal es **maximizar recall** (minimizar falsos negativos), puedes modificar el criterio: por ejemplo elegir el umbral que logre un recall mínimo deseado y luego maximizar precisión o F1 dentro de esa región.

### 4.12 Comparación de métricas: umbral 0.5 vs umbral óptimo

El notebook incluye una comparación gráfica (barras) entre métricas para clase 1:

- Precision
- Recall
- F1

**Nota:** en esa celda, los valores están ingresados manualmente (hard-coded). Si cambias datos/semillas/modelos, actualiza esos números para que reflejen tu ejecución real.

### 4.13 Importancia de variables

Se carga el pipeline guardado y se extraen:

- El `model.feature_importances_`
- Los nombres de features después del preprocesamiento (numéricas + one-hot de categóricas)

Se grafica la importancia para interpretar qué variables (y qué categorías) aportan más al modelo.

### 4.14 Guardado del modelo final y umbral

Se guardan:

- Pipeline completo (`joblib.dump`):
  - `modelo_final_rf_smote.joblib`
- Umbral óptimo a un `.txt`:
  - `umbral_optimo.txt`

En Colab, se guardan en Drive bajo `.../Tesis Maestria UDLA/`.

### 4.15 Cómo usar el modelo guardado para predecir

Pasos:

1. Cargar el pipeline con `joblib.load()`.
2. Leer el umbral desde `umbral_optimo.txt`.
3. Preparar un DataFrame nuevo `X_new` con las **mismas columnas de features** que se usaron en entrenamiento.
4. Obtener probabilidades con:

```python
proba = loaded_pipeline.predict_proba(X_new)[:, 1]
```

5. Convertir a predicción final usando el umbral:

```python
pred = (proba >= loaded_optimal_threshold).astype(int)
```

---

## 5) Estructura de artefactos (salidas)

- `modelo_final_rf_smote.joblib`: pipeline completo entrenado (preprocesamiento + SMOTE + RandomForest).
- `umbral_optimo.txt`: umbral numérico (float) calculado para maximizar F1 (según el notebook).

---

## 6) Notas y recomendaciones

- **Split temporal**: es apropiado cuando se quiere simular predicción hacia el futuro y evitar fuga de información.
- **Valores faltantes**: se imputan automáticamente (mediana para numéricas, moda para categóricas).
- **Celdas con valores manuales**: la comparación de métricas por umbral (celda de barras) usa números hard-coded. Para reproducibilidad estricta, conviene calcularlos directamente desde `y_test`, `proba_best_smote` y el `optimal_threshold`.

---

## 7) Referencia rápida (comandos)

Instalar dependencias:

```bash
pip install -U pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib openpyxl
```

Abrir notebook local:

```bash
jupyter lab
# o
jupyter notebook
```

