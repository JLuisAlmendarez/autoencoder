# Convolutional Variational Autoencoder for Shoe Image Generation

Autoencoder Variacional Convolucional (CVAE) implementado en TensorFlow/Keras para aprender una representación latente de imágenes de calzado y generar nuevas imágenes a partir de esa representación. El dataset se construyó mediante *web scraping* de imágenes de zapatos de StockX.

> 📄 Reporte académico completo (metodología, arquitectura y resultados) disponible en [`Report/`](./Report).

## Descripción

El proyecto entrena un **CVAE** capaz de:

- Codificar imágenes de zapatos (64x64 px) en un espacio latente comprimido.
- Reconstruir imágenes a partir de ese espacio latente.
- Generar nuevas imágenes de zapatos muestreando puntos del espacio latente.

El trabajo incluye además un análisis exploratorio del dataset (distribución de tamaños y perfiles de color) y un perfilamiento de rendimiento del entrenamiento con NVIDIA Nsight Compute.

## Dataset

Las imágenes originales (800x571 px, scrapeadas de StockX) y su versión preprocesada (64x64 px, normalizada y dividida en train/test) están disponibles en Google Drive:

🔗 [Dataset de zapatos](https://drive.google.com/drive/folders/1XJCrXah3QdzCXaflQRaDx93pS9XhLr3x?usp=sharing)

Descarga la carpeta `Dataset/` y colócala en la raíz del proyecto (ver estructura abajo) antes de ejecutar los notebooks.

## Estructura del proyecto

```
Image Reconstruction
├── README.md                        <- Este archivo.
│
├── Code
│   ├── design
│   │   ├── cvae.ipynb                <- Notebook principal: arquitectura, entrenamiento, evaluación y generación.
│   │   └── cvae_model/                <- Modelo entrenado guardado (formato TensorFlow SavedModel).
│   │       ├── saved_model.pb
│   │       └── variables/
│   │
│   ├── drafts
│   │   └── experimental_classes.py   <- Código experimental y prototipos de clases (no productivo).
│   │
│   ├── preprocessing
│   │   ├── exploration.ipynb         <- Exploración del dataset: perfiles de color, distribución de tamaños.
│   │   ├── preprocessing.ipynb       <- Redimensionado, normalización y split train/test.
│   │   └── ternary_density.html      <- Visualización interactiva de densidad RGB.
│   │
│   └── profiling
│       ├── cvae.py                    <- Versión en script del modelo, usada para las corridas de profiling.
│       ├── ncu-report.txt             <- Reporte de NVIDIA Nsight Compute (resumen en texto).
│       └── nsight-report.json         <- Reporte completo de Nsight (archivo grande, no versionado en git).
│
├── Dataset                           <- Imágenes de zapatos scrapeadas de StockX (descargar de Google Drive).
│   ├── ZAPATOS/                       <- Imágenes originales descargadas (800x571 px).
│   └── preprocessed/                  <- Imágenes redimensionadas (64x64), normalizadas y separadas en train/test.
│
└── Report
    ├── Log File 1.tex                <- Fuente LaTeX del reporte académico.
    ├── Log File 1.pdf                <- Reporte compilado en PDF.
    └── Images/                        <- Figuras utilizadas en el reporte.
```


**Dependencias principales** (`requirements.txt`):

| Paquete | Uso |
|---|---|
| `tensorflow>=2.15` | Construcción y entrenamiento del CVAE |
| `numpy`, `pandas` | Manipulación de datos |
| `matplotlib`, `plotly` | Visualización estática e interactiva |
| `scikit-learn`, `scipy` | Utilidades de preprocesamiento y métricas |
| `Pillow` | Carga y manipulación de imágenes |
| `colorgram.py` | Extracción de paletas de color para el análisis exploratorio |
| `gputil`, `psutil` | Monitoreo de uso de GPU/CPU durante el entrenamiento |

> Si vas a ejecutar el entrenamiento en GPU, asegúrate de tener instalados los drivers de CUDA/cuDNN compatibles con la versión de TensorFlow indicada.

## Uso

### 1. Exploración y preprocesamiento

Antes de entrenar, procesa el dataset descargado con los notebooks en `Code/preprocessing/`:

1. `exploration.ipynb` — analiza distribución de tamaños y perfiles de color de las imágenes originales.
2. `preprocessing.ipynb` — redimensiona las imágenes a 64x64, las normaliza y genera el split de train/test en `Dataset/preprocessed/`.

### 2. Entrenamiento del modelo

Abre y ejecuta `Code/design/cvae.ipynb`. Este notebook contiene:

- Definición de la arquitectura del encoder/decoder convolucional.
- Ciclo de entrenamiento del CVAE.
- Evaluación de la reconstrucción sobre el set de test.
- Generación de nuevas imágenes muestreando el espacio latente.

Al finalizar, el modelo entrenado se guarda en `Code/design/cvae_model/` (formato TensorFlow SavedModel).

### 3. Uso del modelo entrenado

Puedes cargar el modelo ya entrenado directamente sin reentrenar:

```python
import tensorflow as tf

model = tf.keras.models.load_model("Code/design/cvae_model")
```

## Profiling

`Code/profiling/cvae.py` es una versión en script (no notebook) del modelo, usada para perfilar el rendimiento del entrenamiento en GPU con **NVIDIA Nsight Compute**:

```bash
ncu --export ncu-report python Code/profiling/cvae.py
```

Los resultados de estas corridas están en `ncu-report.txt` (resumen legible) y `nsight-report.json` (reporte completo, no incluido en el repositorio por su tamaño).

## Reporte

El análisis completo —justificación, arquitectura, hiperparámetros, resultados y conclusiones— está documentado en `Report/`:

- [`Log File 1.pdf`](./Report/Log%20File%201.pdf) — versión compilada.
- [`Log File 1.tex`](./Report/Log%20File%201.tex) — fuente LaTeX, editable.
