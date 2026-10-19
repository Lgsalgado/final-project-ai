# Sistema Inteligente de Detección y Corrección de Postura en Sentadillas en Tiempo Real
**Proyecto Integrador en Inteligencia Artificial (MIAR0545)**  
**Arquitectura Backend Modular y MLOps Pipeline**

---

## 1. Estructura del Repositorio

```text
squat_pose_backend/
├── .gitignore                         # Exclusión de videos pesados, venv y cachés
├── requirements.txt                   # Dependencias con versiones estables fijadas
├── train.py                           # Pipeline CLI de entrenamiento, validación y test ciego
├── README.md                          # Documentación técnica de arquitectura y ejecución
├── data/
│   ├── raw_videos/                    # Almacenamiento de videos por clase
│   │   ├── 0/                         # Clase 0: Técnica Correcta
│   │   ├── 1/                         # Clase 1: Valgo Dinámico de Rodilla
│   │   ├── 2/                         # Clase 2: Cifosis Lumbar / Butt Wink
│   │   └── 3/                         # Clase 3: Profundidad Insuficiente
│   └── processed/
│       └── squat_features.csv         # Dataset tabular procesado (12 features + label)
├── models/
│   ├── pose_landmarker_full.task      # Bundle binario BlazePose (MediaPipe Tasks)
│   ├── best_model.pth                 # Checkpoint PyTorch de RegularizedSmallMLP
│   ├── scaler.pkl                     # StandardScaler serializado con Joblib (producción)
│   └── scaler.pt                      # Parámetros del StandardScaler para PyTorch C++ / TorchScript
└── src/
    ├── __init__.py                    # Módulo raíz
    ├── api/
    │   ├── __init__.py
    │   └── main.py                    # Servicio REST FastAPI (predicción numérica y análisis de video)
    ├── kinematics/
    │   ├── __init__.py
    │   └── vector_engine.py           # Álgebra vectorial 3D (clip [-1,1]), 12 features y filtro temporal
    ├── vision/
    │   ├── __init__.py
    │   └── pose_detector.py           # SquatPoseDetector con API moderna MediaPipe Tasks
    ├── models/
    │   ├── __init__.py
    │   └── architecture.py            # Red RegularizedSmallMLP (12,580 parámetros exactos)
    └── data/
        ├── __init__.py
        └── video_dataset_extractor.py # Extracción en lote de videos a CSV
```

---

## 2. Instalación del Entorno

```bash
# 1. Crear entorno virtual
python -m venv .venv

# 2. Activar entorno virtual
# En Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# En Linux / macOS:
# source .venv/bin/activate

# 3. Instalar librerías requeridas
pip install -r requirements.txt
```

---

## 3. Extracción de Características Cinemáticas en Lote

Coloca tus videos en la carpeta de la clase correspondiente (formatos `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`):

| Carpeta | Etiqueta |
|---|---|
| `data/raw_videos/0/` | Técnica correcta |
| `data/raw_videos/1/` | Valgo dinámico de rodilla |
| `data/raw_videos/2/` | Cifosis lumbar / butt wink |
| `data/raw_videos/3/` | Profundidad insuficiente |

Cada video debe mostrar una sentadilla completa y tener la etiqueta de su carpeta. Incluye al menos tres videos distintos por clase (mejor si son muchos más), con distintas personas y condiciones de grabación. Luego extrae las características:

```bash
python src/data/video_dataset_extractor.py --input_dir data/raw_videos --output_file data/processed/squat_features.csv --stride 2
```

*Opciones disponibles:*
- `--input_dir`: Directorio que contiene las subcarpetas `0`, `1`, `2`, `3`.
- `--output_file`: Ruta de destino del archivo CSV.
- `--stride 2`: Analiza un cuadro de cada dos para acelerar la extracción. Usa `1` para procesar todos los cuadros.

---

## 4. Entrenamiento y Evaluación del Modelo (Train.py)

Ejecuta el pipeline de optimización y prueba con Early Stopping:

```bash
python train.py --data_path data/processed/squat_features.csv --epochs 300 --patience 20
```

El entrenamiento exige datos extraídos de las cuatro carpetas; ya no genera un modelo con datos sintéticos si el CSV está vacío. Separa videos completos entre entrenamiento, validación y test para que cuadros del mismo clip no inflen la evaluación.

*Flujo de trabajo:*
1. **Prevención de Data Leakage:** El `StandardScaler` se ajusta solo con el conjunto de entrenamiento (`fit_transform`) y transforma de forma pasiva validación y test (`transform`).
2. **Serialización MLOps:** Guarda `models/scaler.pkl` y `models/scaler.pt` junto a `models/best_model.pth`.
3. **Early Stopping:** Monitorea la pérdida de validación con paciencia de 20 épocas y restaura el mejor estado de pesos.
4. **Test Set Ciego (15%):** Evalúa el clasificador sobre observaciones no vistas, generando la Matriz de Confusión y el Reporte de Clasificación detallado.

---

## 5. Servicio API REST (Inferencia y Análisis de Video)

Inicia el servidor backend de producción local con Uvicorn:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Acceso a la documentación interactiva Swagger UI: `http://localhost:8000/docs`

### Endpoints Disponibles

- **`GET /`**: Health check y estado de los modelos en memoria.
- **`POST /predict`**: Inferencia a partir del vector numérico de 12 variables cinemáticas precalculadas.
- **`POST /predict/video`**: Endpoint para analizar un archivo de video (`.mp4`, `.avi`, `.mov`, etc.). Usa seguimiento de pose en modo VIDEO y por defecto analiza uno de cada dos cuadros (`sample_stride=2`). Envía `sample_stride=1` para más detalle o un valor mayor para priorizar velocidad. No es una transmisión en vivo desde cámara; procesa el clip recibido. Devuelve:
  - `is_correct`: Booleano (`true` si la técnica fue correcta, `false` ante fallos).
  - `overall_posture`: Diagnóstico general de la postura.
  - `detected_flaws`: Lista detallada de fallas detectadas (ej. falta de profundidad, valgo de rodilla, inclinación o curvatura lumbar).
  - `recommendations`: Consejos correctivos biomecánicos basados en estándares NSCA.
  - `critical_frame`: Información del punto de máxima profundidad (cuadro, segundo exacto, ángulos articulares y probabilidades).
  - `biomechanical_metrics`: Resumen de ángulos mínimos, inclinación máxima del tronco, asimetrías y duración.

---

## 6. Especificaciones Biomecánicas del Motor

- **Álgebra Vectorial 3D:** Todos los ángulos articulares se calculan mediante producto escalar tridimensional protegido:
  $$\theta = \arccos\left(\text{clip}\left(\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}, -1.0, 1.0\right)\right) \times \left(\frac{180}{\pi}\right)$$
- **Tolerancia a Oclusiones:** Si un landmark presenta visibilidad $c_i < 0.60$, el motor ejecuta una imputación simétrica contralateral bilateral proyectada sobre el plano sagital medio.
- **Topología de Red:** $12 \to 128 \to 64 \to 32 \to 4$ con BatchNorm1d y Dropout decreciente ($0.35 \to 0.245 \to 0.14$), totalizando exactamente **12,580 parámetros**.
