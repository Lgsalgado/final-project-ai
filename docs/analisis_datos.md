# Análisis Exploratorio de Datos (EDA)

**Sistema Inteligente de Detección y Corrección de Postura en Ejercicios de Fuerza**  
**Maestría en IA (MIAR0545) — Universidad de Especialidades Espíritu Santo (UEES)**

---

## 1. Descripción del Dataset Biomecánico

El dataset sintético contiene **1,200 observaciones** de vectores cinemáticos extraídos (o simulados siguiendo distribuciones calibradas) a partir del motor vectorial determinista basado en normas NSCA.

| Propiedad | Valor |
|:---|:---|
| Observaciones totales | 1,200 |
| Features por muestra | 12 |
| Clases (etiquetas) | 4 |
| Distribución por clase | Balanceada (300 muestras / clase) |
| Split Train/Val/Test | 70% / 15% / 15% (840 / 180 / 180) |
| Ruido de etiqueta | 6% estocástico (σ = 8–10°) |
| Generación | Sintética calibrada con MediaPipe BlazePose como referencia |

---

## 2. Variables Cinemáticas (12 Features)

| # | Feature | Unidad | Descripción Biomecánica |
|:---:|:---|:---:|:---|
| 1 | `knee_angle_left` | grados (°) | Ángulo de flexión de la rodilla izquierda (cadera-rodilla-tobillo). Rango normal activo: 70°–170°. |
| 2 | `knee_angle_right` | grados (°) | Ángulo de flexión de la rodilla derecha. Simétrico al izquierdo; asimetría > 15° indica riesgo de varo/valgo. |
| 3 | `hip_angle_left` | grados (°) | Ángulo de flexión de la cadera izquierda (hombro-cadera-rodilla). Indicador de profundidad y apertura pélvica. |
| 4 | `hip_angle_right` | grados (°) | Ángulo de flexión de la cadera derecha. Diferencias > 10° indican asimetría pélvica compensatoria. |
| 5 | `ankle_angle_left` | grados (°) | Ángulo de dorsiflexión del tobillo izquierdo. Limitaciones < 20° condicionan valgo compensatorio. |
| 6 | `ankle_angle_right` | grados (°) | Ángulo de dorsiflexión del tobillo derecho. |
| 7 | `knee_asymmetry_mm` | mm (normalizado) | Asimetría transversal de rodillas normalizada a la distancia inter-trocantérica. Valores > 25 mm indican colapso. |
| 8 | `squat_depth_deg` | grados (°) | Profundidad media de flexión bilateral de rodilla. NSCA establece ≤ 90° para sentadilla paralela válida. |
| 9 | `trunk_lean_deg` | grados (°) | Inclinación anterior del tronco respecto a la vertical. Valores > 45° asociados a cifosis lumbar compensatoria. |
| 10 | `angular_velocity` | °/s | Velocidad angular media de flexión de rodillas entre cuadros consecutivos a 30 FPS. |
| 11 | `valgus_coeff` | adimensional | Índice de colapso medial de rodilla: (distancia_tobillos - distancia_rodillas) / d_caderas. Valores > 0.15 = valgo clínico. |
| 12 | `hip_shoulder_ratio` | adimensional | Ratio d_caderas / d_hombros. Descriptor antropométrico de proporciones esqueléticas. |

---

## 3. Descripción de las 4 Clases Posturales

| ID | Clase | Descripción Biomecánica | Features Discriminantes Principales |
|:---:|:---|:---|:---|
| 0 | **Técnica Correcta** | Ejercicio ejecutado dentro de los umbrales NSCA: rodillas > 90°, valgo < 0.15, tronco < 45°, simetría < 25 mm | `squat_depth_deg` ≤ 95°, `valgus_coeff` < 0.15, `trunk_lean_deg` < 40° |
| 1 | **Valgo Dinámico de Rodilla** | Colapso medial de rodillas durante el descenso; riesgo de lesión LCA, meniscos y ligamentos laterales | `valgus_coeff` > 0.15, `knee_asymmetry_mm` > 20 mm |
| 2 | **Cifosis Lumbar / Butt Wink** | Retroversión pélvica y pérdida de curvatura lordótica lumbar en el fondo del ejercicio; estrés discal L4-S1 | `trunk_lean_deg` > 45°, `squat_depth_deg` extremo |
| 3 | **Profundidad Insuficiente** | Flexión de rodilla insuficiente (no alcanza el paralelo); activación subóptima de glúteo e isquiotibiales | `squat_depth_deg` > 110°, `knee_angle_left` + `knee_angle_right` > 220° |

---

## 4. Proceso de Generación del Dataset

El dataset se generó mediante simulación estocástica calibrada, siguiendo las distribuciones angulares de los estándares biomecánicos NSCA:

```python
def generate_squat_dataset(n_samples=1200, noise_std=9.0, label_noise=0.06, seed=42):
    """
    Genera dataset sintético biomecánico de ejercicios de fuerza.
    - n_samples: 1200 observaciones (300 por clase, balanceado)
    - noise_std: desviacion estandar gaussiana de 9 grados en angulos
    - label_noise: 6% de etiquetas aleatorias para simular solapamiento real
    - seed: reproducibilidad garantizada
    """
```

Parámetros de simulación por clase:

| Clase | knee_angle (media°) | hip_angle (media°) | valgus_coeff (media) | trunk_lean (media°) |
|:---|:---:|:---:|:---:|:---:|
| Técnica Correcta | 88° ± 9° | 82° ± 9° | 0.08 ± 0.03 | 25° ± 9° |
| Valgo Dinámico | 92° ± 9° | 78° ± 9° | 0.28 ± 0.05 | 30° ± 9° |
| Cifosis Lumbar | 85° ± 9° | 88° ± 9° | 0.10 ± 0.03 | 52° ± 9° |
| Prof. Insuficiente | 120° ± 9° | 110° ± 9° | 0.07 ± 0.03 | 20° ± 9° |

---

## 5. Preprocesamiento y Normalización

1. **StandardScaler:** Normalización Z-score aplicada sobre el conjunto de entrenamiento (media=0, σ=1). El scaler ajustado se serializa como `models/scaler.pkl` y `models/scaler.pt`.
2. **Partición tripartita estratificada (70/15/15):**
   - Train (70%, 840 muestras): Optimización de pesos
   - Validación (15%, 180 muestras): Early Stopping
   - Test Ciego (15%, 180 muestras): Evaluación final no sesgada
3. **Stratify:** La partición mantiene la distribución de clases (25% por clase) en cada split.

---

## 6. Análisis Estadístico Descriptivo (Resumen)

| Feature | Min | Max | Media | Std | Correlación con Clase |
|:---|:---:|:---:|:---:|:---:|:---:|
| knee_angle_left | 68° | 148° | 96.2° | 17.8° | Alta (r=0.71) |
| valgus_coeff | 0.001 | 0.41 | 0.13 | 0.08 | Alta (r=0.68) |
| trunk_lean_deg | 12° | 68° | 31.8° | 12.3° | Alta (r=0.64) |
| squat_depth_deg | 68° | 148° | 96.2° | 17.8° | Alta (r=0.71) |
| knee_asymmetry_mm | 0.1 | 42.3 | 11.2 | 8.9 | Media (r=0.45) |
| hip_shoulder_ratio | 0.62 | 1.38 | 0.94 | 0.14 | Baja (r=0.12) |

---

## Referencias

[1] National Strength and Conditioning Association (NSCA), *Essentials of Strength Training and Conditioning*, 4th ed. Human Kinetics, 2016.  
[2] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *JMLR*, vol. 12, pp. 2825–2830, 2011.  
