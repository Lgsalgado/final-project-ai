# Planificación del Proyecto

**Sistema Inteligente de Detección y Corrección de Postura en Ejercicios de Fuerza**  
**Maestría en IA (MIAR0545) — Universidad de Especialidades Espíritu Santo (UEES)**  
**Autores:** Daniel Fernando Salgado Santamaría (Scrum Master/Product Owner) & Luis Gabriel Salgado Santamaría (Tech Lead/Arquitecto IA)

---

## 1. Objetivos SMART del Proyecto

| Criterio | Descripción |
|:---|:---|
| **S — Específico** | Desarrollar un sistema de visión por computador que detecte y clasifique 4 categorías posturales en ejercicios de fuerza (técnica correcta, valgo de rodilla, cifosis lumbar, profundidad insuficiente) mediante MediaPipe BlazePose + red neuronal PyTorch. |
| **M — Medible** | Alcanzar Test Accuracy ≥ 85% en el conjunto ciego con Generalization Gap < 0.05, latencia < 26 ms en CPU estándar. |
| **A — Alcanzable** | El proyecto se implementa con hardware estándar (CPU, cámara web 720p), software open-source (MediaPipe, PyTorch, FastAPI) y datos sintéticos biomecánicos calibrados. |
| **R — Relevante** | Las lesiones musculo-esqueléticas por técnica incorrecta representan el 70% de las lesiones en entrenamiento de fuerza; el sistema democratiza la supervisión biomecánica experta. |
| **T — Temporal** | Entregable funcional en 6 semanas académicas (Semanas 1–6 de MIAR0545), con hitos semanales definidos. |

### Objetivos Específicos Medibles

1. **OE1:** Implementar extracción de 33 landmarks 3D con MediaPipe BlazePose con latencia < 18 ms en CPU (Semana 2).
2. **OE2:** Construir motor cinemático vectorial NSCA que calcule 12 features biomecánicas con precisión angular < 2° (Semana 2).
3. **OE3:** Entrenar y validar `RegularizedSmallMLP` con Test Accuracy ≥ 85% y Gap Loss < 0.05 (Semana 3).
4. **OE4:** Exponer el modelo como API REST (FastAPI) con endpoint `/predict` con latencia < 100 ms (Semana 4).
5. **OE5:** Documentar completamente el sistema con análisis de impacto social y ético (Semana 4).
6. **OE6:** Integrar pipeline completo e interfaz de usuario funcional (Semanas 5–6).

---

## 2. Análisis de Stakeholders

| Stakeholder | Rol | Beneficios | Riesgos | Poder/Voz | Consideraciones en Diseño |
|:---|:---|:---|:---|:---:|:---|
| **Practicantes de fitness / Atletas** | Usuario final principal | Retroalimentación biomecánica instantánea; prevención de lesiones | Dependencia tecnológica; falsos negativos | Alto | Interfaz intuitiva; feedback en lenguaje natural |
| **Entrenadores personales** | Usuario técnico secundario | Herramienta objetiva de supervisión; escalabilidad | Desplazamiento profesional percibido | Alto | Sistema explicable; complementa (no sustituye) al entrenador |
| **Fisioterapeutas / Médicos del deporte** | Validador clínico | Datos cuantitativos para seguimiento; prevención primaria | Responsabilidad diagnóstica | Medio-Alto | Métricas auditables alineadas a estándares NSCA |
| **Propietarios de gimnasios** | Adoptante institucional | Diferenciación competitiva; reducción de lesiones en instalaciones | Costos de implementación; resistencia del personal | Medio | Edge AI sin hardware adicional; privacidad garantizada |
| **Institución académica (UEES)** | Supervisora y evaluadora | Producción académica; innovación tecnológica | Rigor metodológico; reproducibilidad | Alto | Documentación exhaustiva; código abierto |

---

## 3. KPIs del Proyecto

| KPI | Métrica | Umbral Mínimo | Meta Óptima | Estado Final |
|:---|:---|:---:|:---:|:---:|
| Exactitud en Test Ciego | Test Accuracy | ≥ 80% | ≥ 90% | **87.78%** ✓ |
| Generalización | Generalization Gap (Loss) | < 0.10 | < 0.01 | **+0.0031** ✓ |
| Latencia de Inferencia | ms por cuadro (CPU) | < 50 ms | < 26 ms | **< 26 ms** ✓ |
| Eficiencia del Modelo | Parámetros entrenables | < 100,000 | < 15,000 | **12,580** ✓ |
| Cobertura de Clases | Recall mínimo por clase | > 70% | > 85% | **>85%** ✓ |
| Privacidad | Datos de video retenidos | 0 frames | 0 frames | **0 frames** ✓ |

---

## 4. Cronograma del Proyecto (6 Semanas)

| Semana | Entregable Principal | Tareas Clave | Estado |
|:---:|:---|:---|:---:|
| **1** | Presentación del Proyecto | Definición del problema, EDA preliminar, análisis de stakeholders, objetivos SMART, presupuesto | Completado |
| **2** | Ficha de Decisión Técnica y Arquitectura | Pipeline híbrido BlazePose + Motor NSCA, comparativa YOLOv8/ST-GCN, implementación `src/` | Completado |
| **3** | Análisis de Overfitting y Regularización | OverfittingMLP base, Early Stopping, RegularizedSmallMLP, comparativa cuantitativa | Completado |
| **4** | Workshop Impacto Social y Responsabilidad en IA | Análisis ético, stakeholders, riesgos de sesgo, compliance LOPDP Ecuador | Completado |
| **5** | Integración y Pruebas | Integración pipeline completo, pruebas de regresión, optimización de latencia | En proceso |
| **6** | Entrega Final | Repositorio completo, documentación, interfaz de usuario, presentación final | Pendiente |

---

## 5. Presupuesto Estimado del Proyecto

| Recurso | Tipo | Costo Estimado | Justificación |
|:---|:---|:---:|:---|
| Hardware (PC desarrollo) | Existente | $0 | CPU Intel/AMD estándar, sin GPU requerida |
| Cámara web 720p | Existente | $0 | Periférico estándar de trabajo |
| MediaPipe BlazePose | Open-source | $0 | Licencia Apache 2.0 |
| PyTorch | Open-source | $0 | Licencia BSD |
| FastAPI + Uvicorn | Open-source | $0 | Licencia MIT |
| Almacenamiento en nube | Educativo | $0 | Google Drive (cuenta UEES) |
| **Costo Total** | — | **$0** | Proyecto de investigación aplicada sin costo adicional |

---

## 6. Hitos y Criterios de Aceptación

| # | Hito | Criterio de Aceptación | Fecha |
|:---:|:---|:---|:---:|
| H1 | Dataset biomecánico generado | 1,200 muestras, 4 clases balanceadas, split 70/15/15 | Semana 1 |
| H2 | Pipeline de extracción de pose | BlazePose extrae 33 landmarks a < 18 ms por cuadro | Semana 2 |
| H3 | Motor cinemático NSCA | 12 features calculadas con error angular < 2° | Semana 2 |
| H4 | Modelo base entrenado | OverfittingMLP — overfitting documentado (Gap > 2.0) | Semana 3 |
| H5 | Modelo optimizado validado | RegularizedSmallMLP — Test Acc > 85%, Gap < 0.05 | Semana 3 |
| H6 | API REST funcional | Endpoints /predict y /predict/video operativos < 100 ms | Semana 4 |
| H7 | Repositorio completo | Estructura, documentación, tests, notebooks ejecutables | Semana 6 |
