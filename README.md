# Asistente Experto de Soporte Técnico basado en RAG y Agentes

## Descripción del proyecto

Este proyecto desarrolla un asistente de IA capaz de leer una base de
conocimientos específica —manuales técnicos e instructivos de producto— para
responder preguntas de soporte técnico y guiar al usuario paso a paso hacia
la solución de su problema. El sistema conserva el control del índice
vectorial: los manuales nunca se entregan a servicios externos de
entrenamiento, y al modelo solo se le envían los fragmentos recuperados
necesarios para cada consulta, nunca el corpus completo.

## Enfoque elegido: Analista de Soporte Técnico

Se eligió el enfoque de Analista de Soporte Técnico porque:

- Los manuales de producto tienen una estructura clara (secciones,
  advertencias, pasos numerados) que facilita delimitar el contexto que se
  entrega al modelo.
- Las respuestas esperadas son verificables: un usuario con un problema
  concreto (ej. "el LED parpadea en rojo") tiene una solución específica
  documentada en el manual.
- Permite definir un formato de salida estructurado (JSON) que puede
  integrarse fácilmente a un sistema de tickets o a un chatbot.

## ¿Qué hace el asistente?

1. Recibe una pregunta del usuario relacionada con un producto (ej. un
   router doméstico), dentro de una conversación que puede tener preguntas
   de seguimiento.
2. Recupera del manual técnico indexado los fragmentos más relevantes para
   esa pregunta (retrieval por similitud sobre una base vectorial Chroma).
3. Combina esos fragmentos, la pregunta y el historial reciente de la
   conversación dentro de un prompt estructurado con delimitadores XML.
4. Genera una respuesta estructurada en JSON con: diagnóstico, pasos de
   solución, nivel de urgencia, fuente citada del manual, y si requiere
   escalar a soporte humano.

---

## Avance 1 — Diseño de Prompts

Este avance se enfocó exclusivamente en el **diseño del prompt**, sin aún
conectar el pipeline de recuperación (RAG) completo. El archivo
`asistente_soporte_tecnico.py` contiene este trabajo **intacto, sin
modificar**, y se importa tal cual desde `rag_core.py` para el Avance 2.

### 1. System Prompt

Define el rol del asistente, sus reglas de comportamiento y el formato de
salida obligatorio. Incluye una regla explícita contra *prompt injection*:
cualquier instrucción que aparezca dentro del contexto recuperado del manual
se trata como dato, nunca como orden.

### 2. Few-Shot Prompting

Se incluyen 3 ejemplos fijos que se insertan antes de la pregunta real del
usuario, cubriendo:

- Un caso resuelto directamente desde el manual (urgencia media).
- Un caso de urgencia alta (riesgo de seguridad).
- Un caso donde el contexto **no** contiene la respuesta, para enseñar al
  modelo a admitir la limitación en lugar de inventar información.

### 3. Estrategia de Delimitadores

Se usan **tags XML** (`<contexto>`, `<pregunta>`) en lugar de triple
comillas, porque:

- Evitan ambigüedad cuando el manual contiene comillas o bloques de código
  (comandos, configuraciones).
- Permiten identificar múltiples fragmentos de contexto recuperados
  (`<contexto id="1">`, `<contexto id="2">`, ...) sin que se mezclen.
- Refuerzan la separación entre "dato" (contenido del manual) e
  "instrucción" (reglas del sistema), mitigando ataques de inyección de
  prompt.

### Formato de salida (sin cambios en el Avance 2)

```json
{
  "diagnostico": "string",
  "pasos_solucion": ["string", "..."],
  "nivel_urgencia": "bajo | medio | alto",
  "fuente": "string",
  "requiere_soporte_humano": true
}
```

---

## Avance 2 — Flujo RAG completo, evaluación con Ragas y despliegue conversacional

### Selección de documentos (corpus)

Para este avance se usa un **manual sintético de prueba**: el router
doméstico ficticio **RT-900**, en tres archivos dentro de `manuales/`:

| Archivo | Formato | Contenido |
|---|---|---|
| `manual_router_rt900.pdf` | PDF | Instalación, significado de LEDs, configuración de WiFi, problemas comunes, advertencia de seguridad, controles parentales |
| `politica_garantia_rt900.txt` | TXT | Política de garantía (12 meses, exclusiones, procedimiento de reclamo) |
| `guia_rapida_rt900.html` | HTML | Restablecer contraseña de WiFi, cambiar canal WiFi |

**Justificación:** se eligió un corpus sintético y controlado (en vez de un
manual real descargado de internet) porque permite: (1) cubrir
deliberadamente los tres formatos de ingesta exigidos por la rúbrica
(PDF/TXT/HTML) con contenido 100% verificable, (2) construir un
conjunto de evaluación con *ground truth* exacto porque el contenido se
escribió a propósito, y (3) incluir preguntas "fuera de corpus" cuya
ausencia de respuesta se puede confirmar con certeza (requisito para medir
control de alucinaciones). El pipeline es agnóstico al corpus: cambiar a
manuales reales solo requiere subirlos desde la interfaz web o colocarlos en
`manuales/`.

### Ingesta (PDF, DOCX, TXT, HTML)

`rag_core.py` → `_cargar_documento()` despacha el loader de LangChain según
la extensión del archivo:

| Extensión | Loader |
|---|---|
| `.pdf` | `PyPDFLoader` |
| `.docx` | `Docx2txtLoader` |
| `.txt` | `TextLoader` |
| `.html` / `.htm` | `BSHTMLLoader` |

Esto permite subir manuales en cualquiera de los cuatro formatos exigidos
desde la propia interfaz web (botón "Subir manuales").

### Chunking

`RecursiveCharacterTextSplitter` con:

- `chunk_size = 500` caracteres
- `chunk_overlap = 50` caracteres
- Separadores en orden: `\n\n`, `\n`, `.`, ` `

**Justificación:** los manuales técnicos tienen párrafos cortos y
autocontenidos (un LED, una advertencia, un paso), así que 500 caracteres
suele capturar una idea completa sin mezclar dos secciones distintas. El
solapamiento de 50 caracteres evita cortar una oración justo en el límite
entre dos fragmentos, sin inflar demasiado el índice. Este parámetro se
revisita explícitamente en la iteración de mejora (ver más abajo).

### Vectorización (embeddings)

`HuggingFaceEmbeddings` con el modelo
**`paraphrase-multilingual-MiniLM-L12-v2`**, ejecutado localmente en CPU,
con normalización de vectores (coseno).

**Justificación:** es un modelo multilingüe (clave porque los manuales y las
preguntas están en español), liviano (~120 MB, corre en CPU sin GPU), y de
uso local — ningún fragmento del manual se envía a un servicio externo de
embeddings, cumpliendo el requisito de control de la base de conocimiento.

### Base de datos vectorial

**ChromaDB**, persistente en disco (`chroma/`), con `hnsw:space: "cosine"`.
Cada fragmento conserva como metadato su archivo de origen (`source`) y,
para PDFs, el número de página — lo que permite citar la fuente exacta de
cada respuesta (campo `"fuente"` del JSON de salida y el panel de
depuración "Ver fragmentos" en la interfaz).

### Recuperación

Búsqueda por similitud de coseno, `top_k` configurable desde la interfaz
(por defecto `k = 6`). La búsqueda usa solo la pregunta del turno actual
(no el historial completo), para no diluir la similitud semántica con texto
de turnos anteriores que puede tratar un tema distinto.

### Generación

Se reutiliza **sin modificar** `construir_mensajes_para_llm()` del Avance 1
(System Prompt + 3 ejemplos Few-Shot + delimitadores XML). El Avance 2 añade
una capa de **memoria conversacional real**: los turnos previos de la misma
conversación (pregunta real + respuesta cruda del asistente) se insertan
entre los ejemplos few-shot y el turno real final, de modo que preguntas de
seguimiento como *"¿y si ya lo reinicié y sigue igual?"* se puedan responder
usando el contexto de lo ya hablado (hasta 6 turnos previos,
`MAX_TURNOS_HISTORIAL` en `rag_core.py`).

El LLM generador es **Groq** (`ChatGroq`, modelo `openai/gpt-oss-120b` por
defecto) en vez de un modelo local vía Ollama como se planteaba en el
Avance 1: Groq ofrece inferencia gratuita y mucho más rápida que correr un
LLM local en el hardware limitado del plan gratuito de Render, sin perder
control sobre qué se envía (solo los fragmentos recuperados y la pregunta,
nunca el corpus completo ni se usan los datos para entrenar).

### Diagrama del flujo

```
 Manuales (PDF/DOCX/TXT/HTML)
          │
          ▼
   Ingesta (loader según extensión)
          │
          ▼
   Chunking (500 / 50, RecursiveCharacterTextSplitter)
          │
          ▼
   Embeddings locales (MiniLM multilingüe, CPU)
          │
          ▼
   ChromaDB (persistente, metadatos: archivo + página)
          │
   ┌──────┴───────┐
   │  pregunta +    │
   │  historial     │   (hasta 6 turnos previos)
   └──────┬───────┘
          ▼
   Retrieval (similitud coseno, top_k)
          │
          ▼
   construir_mensajes_para_llm()  (Avance 1: System + Few-Shot + XML)
          +  turnos previos insertados antes del turno real
          │
          ▼
   LLM Groq (openai/gpt-oss-120b)
          │
          ▼
   Parseo robusto de JSON  →  {diagnostico, pasos_solucion,
                               nivel_urgencia, fuente,
                               requiere_soporte_humano}
          │
          ▼
   Interfaz de chat (Flask) — muestra respuesta + fuentes +
   panel de depuración con fragmentos y prompt enviado
```

---

## Evaluación del RAG con Ragas

- **Conjunto de evaluación:** `eval/preguntas_eval.json`, con **20
  preguntas** (≥15 exigidas): 15 dentro del corpus (con *ground truth* exacto
  tomado del manual) y 5 explícitamente fuera del corpus, para medir control
  de alucinaciones.
- **Script de evaluación:** `eval/evaluar_ragas.py` — indexa los manuales,
  corre el RAG sobre las 20 preguntas, y calcula con Ragas las 4 métricas
  exigidas: `faithfulness`, `answer_relevancy`, `context_precision`
  (`LLMContextPrecisionWithReference`) y `context_recall`
  (`LLMContextRecall`). Guarda un CSV detallado y un gráfico de barras con
  los promedios.

  ```bash
  pip install -r requirements.txt -r eval/requirements-eval.txt
  export GROQ_API_KEY="tu-api-key"
  python eval/evaluar_ragas.py
  ```

- **Iteración de mejora (antes/después):** `eval/ejecutar_evaluacion_completa.sh`
  corre automáticamente la evaluación dos veces: primero con la
  configuración base (`chunk_size=500, chunk_overlap=50, top_k=6`) y luego
  con el parámetro modificado (`top_k=10`), guardando
  `eval/resultados_before.csv/.png` y `eval/resultados_after.csv/.png`.

  **Parámetro modificado:** `top_k` de 6 → 10 fragmentos recuperados.
  **Justificación de la elección:** algunas preguntas de evaluación
  requieren combinar información de más de una sección del manual (por
  ejemplo, una pregunta sobre seguridad que también roza la política de
  garantía); con `top_k=6` el retriever puede quedarse corto y no traer uno
  de los fragmentos necesarios, lo que baja específicamente
  `context_recall`. Subir a `top_k=10` da más margen sin diluir demasiado el
  contexto que recibe el modelo (sigue siendo pequeño frente al límite de
  tokens de Groq).

  *(Los valores reales de las 4 métricas, antes y después, y el análisis de
  cuál quedó más baja y a qué componente del pipeline se atribuye, se
  completan en el PDF de evidencias tras ejecutar el script con una
  `GROQ_API_KEY` real — ver sección siguiente.)*

---

## Interfaz de chat conversacional

- Construida en **Flask**, con HTML/CSS/JS propios (sin frameworks de
  frontend), con un tema visual "mesa de ayuda técnica" en los colores de la
  Konrad Lorenz (navy `#0a2f5c` + dorado `#e2ab3d`).
- **Conversación real:** cada sesión del navegador mantiene su propio
  historial (cookie de sesión + diccionario en memoria en el servidor); las
  preguntas de seguimiento se responden usando los turnos previos reales
  (ver "Generación" arriba). Botón **"+ Nuevo chat"** para reiniciar la
  conversación sin perder el índice de manuales.
- Muestra las **fuentes** (documento y página) que respaldan cada respuesta,
  y un panel expandible **"Ver fragmentos y prompt enviado al modelo"** con
  el detalle completo de lo que se le envió al LLM.
- Si la pregunta está fuera del corpus, el propio modelo lo indica en
  `diagnostico` y marca `requiere_soporte_humano: true`, mostrando una
  alerta visual en la tarjeta de respuesta.
- Permite **subir manuales** (PDF/DOCX/TXT/HTML) desde la barra lateral, que
  se reindexan automáticamente en segundo plano.

### Despliegue

Desplegado en **Render.com** (plan free):

**URL pública:** `https://asistente-soporte-tecnico-2ux8.onrender.com`
*(puede tardar ~30-50 segundos en responder la primera vez si el servicio
estaba dormido — es una limitación del plan gratuito de Render, no un
error.)*

- `Procfile` → `gunicorn app:app --workers 1 --threads 8 --timeout 120`.
  Se usa un solo worker porque el estado (índice de Chroma en memoria,
  historial de conversación por sesión) vive en el proceso, no en un
  datastore compartido; con más de un worker, cada uno tendría su propia
  copia del índice y las sesiones se repartirían de forma inconsistente
  entre procesos.
- `render.yaml` define el blueprint (build command, start command, variable
  `GROQ_API_KEY` marcada como secreta para configurarse manualmente en el
  dashboard de Render, nunca en el repositorio).
- `bootstrap()` en `app.py` se ejecuta al importar el módulo (no solo dentro
  de `if __name__ == "__main__":`), para que también corra bajo gunicorn.
- **Disco efímero:** el plan gratuito de Render borra `chroma/` en cada
  reinicio o redeploy. Por eso los manuales de prueba (`manuales/*`) sí se
  versionan en el repositorio (no están en `.gitignore`): así `bootstrap()`
  siempre los encuentra y puede reindexar desde cero en cada arranque.

### Instalación y ejecución local

```bash
git clone <url-del-repositorio>
cd <carpeta-del-proyecto>
python -m venv .venv
# Windows:
.venv\Scripts\Activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

# Crear un archivo .env en la raíz con:
# GROQ_API_KEY=tu-api-key-de-groq
# FLASK_SECRET_KEY=cualquier-cadena-secreta

python app.py
# abrir http://localhost:5000
```

La primera vez que se indexa, se descarga el modelo de embeddings
(~120 MB); quedará cacheado localmente para las siguientes ejecuciones.

### Procedimiento de despliegue (Render.com)

1. Subir el repositorio a GitHub (sin `.env`, excluido por `.gitignore`).
2. En Render: **New > Web Service**, conectar el repositorio.
3. Render detecta `render.yaml` (o configurar manualmente: build command
   `pip install -r requirements.txt`, start command
   `gunicorn app:app --workers 1 --threads 8 --timeout 120`).
4. En **Environment**, agregar la variable `GROQ_API_KEY` con la API key
   real (nunca se incluye en el repositorio).
5. Desplegar y esperar el build; copiar la URL pública generada.

---

## Estructura del repositorio

```
├── README.md
├── asistente_soporte_tecnico.py   # Avance 1: System prompt, few-shot, delimitadores (sin modificar)
├── rag_core.py                     # Avance 2: pipeline RAG completo + memoria conversacional
├── app.py                          # Interfaz Flask (rutas, sesión, indexación en segundo plano)
├── templates/
│   └── index.html
├── static/
│   ├── css/style.css
│   └── js/chat.js
├── manuales/                       # Corpus de prueba (router RT-900: pdf, txt, html)
├── eval/
│   ├── preguntas_eval.json         # 20 preguntas con ground truth (15 en corpus + 5 fuera)
│   ├── evaluar_ragas.py            # Script de evaluación con Ragas
│   ├── ejecutar_evaluacion_completa.sh  # Corre evaluación antes/después de la mejora
│   ├── requirements-eval.txt       # Dependencias extra solo para evaluar (no para desplegar)
│   ├── resultados_before.csv/.png  # (se genera al ejecutar el script)
│   └── resultados_after.csv/.png   # (se genera al ejecutar el script)
├── requirements.txt
├── Procfile
├── render.yaml
└── .gitignore
```

## Próximos pasos

- Ejecutar `eval/ejecutar_evaluacion_completa.sh` con una `GROQ_API_KEY`
  real para obtener los valores numéricos antes/después y completarlos en
  el PDF de evidencias.
- Tomar las capturas de pantalla de la interfaz desplegada: una conversación
  con pregunta de seguimiento, y un caso fuera del corpus.
- Ensamblar el PDF final de evidencias (diagrama, capturas, tabla de
  métricas Ragas con el análisis antes/después, URL pública y enlace al
  repositorio).
