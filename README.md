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
doméstico ficticio **RT-900**, en cuatro archivos dentro de `manuales/`:

| Archivo | Formato | Contenido |
|---|---|---|
| `manual_router_rt900.pdf` | PDF | Instalación, significado de LEDs, configuración de WiFi, problemas comunes, advertencia de seguridad, controles parentales |
| `Guia_Usuario_Router.pdf` | PDF | Guía de usuario: puertos (WAN/LAN), botón RESET y restablecimiento a valores de fábrica |
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
desde la propia interfaz web (botón **+** junto al cuadro de mensaje, o arrastrando el archivo al chat).

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

`FastEmbedEmbeddings` (LangChain + **fastembed**, motor ONNX) con el modelo
**`sentence-transformers/all-MiniLM-L6-v2`** (~90 MB, vectores de 384
dimensiones), calculados **localmente** dentro del propio servidor. El modelo
se puede cambiar con la variable de entorno `EMBEDDING_MODEL`.

**Justificación:** la primera versión usaba `paraphrase-multilingual-MiniLM-L12-v2`
con `sentence-transformers`/PyTorch (~470 MB de modelo más PyTorch). En el plan
gratuito de Render (512 MB de RAM) el proceso se quedaba sin memoria a mitad
del indexado y Render lo reiniciaba. Se cambió a fastembed, que ejecuta el
modelo con ONNX sin PyTorch, y a un modelo más pequeño que sí cabe en la
memoria disponible. Al calcularse localmente, **ningún texto del corpus sale
del servidor para vectorizarse** y no se depende de otra API key, lo que
refuerza el control del índice. La contrapartida es que `all-MiniLM-L6-v2`
está entrenado principalmente en inglés: con manuales técnicos en español
funciona razonablemente porque los términos técnicos coinciden (WAN, RESET,
LED, WiFi), pero es algo menos preciso que un modelo multilingüe.

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
   Embeddings locales con fastembed (all-MiniLM-L6-v2, ONNX)
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
  # GROQ_API_KEY en el archivo .env

  # Antes (configuración base)
  python eval/evaluar_ragas.py --top-k 6 --modelo openai/gpt-oss-20b --modelo-juez openai/gpt-oss-20b --salida eval/resultados_before.csv
  # Después (parámetro modificado)
  python eval/evaluar_ragas.py --top-k 10 --modelo openai/gpt-oss-20b --modelo-juez openai/gpt-oss-20b --salida eval/resultados_after.csv
  ```

- **Adaptación al plan gratuito de Groq:** Ragas hace muchas llamadas al
  LLM juez y el plan gratuito limita los tokens por minuto y por día. Por
  eso el script: hace una pausa entre preguntas (`--pausa`), califica de a
  una llamada a la vez y reintenta ante errores 429; guarda cada respuesta
  generada en `eval/resultados_*.respuestas.json` y cada calificación en
  una caché en disco (`eval/.cache_ragas/`), de modo que si se agota la
  cuota diaria basta con repetir el mismo comando más tarde y la evaluación
  retoma donde quedó. La evaluación usó `openai/gpt-oss-20b` como generador
  y como juez (la aplicación desplegada usa `openai/gpt-oss-120b`); como
  ambas corridas comparten la misma configuración salvo `top_k`, la
  comparación antes/después es válida.

- **Iteración de mejora (antes/después):** la evaluación se corre dos veces,
  primero con la configuración base (`chunk_size=500, chunk_overlap=50,
  top_k=6`) y luego con el parámetro modificado (`top_k=10`), guardando
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

### Resultados

| Métrica | Antes (`top_k=6`) | Después (`top_k=10`) | Cambio |
|---|---|---|---|
| faithfulness | 0,628 | 0,730 | +0,10 |
| answer_relevancy | 0,427 | 0,363 | −0,06 |
| context_precision | 0,492 | 0,467 | −0,03 |
| context_recall | 0,650 | 0,750 | +0,10 |

- **Métrica más baja:** `answer_relevancy`, antes y después de la mejora.
- **Componente al que se atribuye:** principalmente a la **generación**. La
  respuesta es un JSON con campos (`nivel_urgencia`, `fuente`,
  `requiere_soporte_humano`) que no responden la pregunta en sí, y en las 5
  preguntas fuera del corpus el asistente responde, como debe, que no hay
  información, lo que Ragas califica con 0. No se atribuye al chunking:
  `context_recall` muestra que los fragmentos sí contienen lo necesario.
- **Efecto de la mejora:** subir `top_k` aumentó la cobertura
  (`context_recall` +0,10) y el respaldo de las respuestas (`faithfulness`
  +0,10), con una leve baja de `context_precision` porque con 10 de 14
  fragmentos se recupera el 71 % del corpus y entra más ruido.
  `answer_relevancy` no mejoró, lo que confirma que su causa está en el
  formato de salida y no en la recuperación. En un corpus tan pequeño
  conviene un valor intermedio (p. ej. `k = 8`), y la mejora de
  `answer_relevancy` debe buscarse en el formato de la respuesta.

---

## Interfaz de chat conversacional

- Construida en **Flask**, con HTML/CSS/JS propios (sin frameworks de
  frontend), con un diseño de chat oscuro en los colores de la Konrad
  Lorenz (navy `#0a2f5c` + dorado `#e2ab3d`).
- **Conversación real:** cada sesión del navegador mantiene su propio
  historial (cookie de sesión + diccionario en memoria en el servidor); las
  preguntas de seguimiento se responden usando los turnos previos reales
  (ver "Generación" arriba). En **Ajustes** (ícono de engranaje) está el
  botón **"Nueva conversación"** para reiniciar el historial sin perder el
  índice de manuales; ahí también se puede pegar la API key de Groq, elegir
  el modelo y ajustar `k`.
- Muestra las **fuentes** (documento y página) que respaldan cada respuesta,
  y un panel expandible **"Ver fragmentos y prompt enviado al modelo"** con
  el detalle completo de lo que se le envió al LLM.
- Si la pregunta está fuera del corpus, el propio modelo lo indica en
  `diagnostico` y marca `requiere_soporte_humano: true`, mostrando una
  alerta visual en la tarjeta de respuesta.
- Permite **subir manuales** (PDF/DOCX/TXT/HTML) con el botón **+** o
  arrastrándolos al chat; se indexan en segundo plano con una barra de
  progreso.

### Despliegue

Desplegado en **Render.com** (plan free):

**URL pública:** https://asistente-soporte-tecnico-nq1w.onrender.com
*(puede tardar ~30-50 segundos en responder la primera vez si el servicio
estaba dormido — es una limitación del plan gratuito de Render, no un
error.)*

- `Procfile` → `gunicorn app:app --workers 1 --worker-class gthread --threads 4 --timeout 300`.
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
- **Memoria:** el plan gratuito tiene 512 MB de RAM; por eso los embeddings
  usan fastembed/ONNX con un modelo pequeño (ver "Vectorización").
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
(~90 MB); quedará cacheado localmente para las siguientes ejecuciones.

### Procedimiento de despliegue (Render.com)

1. Subir el repositorio a GitHub (sin `.env`, excluido por `.gitignore`).
2. En Render: **New > Web Service**, conectar el repositorio.
3. Render detecta `render.yaml` (o configurar manualmente: build command
   `pip install -r requirements.txt`, start command
   `gunicorn app:app --workers 1 --worker-class gthread --threads 4 --timeout 300`).
4. En **Environment**, agregar la variable `GROQ_API_KEY` con la API key
   real y, para la búsqueda web del agente (Avance 3), `TAVILY_API_KEY`
   (nunca se incluyen en el repositorio).
5. Desplegar y esperar el build; copiar la URL pública generada.

---

## Avance 3 — Agente con herramientas (tool calling)

El Avance 3 convierte el RAG del Avance 2 en un **agente**. En el Avance 2 el
programa siempre buscaba y pegaba los fragmentos en el prompt (pipeline fijo,
una llamada al LLM por pregunta). Ahora **el LLM decide** si necesita consultar
la base, calcular algo o buscar en la web, y el programa ejecuta la herramienta
y le devuelve el resultado. Se reutiliza todo lo anterior: ingesta, chunking
(500/50), índice Chroma, embeddings fastembed, conjunto de evaluación de Ragas
e interfaz desplegada.

Framework: **LangChain** (`ChatGroq.bind_tools`, tools con `@tool`) y un
**bucle manual** en `agent.py` (no `create_agent`), para controlar explícitamente
el límite de iteraciones, el registro y los límites de cada tool. Modelo:
`openai/gpt-oss-120b` en Groq, que soporta tool calling.

### Arquitectura del agente

```
 Usuario ── pregunta ──► Flask (/api/ask) ──► agent.py (bucle)
                                                  │
        ┌─────────────────────────────────────────┘
        ▼
   LLM Groq  ◄──────────── resultado de la tool ◄───────────┐
   (system prompt + few-shot + historial + pregunta)          │
        │                                                      │
        ├─ sin tool_calls ─────────────► respuesta final (JSON) │
        │                                                      │
        └─ tool_calls ──► el PROGRAMA ejecuta la tool ─────────┘
                           ├─ buscar_en_base     → ChromaDB local (el corpus NO sale)
                           ├─ calcular_garantia  → cálculo local
                           └─ buscar_en_web      → Tavily (sale SOLO la consulta)
```

**Qué sale del sistema y hacia dónde:**

| Destino | Qué se envía | Qué nunca se envía |
|---|---|---|
| Groq (LLM) | La pregunta, el historial reciente y los fragmentos/resultados que devolvieron las tools (máx. 6 fragmentos de 500 caracteres por búsqueda) | El corpus completo |
| Tavily (buscador) | Solo el texto de la consulta (máx. 300 caracteres) | Fragmentos de los manuales, historial, datos del usuario |
| ChromaDB / cálculo | Nada: corren dentro del servidor | — |

### Tools

| Tool | Tipo | Qué hace y cuándo se usa | Argumentos | Límites (en el código) |
|---|---|---|---|---|
| `buscar_en_base` | búsqueda | Recupera fragmentos de los manuales. Siempre que la pregunta trate del producto. | `consulta: str` | `MAX_FRAGMENTOS_POR_LLAMADA = 6`, `MAX_CHARS_FRAGMENTO = 500`, `MAX_CHARS_CONSULTA = 300`. Cada fragmento trae documento y página. |
| `calcular_garantia` | tarea del dominio | Calcula si la garantía (12 meses, según `politica_garantia_rt900.txt`) está vigente y cuándo vence. El LLM extrae la fecha de la pregunta; el cálculo de fechas lo hace el programa. | `fecha_compra: AAAA-MM-DD`, `fecha_consulta` (opcional) | Valida formato y que la compra no sea futura; plazo fijo `MESES_GARANTIA = 12`. |
| `buscar_en_web` | web | Busca información pública o actual en sitios permitidos (ver política). | `consulta: str` | `MAX_RESULTADOS_WEB = 3`, `MAX_CHARS_EXTRACTO_WEB = 300`, lista de dominios fija, timeout 15 s. |

Todos los límites son constantes de `agent_tools.py`: el LLM no puede cambiarlos
aunque se lo pidan en el chat. Cada tool devuelve además un *artefacto*
estructurado (documento, página, URL) que usan la interfaz, el registro y la
evaluación; el LLM no lo ve.

**Probar las tools sin el LLM:** `python probar_tools.py` llama cada tool con
`.invoke()` y muestra la salida exacta que recibiría el modelo.

### Bucle del agente y registro de uso

`AgenteSoporte.responder()` (`agent.py`):

1. Arma los mensajes: system prompt + ejemplos few-shot (que incluyen llamadas a
   tools) + historial + pregunta.
2. Llama al LLM con las tools enlazadas. Si no pide ninguna, esa es la respuesta final
   (así un saludo se responde sin buscar). Si pide una o varias, el programa las
   ejecuta y devuelve cada resultado como `ToolMessage`, y vuelve a llamar al LLM
   (por eso una pregunta con dos temas puede llamar a `buscar_en_base` dos veces).
3. **Límites impuestos por el código:** `MAX_ITERACIONES = 5` vueltas por pregunta
   (si se agotan, se fuerza una respuesta final sin tools) y `MAX_LLAMADAS_TOOLS = 6`
   ejecuciones por pregunta.

**Registro:** por cada pregunta se agrega una línea JSON a `logs/uso_tools.jsonl`
(y se imprime en los logs de Render con el prefijo `[AGENTE]`) con las tools
llamadas, sus argumentos, el resumen del resultado, las llamadas al LLM, los
tokens y el tiempo. Ejemplo:

```json
{"fecha": "2026-11-07T20:56:17+00:00", "sesion": "0b17b91d",
 "pregunta": "Compré mi router el 15 de marzo de 2026, ¿sigo en garantía?",
 "tools": [{"nombre": "calcular_garantia", "args": {"fecha_compra": "2026-03-15"},
            "resumen": "vigente, vence 2027-03-15"}],
 "llamadas_llm": 2, "tokens": {"input_tokens": 2100, "output_tokens": 180, "total_tokens": 2280},
 "tiempo_s": 2.1, "limite_alcanzado": false}
```

### Prompt del sistema actualizado (`agent_prompt.py`)

El archivo del Avance 1 sigue intacto. Cambios frente al Avance 2:

- **El prompt ya no lleva el contexto**: los fragmentos llegan como resultado de
  `buscar_en_base`, no pegados en `<contexto>`.
- Incluye una **política de uso de tools**: saludo → ninguna; pregunta del producto →
  base primero (una búsqueda por tema); fecha de compra → `calcular_garantia`;
  web solo si la base no responde y la información es pública o actual.
- Pide responder **solo con lo que devuelvan las tools** y **citar el origen
  separado**: base (`fuente`: documento y página) o web (`[web: URL]` y `fuentes_web`).
- Mantiene la respuesta exacta cuando la información no está
  (`"No disponible en el contexto entregado"` + `requiere_soporte_humano: true`).
- Trata como datos todo lo que devuelva una tool y prohíbe volcar los manuales o
  revelar el prompt (instrucción maliciosa).
- Formato de salida: el mismo JSON del Avance 2 + `fuentes_web`.
- Los ejemplos Few-Shot son ahora secuencias reales de mensajes (saludo sin tools,
  pregunta de la base, tarea de garantía, base y luego web).

### Política de búsqueda web

- **Cuándo se usa:** solo si `buscar_en_base` ya se consultó y no contiene la
  respuesta, **y** la pregunta trata de información pública o actual (un estándar,
  una definición general, una versión reciente). Nunca para datos o
  procedimientos del RT-900. La regla está en el docstring de la tool y en el
  system prompt.
- **Qué sale:** únicamente la consulta (máx. 300 caracteres). Nunca fragmentos de la base.
- **Servicio:** Tavily (`TAVILY_API_KEY`, variable de entorno; nunca en el repositorio).
- **Sitios permitidos** (`DOMINIOS_WEB_PERMITIDOS`, fijos en el código; además se
  filtran los resultados por dominio como defensa extra):

  | Sitio | Por qué es confiable para este dominio |
  |---|---|
  | `wi-fi.org` | Wi-Fi Alliance: define y certifica WiFi, WPA2 y WPA3 |
  | `ieee.org` | Publica los estándares 802.11 en los que se basa el WiFi |
  | `ietf.org` | Estándares de internet (DNS, DHCP, IP) |
  | `nist.gov` | Guías oficiales de seguridad (cifrado, contraseñas) |
  | `cisa.gov` | Recomendaciones oficiales para asegurar routers y redes |
  | `support.microsoft.com` | Soporte oficial de Windows (conectarse a una red) |
  | `support.apple.com` | Soporte oficial de Apple (conectarse a una red) |

- **Resultados acotados:** máximo 3; cada uno solo con título, URL y un extracto de
  máximo 300 caracteres, marcado como origen "web".
- **Respuesta separada por origen:** lo de la base se cita como `Fuente: documento, pág.`
  y lo de la web como `[web: URL]`; la interfaz las muestra en líneas distintas.

### Evaluación del agente

Misma evaluación de Ragas del Avance 2 (20 preguntas), con las mismas métricas y
el mismo modelo generador y juez que se usaron en el Avance 2 (`openai/gpt-oss-20b`,
por los límites del plan gratuito de Groq), para que la comparación sea válida.
Los `retrieved_contexts` del agente son los fragmentos que devolvieron sus tools.

```bash
pip install -r requirements.txt -r eval/requirements-eval.txt
# En .env: GROQ_API_KEY y TAVILY_API_KEY

# 1) Ragas del agente + tabla Avance 2 vs agente (eval/comparacion_ragas.md)
python eval/evaluar_agente.py --modelo openai/gpt-oss-20b --modelo-juez openai/gpt-oss-20b

# 2) Casos de comportamiento (8 casos, esperado escrito antes en eval/casos_comportamiento.json)
python eval/evaluar_casos.py --modelo openai/gpt-oss-20b

# 3) Costo: llamadas al LLM, tokens y tiempo, pipeline fijo vs agente
python eval/comparar_costo.py --modelo openai/gpt-oss-20b
```

Salidas: `eval/resultados_agente.csv/.png`, `eval/comparacion_ragas.md`,
`eval/resultados_casos.csv/.md`, `eval/costo_comparacion.csv/.md` y los registros
`eval/registro_agente.jsonl` y `eval/registro_casos.jsonl`. Los resultados y su
análisis están en el PDF de entrega.

### Interfaz

La interfaz del Avance 2 usa ahora el agente y mantiene el historial de la
conversación. Cada respuesta incluye el desplegable **"Ver herramientas usadas y
fuentes"**: qué tools llamó el agente y con qué argumentos, llamadas al LLM,
tokens y tiempo, y las fuentes **separadas por origen** (fragmentos de la base con
documento y página; resultados web con título y URL). Si la información no está
ni en la base ni en los sitios permitidos, el asistente lo dice y recomienda
soporte humano.

### Instalación, ejecución y despliegue

```bash
pip install -r requirements.txt
# crea un archivo .env en la raíz con:  GROQ_API_KEY=...  y  TAVILY_API_KEY=...
python app.py          # http://127.0.0.1:5000
```

En Render, además de `GROQ_API_KEY`, se agrega la variable de entorno
**`TAVILY_API_KEY`** (Environment → Add). Sin ella el agente funciona con la base
y el cálculo, y la búsqueda web responde que no está disponible.

**URL pública:** https://asistente-soporte-tecnico-nq1w.onrender.com

---

## Estructura del repositorio

```
├── README.md
├── asistente_soporte_tecnico.py   # Avance 1: System prompt, few-shot, delimitadores (sin modificar)
├── rag_core.py                     # Avance 2: pipeline RAG completo + memoria conversacional
├── agent.py                        # Avance 3: bucle del agente, límites y registro de uso
├── agent_tools.py                  # Avance 3: tools (buscar_en_base, calcular_garantia, buscar_en_web)
├── agent_prompt.py                 # Avance 3: system prompt y few-shot del agente
├── probar_tools.py                 # Avance 3: prueba de las tools sin LLM (.invoke())
├── app.py                          # Interfaz Flask (rutas, sesión, indexación en segundo plano)
├── templates/
│   └── index.html
├── static/
│   ├── css/style.css
│   └── js/chat.js
├── manuales/                       # Corpus de prueba (router RT-900: 2 pdf, txt, html)
├── eval/
│   ├── preguntas_eval.json         # 20 preguntas con ground truth (15 en corpus + 5 fuera)
│   ├── evaluar_ragas.py            # Script de evaluación con Ragas
│   ├── ejecutar_evaluacion_completa.sh  # Corre evaluación antes/después de la mejora
│   ├── requirements-eval.txt       # Dependencias extra solo para evaluar (no para desplegar)
│   ├── resultados_before.csv/.png  # Resultados con top_k = 6
│   ├── resultados_after.csv/.png   # Resultados con top_k = 10
│   ├── evaluar_agente.py           # Avance 3: Ragas del agente y tabla Avance 2 vs agente
│   ├── casos_comportamiento.json   # Avance 3: 8 casos con el comportamiento esperado
│   ├── evaluar_casos.py            # Avance 3: ejecuta y verifica los casos
│   ├── comparar_costo.py           # Avance 3: llamadas, tokens y tiempo (pipeline vs agente)
│   └── resultados_*.respuestas.json  # Respuestas generadas (para retomar sin regenerar)
├── requirements.txt
├── Procfile
├── render.yaml
└── .gitignore
```

## Trabajo futuro

- Ajustar el formato de salida (o evaluar solo `diagnostico` y
  `pasos_solucion`) para mejorar `answer_relevancy`, la métrica más baja.
- Probar un valor intermedio de recuperación (`top_k = 8`) y un modelo de
  embeddings multilingüe si se dispone de más memoria en el servidor.
- Persistir el índice y el historial en un almacenamiento externo para no
  depender del disco efímero de Render.