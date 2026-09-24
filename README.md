# Asistente Experto de Soporte Técnico basado en RAG y Agentes

## Descripción del proyecto

Este proyecto desarrolla un asistente de IA capaz de leer una base de
conocimientos específica —en este caso, manuales técnicos e instructivos
de producto— para responder preguntas de soporte técnico y guiar al
usuario paso a paso hacia la solución de su problema.

## Enfoque elegido: Analista de Soporte Técnico

Se eligió el enfoque de Analista de Soporte Técnico porque:

- Los manuales de producto tienen una estructura clara (secciones,
  advertencias, pasos numerados) que facilita delimitar el contexto que
  se entrega al modelo.
- Las respuestas esperadas son verificables: un usuario con un problema
  concreto (ej. "el LED parpadea en rojo") tiene una solución específica
  documentada en el manual.
- Permite definir un formato de salida estructurado (JSON) que puede
  integrarse fácilmente a un sistema de tickets o a un chatbot.

## ¿Qué hace el asistente?

1. Recibe una pregunta del usuario relacionada con un producto (ej. un
   router doméstico).
2. Recupera del manual técnico (base de conocimiento local, indexada en
   ChromaDB) los fragmentos más relevantes para esa pregunta.
3. Combina esos fragmentos con la pregunta dentro del prompt estructurado
   con delimitadores XML diseñado en el Avance 1.
4. Genera una respuesta estructurada en JSON con: diagnóstico, pasos de
   solución, nivel de urgencia, fuente citada del manual, y si requiere
   escalar a soporte humano.

---

## Avance 1 — Diseño de Prompts

Este avance se enfocó exclusivamente en el **diseño del prompt**, sin aún
conectar el pipeline de recuperación (RAG) completo. Se trabajaron tres
componentes, implementados en `asistente_soporte_tecnico.py` (archivo que
se mantiene intacto en el Avance 2: se importa tal cual, sin modificarlo,
para que quede como evidencia entregada).

### 1. System Prompt

Define el rol del asistente, sus reglas de comportamiento y el formato de
salida obligatorio. Incluye una regla explícita contra *prompt injection*:
cualquier instrucción que aparezca dentro del contexto recuperado del
manual se trata como dato, nunca como orden.

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

- Evitan ambigüedad cuando el manual contiene comillas o bloques de
  código (comandos, configuraciones).
- Permiten identificar múltiples fragmentos de contexto recuperados
  (`<contexto id="1">`, `<contexto id="2">`, ...) sin que se mezclen.
- Refuerzan la separación entre "dato" (contenido del manual) e
  "instrucción" (reglas del sistema), mitigando ataques de inyección de
  prompt.

---

## Avance 2 — Pipeline RAG, interfaz de chat y despliegue

### Pipeline de recuperación (RAG)

```
Manuales (PDF) → fragmentos → embeddings locales → ChromaDB → búsqueda
              → mensajes (system + few-shot + <contexto>/<pregunta>)
              → LLM (Groq) → respuesta JSON estructurada
```

- **Carga y fragmentación:** cada manual se carga con `PyPDFLoader` y se
  divide en fragmentos de ~500 caracteres con 50 de solapamiento
  (`RecursiveCharacterTextSplitter`), conservando el número de página de
  origen.
- **Embeddings locales:** `sentence-transformers`
  (`paraphrase-multilingual-MiniLM-L12-v2`), corren en la propia máquina,
  sin enviar el contenido del manual a un servicio externo solo para
  indexarlo.
- **Base vectorial:** ChromaDB, persistida en disco (`chroma/`), para no
  tener que reprocesar los manuales en cada reinicio.
- **Construcción del prompt:** los fragmentos recuperados (top-k, por
  defecto k=6) se pasan a `construir_mensajes_para_llm` del Avance 1 sin
  modificar esa función, así el diseño de prompts queda intacto dentro
  del pipeline real.
- **Generación:** se usa la API de **Groq** (modelo `openai/gpt-oss-120b`)
  en lugar de un modelo local (Ollama), decisión tomada para simplificar
  el despliegue en un servicio web gratuito, donde correr un LLM local de
  varios GB no es viable con los recursos disponibles.
- **Parseo robusto:** la respuesta del modelo se interpreta como JSON
  tolerando que venga envuelta en \`\`\`json o con texto alrededor; si no
  se puede interpretar, la interfaz lo indica en vez de fallar en
  silencio.

### Interfaz de chat

Aplicación web (Flask) con estilo de mesa de ayuda técnica:

- Permite adjuntar manuales PDF desde el chat y muestra el progreso de
  indexado en tiempo real.
- Presenta cada respuesta ya interpretada: diagnóstico, pasos numerados,
  una etiqueta de nivel de urgencia (bajo/medio/alto) y la fuente citada
  por el modelo.
- Si `requiere_soporte_humano` es `true`, se destaca un aviso para
  escalar el caso.
- Incluye un panel "Ver fragmentos y mensajes enviados al modelo" con los
  fragmentos recuperados y los mensajes exactos (system + few-shot +
  turno real) que se enviaron al LLM, para trazabilidad y depuración.

### Estructura del repositorio

```
├── README.md
├── app.py                          # Servidor Flask (rutas, subida de manuales, estado)
├── rag_core.py                     # Pipeline RAG (carga, embeddings, Chroma, LLM)
├── asistente_soporte_tecnico.py    # Avance 1: system prompt, few-shot (sin modificar)
├── requirements.txt
├── Procfile                        # Comando de arranque para Render (gunicorn)
├── render.yaml                     # Blueprint de despliegue en Render
├── manuales/                       # PDFs de manuales técnicos indexados
├── templates/
│   └── index.html
└── static/
    ├── css/style.css
    └── js/chat.js
```

### Formato de salida

Toda respuesta del asistente sigue este esquema JSON fijo:

```json
{
  "diagnostico": "string",
  "pasos_solucion": ["string", "..."],
  "nivel_urgencia": "bajo | medio | alto",
  "fuente": "string",
  "requiere_soporte_humano": true
}
```

### Cómo ejecutarlo en local

```bash
python -m venv env
.\env\Scripts\Activate      # Windows PowerShell
pip install -r requirements.txt
```

Crear un archivo `.env` en la raíz del proyecto con:

```
GROQ_API_KEY=gsk_tu_key_aqui
```

Y ejecutar:

```bash
python app.py
```

Luego abrir `http://127.0.0.1:5000` en el navegador.

### Despliegue

El servicio está desplegado en **Render** (plan gratuito), usando
`gunicorn` como servidor de producción:

- URL pública: **https://asistente-soporte-tecnico-2ux8.onrender.com**
- El repositorio incluye `render.yaml`, así que Render crea el servicio
  automáticamente al conectarlo (New → Blueprint).
- La variable `GROQ_API_KEY` se configura desde el panel de Render
  (Environment), nunca se sube al repositorio.
- Se corre con un solo worker (`--workers 1`) porque el estado del índice
  vive en memoria del proceso, no en una base de datos compartida.
- **Limitación del plan gratuito:** el disco es temporal. Si el servicio
  se reinicia o se redespliega, se pierden los manuales subidos y el
  índice de ChromaDB, y hay que volver a indexar.

## Próximos pasos

- Persistencia del índice entre despliegues (disco persistente de Render,
  o una base vectorial gestionada) para no depender de reindexar en cada
  reinicio.
- Evaluar la precisión de las respuestas contra un set de preguntas de
  prueba basadas en manuales reales.
- Explorar un enfoque de agente que, además de responder, pueda escalar
  automáticamente un ticket a soporte humano cuando
  `requiere_soporte_humano` sea `true`.
