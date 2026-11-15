# Asistente Experto de Soporte Técnico basado en RAG y Agentes

## Descripción del proyecto

Este proyecto desarrolla un asistente de IA que funciona de manera local
(preservando la privacidad de los datos) capaz de leer una base de
conocimientos específica en este caso, manuales técnicos e instructivos
de producto para responder preguntas de soporte técnico y guiar al
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
2. Recupera del manual técnico (base de conocimiento local) los
   fragmentos más relevantes para esa pregunta (retrieval — se
   implementará en el Avance 2 con embeddings + vector store local).
3. Combina esos fragmentos con la pregunta dentro de un prompt
   estructurado con delimitadores XML.
4. Genera una respuesta estructurada en JSON con: diagnóstico, pasos de
   solución, nivel de urgencia, fuente citada del manual, y si requiere
   escalar a soporte humano.

## Avance 1 — Diseño de Prompts

Este avance se enfoca exclusivamente en el **diseño del prompt**, sin aún
conectar el pipeline de recuperación (RAG) completo. Se trabajaron tres
componentes:

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

## Estructura del repositorio

```
├── README.md
├── src/
│   └── asistente_soporte_tecnico.py   # System prompt, few-shot, delimitadores
├── docs/
│   └── ejecucion_prompt.pdf           # Evidencia de ejecución del prompt
└── manuales/
    └── (manual técnico usado como base de conocimiento)
```

## Formato de salida

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

## Próximos pasos (Avance 2)

- Implementar el pipeline de recuperación (RAG) con embeddings locales
  (ej. `sentence-transformers`) y una base vectorial local (ej. Chroma,
  FAISS).
- Conectar el prompt diseñado en este avance a un LLM ejecutado localmente
  (ej. Ollama con Llama 3 o Mistral) para preservar la privacidad de los
  datos.
- Evaluar la precisión de las respuestas contra un set de preguntas de
  prueba basadas en el manual real.


## Avance 3 — agente con tools (rama liviana para Render)

Esta rama (`deploy-estable`) incluye el agente del Avance 3 pensado para el plan gratuito de Render (512 MB):

- `agent.py`, `agent_tools.py`, `agent_prompt.py`: bucle manual del agente, las 3 tools (`buscar_en_base`, `calcular_garantia`, `buscar_en_web`) y el prompt. Son los mismos archivos de la rama `main`.
- `manuales/` va dentro del repositorio (2 PDF, 1 TXT, 1 HTML): al arrancar, el servidor los indexa solo, sin subir nada desde la interfaz.
- `rag_core.py`: lee PDF, TXT y HTML solo con la biblioteca estándar (sin docx, bs4 ni lxml) y desactiva la telemetría de ChromaDB.
- No incluye `eval/` ni el historial de conversación; la evaluación completa (Ragas, casos y costo) está en `main`.

Variables de entorno en Render: `GROQ_API_KEY` y `TAVILY_API_KEY` (nunca en el repositorio). Sin `TAVILY_API_KEY` el agente funciona, pero sin búsqueda web.
