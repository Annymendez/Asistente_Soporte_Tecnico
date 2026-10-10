"""
rag_core.py — Núcleo RAG del Asistente de Soporte Técnico (Avance 2).

Conecta el diseño de prompts del Avance 1 (asistente_soporte_tecnico.py:
SYSTEM_PROMPT, FEW_SHOT_EXAMPLES, construir_mensajes_para_llm) con un
pipeline RAG real y conversacional:

    Manuales (PDF/DOCX/TXT/HTML) -> fragmentos -> embeddings locales -> ChromaDB
                 -> búsqueda -> mensajes (system + few-shot + historial + <contexto>/<pregunta>)
                 -> LLM (Groq) -> respuesta JSON estructurada

No se modifica asistente_soporte_tecnico.py: se importa tal cual para que
el Avance 1 quede intacto como evidencia entregada.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from asistente_soporte_tecnico import construir_mensajes_para_llm

# ── Configuración ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIR = BASE_DIR / "chroma"
COLLECTION_NAME = "manuales_soporte"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Extensiones de archivo soportadas para la ingesta (Avance 2: PDF, DOCX, TXT, HTML)
EXTENSIONES_SOPORTADAS = (".pdf", ".docx", ".txt", ".html", ".htm")

ProgressCallback = Callable[[float, str], None]

CAMPOS_JSON = ("diagnostico", "pasos_solucion", "nivel_urgencia", "fuente", "requiere_soporte_humano")

# Máximo de turnos previos (pregunta+respuesta) que se inyectan como historial real
# en los mensajes enviados al LLM. Se limita para no exceder el contexto del modelo.
MAX_TURNOS_HISTORIAL = 6


# ── Utilidades ──────────────────────────────────────────────────────────────
def nombre_fuente(doc) -> str:
    return os.path.basename(str(doc.metadata.get("source", "desconocido")))


def numero_pagina(doc) -> str:
    """PyPDFLoader numera desde 0; mostramos la página tal como la ve el usuario.
    Para DOCX/TXT/HTML no hay noción de página, así que se usa '-' como marcador."""
    p = doc.metadata.get("page")
    return str(p + 1) if isinstance(p, int) else "-"


def _limpiar_metadata(doc):
    """Chroma solo acepta str/int/float/bool en los metadatos."""
    doc.metadata = {
        k: v for k, v in doc.metadata.items() if isinstance(v, (str, int, float, bool))
    }
    return doc


def _cargar_documento(path: Path | str):
    """Elige el loader de LangChain según la extensión del archivo.
    Soporta PDF, DOCX, TXT y HTML/HTM, tal como exige el Avance 2."""
    ruta = Path(path)
    ext = ruta.suffix.lower()

    if ext == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader

        return PyPDFLoader(str(ruta)).load()

    if ext == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader

        return Docx2txtLoader(str(ruta)).load()

    if ext == ".txt":
        from langchain_community.document_loaders import TextLoader

        return TextLoader(str(ruta), encoding="utf-8").load()

    if ext in (".html", ".htm"):
        from langchain_community.document_loaders import BSHTMLLoader

        return BSHTMLLoader(str(ruta), open_encoding="utf-8").load()

    raise ValueError(
        f"Formato no soportado: '{ext}'. Formatos permitidos: "
        + ", ".join(EXTENSIONES_SOPORTADAS)
    )


def _a_mensajes_langchain(mensajes: list[dict]):
    """Convierte los dicts {"role", "content"} de construir_mensajes_para_llm
    a objetos BaseMessage, que es lo que espera ChatGroq.invoke()."""
    clases = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}
    return [clases[m["role"]](content=m["content"]) for m in mensajes]


def _mensajes_con_historial(mensajes: list[dict], historial: list[dict] | None) -> list[dict]:
    """Inserta los turnos previos reales (pregunta del usuario + respuesta cruda
    del asistente) ENTRE los ejemplos few-shot y el turno real final, para que el
    modelo pueda responder preguntas de seguimiento que dependan de turnos
    anteriores (requisito de 'interfaz conversacional real' del Avance 2).

    mensajes: salida de construir_mensajes_para_llm — [system, few-shot..., turno_real]
    historial: lista de dicts {"pregunta": str, "respuesta_cruda": str}, en orden
               cronológico, ya limitada a MAX_TURNOS_HISTORIAL turnos.
    """
    if not historial:
        return mensajes

    turno_real = mensajes[-1]
    base = mensajes[:-1]  # system + few-shot

    turnos_previos = []
    for turno in historial[-MAX_TURNOS_HISTORIAL:]:
        turnos_previos.append({"role": "user", "content": turno["pregunta"]})
        turnos_previos.append({"role": "assistant", "content": turno["respuesta_cruda"]})

    return base + turnos_previos + [turno_real]


def parsear_json_respuesta(texto: str) -> tuple[dict | None, str | None]:
    """Extrae el objeto JSON de la respuesta del modelo, tolerando que venga
    envuelto en ```json ... ``` o con texto alrededor pese a la instrucción
    del system prompt. Devuelve (dict, None) o (None, motivo_del_error)."""
    limpio = re.sub(r"```(?:json)?|```", "", texto).strip()
    m = re.search(r"\{.*\}", limpio, re.DOTALL)
    if not m:
        return None, "La respuesta no contenía un objeto JSON."
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return None, f"El JSON no se pudo interpretar ({e})."
    faltan = [c for c in CAMPOS_JSON if c not in data]
    if faltan:
        return None, f"Al JSON le faltan campos: {', '.join(faltan)}."
    return data, None


@dataclass
class IndexStats:
    archivos: int
    paginas: int
    fragmentos: int
    avisos: list[str] = field(default_factory=list)


@dataclass
class RagResult:
    pregunta: str
    fragmentos: list
    mensajes: list          # mensajes system + few-shot + historial + turno real, tal como se enviaron al LLM
    respuesta_cruda: str     # texto tal cual lo devolvió el modelo
    respuesta: dict | None   # JSON ya parseado (diagnostico, pasos_solucion, ...) o None si falló el parseo
    error_parseo: str | None = None

    def fuentes(self) -> str:
        por_archivo: dict[str, set[str]] = {}
        for d in self.fragmentos:
            por_archivo.setdefault(nombre_fuente(d), set()).add(numero_pagina(d))

        def orden(p: str):
            return (0, int(p)) if p.isdigit() else (1, 0)

        return " · ".join(
            f"{nombre} (pág. {', '.join(sorted(pags, key=orden))})"
            if any(p != "-" for p in pags)
            else nombre
            for nombre, pags in por_archivo.items()
        )

    def prompt_texto(self) -> str:
        """Representación legible de los mensajes enviados al LLM, para la
        pestaña de depuración 'Ver prompt enviado'."""
        return "\n\n".join(f"--- {m['role'].upper()} ---\n{m['content']}" for m in self.mensajes)


# ── Motor RAG ───────────────────────────────────────────────────────────────
class RagEngine:
    def __init__(
        self,
        persist_dir: Path | str = PERSIST_DIR,
        collection_name: str = COLLECTION_NAME,
        embeddings=None,
        llm_factory: Optional[Callable] = None,
    ):
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self._embeddings = embeddings
        self._llm_factory = llm_factory  # útil para pruebas sin red
        self._vector_store = None
        self._llm = None
        self._llm_cache_key = None

    # -- estado --------------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self._vector_store is not None

    def count(self) -> int:
        return self._vector_store._collection.count() if self.ready else 0

    def indexed_sources(self) -> list[str]:
        if not self.ready:
            return []
        metas = self._vector_store._collection.get(include=["metadatas"])["metadatas"]
        return sorted({os.path.basename(str(m.get("source", ""))) for m in metas if m})

    def reset(self) -> None:
        if self._vector_store is not None:
            self._vector_store.delete_collection()
            self._vector_store = None

    # -- piezas del pipeline ---------------------------------------------------
    def _get_embeddings(self):
        if self._embeddings is None:
            # El plan gratuito de Render solo da 512 MB de RAM, y CUALQUIER
            # modelo de embeddings cargado EN EL PROCESO (sentence-transformers
            # con PyTorch, o incluso fastembed con ONNX runtime) se queda corto
            # sumado a Flask + ChromaDB + LangChain — se probaron ambos y los
            # dos tumbaron el servicio con "Ran out of memory".
            #
            # La solución es no cargar ningún modelo localmente: se usa la API
            # de inferencia de Hugging Face (HuggingFaceEndpointEmbeddings),
            # que manda el texto a los servidores de Hugging Face y solo
            # devuelve el vector — el proceso de Render casi no gasta memoria
            # en esto. Requiere un token gratuito de Hugging Face en la
            # variable de entorno HF_TOKEN (huggingface.co/settings/tokens).
            from langchain_huggingface import HuggingFaceEndpointEmbeddings

            hf_token = os.getenv("HF_TOKEN")
            if not hf_token:
                raise ValueError(
                    "Falta HF_TOKEN. Crea un token gratuito en "
                    "https://huggingface.co/settings/tokens y defínelo como "
                    "HF_TOKEN en el archivo .env (local) o en las variables de "
                    "entorno de Render."
                )
            self._embeddings = HuggingFaceEndpointEmbeddings(
                model=EMBEDDING_MODEL,
                task="feature-extraction",
                huggingfacehub_api_token=hf_token,
            )
        return self._embeddings

    def _open_store(self):
        from langchain_chroma import Chroma

        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self._get_embeddings(),
            persist_directory=str(self.persist_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )

    def _get_llm(self, model: str, api_key: Optional[str], temperature: float):
        key = api_key or os.getenv("GROQ_API_KEY")
        if self._llm_factory:
            return self._llm_factory(model, key)
        if not key:
            raise ValueError(
                "Falta la API key de Groq. Pégala en el campo de configuración "
                "o defínela como GROQ_API_KEY en el archivo .env."
            )
        cache_key = (model, key, temperature)
        if self._llm_cache_key != cache_key:
            from langchain_groq import ChatGroq

            self._llm = ChatGroq(model=model, temperature=temperature, api_key=key)
            self._llm_cache_key = cache_key
        return self._llm

    # -- reanudar: cargar base vectorial existente -----------------------------
    def load_existing(self) -> int:
        if not self.persist_dir.exists():
            return 0
        store = self._open_store()
        n = store._collection.count()
        self._vector_store = store if n > 0 else None
        return n

    # -- cargar, dividir, embeber, indexar --------------------------------------
    def index(
        self,
        doc_paths: Sequence[Path | str],
        progress: Optional[ProgressCallback] = None,
        replace: bool = True,
    ) -> IndexStats:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        progress = progress or (lambda f, m: None)
        avisos: list[str] = []
        documents = []
        total = max(len(doc_paths), 1)

        for i, path in enumerate(doc_paths):
            nombre = os.path.basename(str(path))
            progress(0.2 * i / total, f"Leyendo {nombre}…")
            try:
                pages = _cargar_documento(path)
            except Exception as e:
                avisos.append(f"{nombre}: no se pudo leer ({e})")
                continue
            con_texto = [p for p in pages if p.page_content.strip()]
            if not con_texto:
                avisos.append(f"{nombre}: sin texto extraíble (¿escaneado o vacío?)")
                continue
            documents.extend(con_texto)

        if not documents:
            raise ValueError(
                "No se pudo extraer texto de ninguno de los manuales.\n" + "\n".join(avisos)
            )

        progress(0.2, "Dividiendo en fragmentos…")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " "],
        )
        chunks = [
            _limpiar_metadata(c)
            for c in splitter.split_documents(documents)
            if c.page_content.strip()
        ]

        progress(0.2, "Cargando modelo de embeddings (la primera vez descarga ~120 MB)…")
        self._get_embeddings()
        if replace:
            self._vector_store = None
            try:
                self._open_store().delete_collection()
            except Exception:
                pass
            store = self._open_store()
        else:  # agregar manuales nuevos al índice existente
            store = self._vector_store or self._open_store()

        lote = 64
        for start in range(0, len(chunks), lote):
            store.add_documents(chunks[start : start + lote])
            hecho = min(start + lote, len(chunks))
            progress(0.2 + 0.8 * hecho / len(chunks), f"Indexando fragmentos {hecho}/{len(chunks)}…")

        self._vector_store = store
        archivos = len({nombre_fuente(d) for d in documents})
        return IndexStats(archivos, len(documents), len(chunks), avisos)

    # -- recuperar, armar mensajes (Avance 1 + historial) y generar -------------
    def ask(
        self,
        pregunta: str,
        historial: list[dict] | None = None,
        k: int = 6,
        model: str = DEFAULT_GROQ_MODEL,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
    ) -> RagResult:
        """
        pregunta: la pregunta real del usuario en este turno.
        historial: lista de turnos previos de la MISMA conversación, en orden
            cronológico, cada uno como {"pregunta": str, "respuesta_cruda": str}.
            Permite preguntas de seguimiento ("¿y si no funciona?", "¿y eso es
            peligroso?") que dependen de turnos anteriores — requisito de
            'interfaz conversacional real' del Avance 2.
        """
        if not self.ready:
            raise RuntimeError("Primero indexa los manuales (o carga un índice existente).")

        llm = self._get_llm(model, api_key, temperature)

        # Retrieval: se busca solo con la pregunta del turno actual (la query de
        # búsqueda no incluye el historial para no diluir la similitud semántica
        # con texto de turnos anteriores que puede no ser relevante al nuevo tema).
        retriever = self._vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": k}
        )
        docs = retriever.invoke(pregunta)

        # Los fragmentos que se pasan a construir_mensajes_para_llm (Avance 1):
        # llevan la fuente y página al frente de cada uno para que el propio
        # modelo pueda citarlas en el campo "fuente" del JSON de salida.
        fragmentos_contexto = [
            f"[Fuente: {nombre_fuente(d)} — Pág. {numero_pagina(d)}]\n{d.page_content}" for d in docs
        ]

        mensajes_base = construir_mensajes_para_llm(fragmentos_contexto, pregunta)
        mensajes = _mensajes_con_historial(mensajes_base, historial)

        respuesta_cruda = llm.invoke(_a_mensajes_langchain(mensajes)).content
        if not isinstance(respuesta_cruda, str):
            respuesta_cruda = str(respuesta_cruda)

        respuesta, error = parsear_json_respuesta(respuesta_cruda)

        return RagResult(
            pregunta=pregunta,
            fragmentos=docs,
            mensajes=mensajes,
            respuesta_cruda=respuesta_cruda,
            respuesta=respuesta,
            error_parseo=error,
        )
