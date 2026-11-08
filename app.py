"""
app.py — Asistente de Soporte Técnico (agente con tools + RAG + Groq) con interfaz de chat.
(Avance 3: /api/ask ahora usa el agente de agent.py en lugar del pipeline fijo.)

    python app.py     →     http://127.0.0.1:5000
    (ábrelo en VS Code con: Ctrl+Shift+P → "Simple Browser: Show")
"""
from __future__ import annotations

import os
import re
import threading
import traceback
import uuid
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, session

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

import rag_core as rag  # noqa: E402  (después de cargar el .env)
from agent import AgenteSoporte  # noqa: E402
from agent_tools import DOMINIOS_WEB_PERMITIDOS  # noqa: E402

MANUALES_DIR = BASE_DIR / "manuales"
MANUALES_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "clave-dev-asistente-soporte-tecnico")
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024  # manuales de hasta 300 MB
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.config["TEMPLATES_AUTO_RELOAD"] = True

engine = rag.RagEngine()
agente = AgenteSoporte(engine)

SETTINGS = {
    "api_key": os.getenv("GROQ_API_KEY", "").strip(),
    "model": os.getenv("GROQ_MODEL", rag.DEFAULT_GROQ_MODEL),
    "k": 6,
}

# Historial de conversación por sesión del navegador:
# {session_id: [{"pregunta": ..., "respuesta_cruda": ...}, ...]}
HISTORIALES: dict[str, list[dict]] = {}
HIST_LOCK = threading.Lock()


def session_id() -> str:
    if "sid" not in session:
        session["sid"] = uuid.uuid4().hex
    return session["sid"]


def archivos_manuales() -> list[str]:
    """Todos los archivos con formato soportado (PDF, DOCX, TXT, HTML) en manuales/."""
    return sorted(str(p) for p in MANUALES_DIR.iterdir()
                  if p.is_file() and p.suffix.lower() in rag.EXTENSIONES_SOPORTADAS)


# ── Estado de la tarea en segundo plano (indexar / cargar índice) ────────────
LOCK = threading.Lock()
STATE = {"state": "idle", "progress": 0.0, "message": "", "error": None,
         "result": None, "id": 0, "docs": []}


def log(msg: str):
    try:
        print(f"[RAG] {msg}", flush=True)
    except Exception:
        pass


def progress_cb(fraction: float, message: str):
    with LOCK:
        STATE["progress"] = float(fraction)
        STATE["message"] = message
    log(f"{int(fraction * 100):3d}%  {message}")


def run_task(label: str, fn) -> bool:
    with LOCK:
        if STATE["state"] == "running":
            return False
        STATE.update(state="running", progress=0.0, message=label, error=None,
                     result=None, id=STATE["id"] + 1)

    def worker():
        try:
            res = fn()
        except Exception as exc:
            traceback.print_exc()
            try:
                docs = engine.indexed_sources()
            except Exception:
                docs = None
            with LOCK:
                if docs is not None:
                    STATE["docs"] = docs
                STATE.update(state="error", error=(str(exc).strip() or type(exc).__name__)[:500])
        else:
            with LOCK:
                if isinstance(res, dict) and "docs" in res:
                    STATE["docs"] = res["docs"]
                STATE.update(state="done", progress=1.0, result=res)

    threading.Thread(target=worker, daemon=True).start()
    return True


def stats_dict(s):
    return {"archivos": s.archivos, "paginas": s.paginas, "fragmentos": s.fragmentos, "avisos": s.avisos}


def index_job(paths):
    stats = engine.index(paths, progress=progress_cb, replace=not engine.ready)
    return {"kind": "index", "docs": engine.indexed_sources(), "stats": stats_dict(stats)}


def clean_name(name: str) -> str:
    name = os.path.basename((name or "").replace("\\", "/"))
    name = re.sub(r"[^\w\-. ()]", "_", name).strip(" .")
    return name or "manual"


def bootstrap():
    """Al arrancar: abre el índice guardado o, la primera vez, indexa manuales/."""
    hay_indice = rag.PERSIST_DIR.exists()
    pdfs = archivos_manuales()
    if not hay_indice and not pdfs:
        return

    def job():
        loaded = engine.load_existing() if hay_indice else 0
        stats = None
        if not hay_indice:
            stats = engine.index(pdfs, progress=progress_cb, replace=True)
        return {"kind": "bootstrap", "loaded": loaded, "docs": engine.indexed_sources(),
                "stats": stats_dict(stats) if stats else None}

    run_task("Cargando modelo de embeddings…", job)


# ── Rutas ────────────────────────────────────────────────────────────────────
@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/status")
def status():
    with LOCK:
        task = {k: STATE[k] for k in ("state", "progress", "message", "error", "result", "id")}
        docs = list(STATE["docs"])
    count = engine.count() if (engine.ready and task["state"] != "running") else None
    return jsonify(ready=engine.ready, count=count, docs=docs, task=task,
                   has_key=bool(SETTINGS["api_key"]), model=SETTINGS["model"], k=SETTINGS["k"],
                   web_enabled=bool(os.getenv("TAVILY_API_KEY", "").strip()))


@app.post("/api/upload")
def upload():
    if STATE["state"] == "running":
        return jsonify(error="Estoy ocupado indexando, espera a que termine."), 409
    guardados, omitidos = [], []
    for f in request.files.getlist("files"):
        name = clean_name(f.filename)
        ext = Path(name).suffix.lower()
        if ext not in rag.EXTENSIONES_SOPORTADAS:
            omitidos.append(f"{name}: formato no soportado (usa PDF, DOCX, TXT o HTML)")
            continue
        if ext == ".pdf":
            es_pdf = f.stream.read(5) == b"%PDF-"
            f.stream.seek(0)
            if not es_pdf:
                omitidos.append(f"{name}: no parece un PDF válido")
                continue
        if name in STATE["docs"]:
            omitidos.append(f"{name}: ya estaba indexado")
            continue
        dest = MANUALES_DIR / name
        f.save(dest)
        guardados.append(str(dest))
    if not guardados:
        return jsonify(error="No recibí ningún manual nuevo. " + " ".join(omitidos), omitidos=omitidos), 400
    run_task("Preparando…", lambda: index_job(guardados))
    return jsonify(started=True, omitidos=omitidos), 202


@app.post("/api/sync")
def sync():
    """Indexa los manuales de la carpeta manuales/ que todavía no están en el índice."""
    if STATE["state"] == "running":
        return jsonify(error="Estoy ocupado indexando, espera a que termine."), 409
    nuevos = [p for p in archivos_manuales() if os.path.basename(p) not in STATE["docs"]]
    if not nuevos:
        return jsonify(started=False, message="No hay manuales nuevos en la carpeta manuales/.")
    run_task("Preparando…", lambda: index_job(nuevos))
    return jsonify(started=True, archivos=[os.path.basename(p) for p in nuevos]), 202


@app.post("/api/ask")
def ask():
    data = request.get_json(silent=True) or {}
    pregunta = (data.get("question") or "").strip()
    if not pregunta:
        return jsonify(error="Escribe una pregunta."), 400
    if STATE["state"] == "running":
        return jsonify(error="Estoy indexando, dame un momento."), 409
    if not engine.ready:
        return jsonify(error="Todavía no tengo manuales. Agrega uno primero."), 400
    if not SETTINGS["api_key"]:
        return jsonify(error="Falta la API key de Groq.", code="no_key"), 400
    sid = session_id()
    with HIST_LOCK:
        historial = list(HISTORIALES.get(sid, []))
    try:
        res = agente.responder(pregunta, historial=historial, k=SETTINGS["k"],
                               model=SETTINGS["model"], api_key=SETTINGS["api_key"], sid=sid)
    except Exception as exc:
        traceback.print_exc()
        return jsonify(error=(str(exc).strip() or type(exc).__name__)[:500]), 500
    with HIST_LOCK:
        h = HISTORIALES.setdefault(sid, [])
        h.append({"pregunta": res.pregunta, "respuesta_cruda": res.respuesta_cruda})
        HISTORIALES[sid] = h[-rag.MAX_TURNOS_HISTORIAL:]
        turno = len(HISTORIALES[sid])
    return jsonify(
        turno=turno,
        respuesta=res.respuesta,
        error_parseo=res.error_parseo,
        respuesta_cruda=res.respuesta_cruda,
        # Fuentes separadas por origen (salen de lo que devolvieron las tools):
        fuentes=res.fuentes_base(),
        fragmentos=res.fragmentos,           # base: documento, página y texto
        web=res.resultados_web,              # web: título, URL y extracto
        # Trazabilidad del agente:
        tools=res.tools_usadas,
        llamadas_llm=res.llamadas_llm,
        tokens=res.tokens,
        tiempo_s=round(res.tiempo_s, 2),
        limite_alcanzado=res.limite_alcanzado,
    )


@app.post("/api/settings")
def settings():
    d = request.get_json(silent=True) or {}
    if (d.get("api_key") or "").strip():
        SETTINGS["api_key"] = d["api_key"].strip()
    if (d.get("model") or "").strip():
        SETTINGS["model"] = d["model"].strip()
    try:
        SETTINGS["k"] = max(1, min(20, int(d.get("k", SETTINGS["k"]))))
    except (TypeError, ValueError):
        pass
    return jsonify(ok=True, has_key=bool(SETTINGS["api_key"]), model=SETTINGS["model"], k=SETTINGS["k"])


@app.post("/api/reset_chat")
def reset_chat():
    """Empieza una conversación nueva: borra el historial de esta sesión."""
    with HIST_LOCK:
        HISTORIALES.pop(session_id(), None)
    return jsonify(ok=True)


@app.post("/api/reset")
def reset():
    if STATE["state"] == "running":
        return jsonify(error="Estoy ocupado indexando, espera a que termine."), 409
    try:
        engine.reset()
    except Exception as exc:
        return jsonify(error=str(exc)[:500]), 500
    with LOCK:
        STATE["docs"] = []
    with HIST_LOCK:
        HISTORIALES.clear()
    return jsonify(ok=True)


# Se llama aquí (no solo en __main__) para que también corra en Render con gunicorn:
# así los PDFs que dejes en manuales/ del repo se indexan solos al arrancar.
bootstrap()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False, threaded=True)
