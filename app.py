"""
app.py — Interfaz web (Flask) del Asistente de Soporte Técnico (Avance 2).

Expone:
  GET  /                -> interfaz de chat
  GET  /api/status      -> estado de la indexación en segundo plano
  POST /api/upload      -> subir manuales (PDF/DOCX/TXT/HTML) y reindexar
  POST /api/sync        -> reindexar los manuales ya presentes en manuales/
  POST /api/ask         -> hacer una pregunta (usa el historial de la sesión)
  POST /api/reset_chat  -> borrar el historial de conversación de la sesión
  POST /api/settings    -> guardar la API key de Groq y parámetros (k, modelo)
  POST /api/reset       -> borrar el índice vectorial completo
"""
from __future__ import annotations

import os
import threading
import uuid
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request, session

load_dotenv()

import rag_core as rag  # noqa: E402  (después de cargar el .env)

BASE_DIR = Path(__file__).resolve().parent
MANUALES_DIR = BASE_DIR / "manuales"
MANUALES_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "clave-dev-asistente-soporte-tecnico")

engine = rag.RagEngine()

# Estado de la indexación en segundo plano, compartido entre hilos.
LOCK = threading.Lock()
STATE = {
    "indexando": False,
    "progreso": 0.0,
    "mensaje": "",
    "error": None,
    "ultimo_resultado": None,
}

# Historial de conversación por sesión: {session_id: [{"pregunta":..., "respuesta_cruda":...}, ...]}
HISTORIALES: dict[str, list[dict]] = {}
HIST_LOCK = threading.Lock()

CONFIG = {
    "groq_api_key": os.environ.get("GROQ_API_KEY", ""),
    "modelo": rag.DEFAULT_GROQ_MODEL,
    "k": 6,
}


def _session_id() -> str:
    if "sid" not in session:
        session["sid"] = uuid.uuid4().hex
    return session["sid"]


def _archivos_manuales() -> list[Path]:
    """Lista todos los archivos con extensión soportada dentro de manuales/."""
    return sorted(
        p for p in MANUALES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in rag.EXTENSIONES_SOPORTADAS
    )


def _indexar_en_segundo_plano(paths: list[Path], replace: bool = True):
    def tarea():
        with LOCK:
            STATE["indexando"] = True
            STATE["error"] = None
            STATE["progreso"] = 0.0
            STATE["mensaje"] = "Iniciando…"

        def progreso(frac, msg):
            with LOCK:
                STATE["progreso"] = frac
                STATE["mensaje"] = msg

        try:
            stats = engine.index(paths, progress=progreso, replace=replace)
            with LOCK:
                STATE["ultimo_resultado"] = {
                    "archivos": stats.archivos,
                    "fragmentos": stats.fragmentos,
                    "avisos": stats.avisos,
                }
                STATE["mensaje"] = "Indexación completa."
                STATE["progreso"] = 1.0
        except Exception as e:
            with LOCK:
                STATE["error"] = str(e)
                STATE["mensaje"] = "Error durante la indexación."
        finally:
            with LOCK:
                STATE["indexando"] = False

    threading.Thread(target=tarea, daemon=True).start()


def bootstrap():
    """Se ejecuta al importar el módulo (también bajo gunicorn): intenta
    cargar un índice ya persistido en disco; si no existe, indexa lo que
    haya en manuales/ automáticamente."""
    n = engine.load_existing()
    if n > 0:
        STATE["mensaje"] = f"Índice existente cargado ({n} fragmentos)."
        STATE["progreso"] = 1.0
        return

    archivos = _archivos_manuales()
    if archivos:
        _indexar_en_segundo_plano(archivos, replace=True)
    else:
        STATE["mensaje"] = "Sin manuales todavía. Sube uno o varios para comenzar."


bootstrap()


# ── Rutas ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    from flask import render_template

    return render_template("index.html")


@app.route("/api/status")
def api_status():
    with LOCK:
        estado = dict(STATE)
    estado["listo"] = engine.ready
    estado["num_fragmentos"] = engine.count()
    estado["fuentes"] = engine.indexed_sources()
    return jsonify(estado)


@app.route("/api/upload", methods=["POST"])
def api_upload():
    archivos = request.files.getlist("archivos")
    if not archivos:
        return jsonify({"error": "No se recibió ningún archivo."}), 400

    guardados = []
    rechazados = []
    for f in archivos:
        if not f.filename:
            continue
        ext = Path(f.filename).suffix.lower()
        if ext not in rag.EXTENSIONES_SOPORTADAS:
            rechazados.append(f.filename)
            continue
        destino = MANUALES_DIR / Path(f.filename).name
        f.save(str(destino))
        guardados.append(destino.name)

    if not guardados:
        return jsonify({
            "error": "Ningún archivo tenía un formato soportado (PDF, DOCX, TXT, HTML).",
            "rechazados": rechazados,
        }), 400

    with LOCK:
        if STATE["indexando"]:
            return jsonify({"error": "Ya hay una indexación en curso, espera a que termine."}), 409

    _indexar_en_segundo_plano(_archivos_manuales(), replace=True)
    return jsonify({"guardados": guardados, "rechazados": rechazados, "indexando": True})


@app.route("/api/sync", methods=["POST"])
def api_sync():
    archivos = _archivos_manuales()
    if not archivos:
        return jsonify({"error": "No hay manuales en la carpeta manuales/."}), 400
    with LOCK:
        if STATE["indexando"]:
            return jsonify({"error": "Ya hay una indexación en curso."}), 409
    _indexar_en_segundo_plano(archivos, replace=True)
    return jsonify({"indexando": True, "archivos": [a.name for a in archivos]})


@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json(silent=True) or {}
    pregunta = (data.get("pregunta") or "").strip()
    if not pregunta:
        return jsonify({"error": "La pregunta no puede estar vacía."}), 400

    if not engine.ready:
        return jsonify({"error": "El asistente aún no tiene manuales indexados."}), 400

    sid = _session_id()
    with HIST_LOCK:
        historial = list(HISTORIALES.get(sid, []))

    try:
        resultado = engine.ask(
            pregunta,
            historial=historial,
            k=CONFIG["k"],
            model=CONFIG["modelo"],
            api_key=CONFIG["groq_api_key"] or None,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    with HIST_LOCK:
        HISTORIALES.setdefault(sid, []).append({
            "pregunta": resultado.pregunta,
            "respuesta_cruda": resultado.respuesta_cruda,
        })
        # limitamos el historial guardado para no crecer indefinidamente
        HISTORIALES[sid] = HISTORIALES[sid][-rag.MAX_TURNOS_HISTORIAL:]

    return jsonify({
        "pregunta": resultado.pregunta,
        "respuesta": resultado.respuesta,
        "respuesta_cruda": resultado.respuesta_cruda,
        "error_parseo": resultado.error_parseo,
        "fuentes": resultado.fuentes(),
        "prompt_debug": resultado.prompt_texto(),
        "num_turno": len(HISTORIALES.get(sid, [])),
    })


@app.route("/api/reset_chat", methods=["POST"])
def api_reset_chat():
    sid = _session_id()
    with HIST_LOCK:
        HISTORIALES[sid] = []
    return jsonify({"ok": True})


@app.route("/api/settings", methods=["POST"])
def api_settings():
    data = request.get_json(silent=True) or {}
    if "groq_api_key" in data:
        CONFIG["groq_api_key"] = (data.get("groq_api_key") or "").strip()
    if "modelo" in data and data["modelo"]:
        CONFIG["modelo"] = data["modelo"]
    if "k" in data and data["k"]:
        try:
            CONFIG["k"] = max(1, min(20, int(data["k"])))
        except (TypeError, ValueError):
            pass
    return jsonify({"ok": True, "k": CONFIG["k"], "modelo": CONFIG["modelo"]})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    engine.reset()
    with HIST_LOCK:
        HISTORIALES.clear()
    with LOCK:
        STATE["ultimo_resultado"] = None
        STATE["mensaje"] = "Índice borrado."
        STATE["progreso"] = 0.0
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=os.environ.get("FLASK_DEBUG") == "1",
        use_reloader=False,
        threaded=True,
    )
