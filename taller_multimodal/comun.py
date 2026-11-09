"""Funciones compartidas por las dos partes del taller (mismo estilo que multimodal.py de clase)."""
import base64
import mimetypes
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

load_dotenv()  # lee GROQ_API_KEY del archivo .env (que NO se sube a GitHub)

MODELO_VISION = os.getenv("MODELO_VISION", "qwen/qwen3.8-27b")
MODELO_SIN_VISION = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")  # solo texto (para el error 1.4)
MODELO_AUDIO = os.getenv("MODELO_AUDIO", "whisper-large-v3-turbo")

CARPETA = Path(__file__).parent
IMAGENES = CARPETA / "imagenes"
AUDIOS = CARPETA / "audios"


def cargar_llm(model_name=MODELO_VISION, temperature=0, **kwargs):
    """Igual que utils/load_model.py del repo de clase."""
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("Falta la variable GROQ_API_KEY en el archivo .env")
    return ChatGroq(model=model_name, temperature=temperature, **kwargs)


def bloque_imagen(ruta: Path, mime: str | None = None) -> dict:
    """Imagen local -> bloque base64 en el formato de LangChain (como en multimodal.py)."""
    mime = mime or mimetypes.guess_type(ruta)[0] or "image/png"
    img_b64 = base64.b64encode(Path(ruta).read_bytes()).decode("utf-8")
    return {"type": "image", "base64": img_b64, "mime_type": mime}


def preguntar_con_imagenes(pregunta: str, rutas: list[Path], modelo: str = MODELO_VISION,
                           json_mode: bool = False, sistema: str | None = None) -> str:
    """Envía texto + una o varias imágenes en UN solo HumanMessage."""
    llm = cargar_llm(model_name=modelo, max_tokens=800, reasoning_effort="low") \
        if modelo == MODELO_VISION else cargar_llm(model_name=modelo)
    if json_mode:
        llm = llm.bind(response_format={"type": "json_object"})
    mensajes = [SystemMessage(sistema)] if sistema else []
    mensajes.append(HumanMessage(content=[{"type": "text", "text": pregunta}]
                                 + [bloque_imagen(r) for r in rutas]))
    return llm.invoke(mensajes).content


def titulo(texto: str) -> None:
    print("\n" + "=" * 70 + f"\n{texto}\n" + "=" * 70)


groq_client = Groq()  # cliente directo, solo para Whisper (como en audio_input.py)
