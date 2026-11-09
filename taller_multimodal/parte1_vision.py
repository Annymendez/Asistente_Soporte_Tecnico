"""Parte 1 - Visión sobre imágenes del dominio (soporte técnico).

Uso:
    python parte1_vision.py            # corre 1.1, 1.2, 1.3 y 1.4
    python parte1_vision.py 1.2        # corre solo un punto
"""
import json
import sys

from pydantic import BaseModel, ValidationError

from langchain_core.messages import HumanMessage

from comun import (IMAGENES, MODELO_SIN_VISION, cargar_llm,
                   preguntar_con_imagenes, titulo)

# Cambia estos nombres por los de TUS imágenes (deben estar en imagenes/)
IMG_PREGUNTA = IMAGENES / "placa1.jpg"
IMGS_EXTRACCION = [IMAGENES / "placa1.jpg", IMAGENES / "placa2.jpg", IMAGENES / "placa3.jpg"]
IMGS_COMPARACION = [IMAGENES / "manual_a.jpg", IMAGENES / "manual_b.jpg"]


# ---------------------------------------------------------------- 1.1
def punto_1_1():
    titulo("1.1 Pregunta sobre una imagen")
    pregunta = ("Eres técnico de soporte. Según la placa de este equipo, ¿a qué voltaje "
                "y frecuencia se puede conectar? ¿Sirve en un tomacorriente de Colombia (120 V, 60 Hz)?")
    print("Imagen:", IMG_PREGUNTA.name)
    print("Pregunta:", pregunta)
    print("\nRespuesta:\n", preguntar_con_imagenes(pregunta, [IMG_PREGUNTA]))


# ---------------------------------------------------------------- 1.2
class PlacaEquipo(BaseModel):
    """Esquema: datos de la placa técnica de un equipo."""
    marca: str | None
    modelo: str | None
    numero_serie: str | None
    voltaje: str | None          # ej. "100-240V~"
    frecuencia: str | None       # ej. "50/60Hz"
    potencia_o_corriente: str | None  # ej. "65W" o "1.5A"


PROMPT_EXTRACCION = """Extrae los datos de la placa/etiqueta técnica de la imagen.
Responde SOLO con un JSON con exactamente estas claves:
{"marca": str, "modelo": str, "numero_serie": str, "voltaje": str,
 "frecuencia": str, "potencia_o_corriente": str}
Copia los valores tal como aparecen. Si un dato no se ve, usa null. No inventes."""


def normalizar(v) -> str:
    return "" if v is None else str(v).lower().replace(" ", "").replace("~", "")


def punto_1_2():
    import time
    titulo("1.2 Extracción estructurada")
    reales = json.loads((IMAGENES / "valores_reales.json").read_text(encoding="utf-8"))
    aciertos = total = 0

    for ruta in IMGS_EXTRACCION:
        print(f"\n--- {ruta.name} ---")
        try:
            crudo = preguntar_con_imagenes(PROMPT_EXTRACCION, [ruta])
        except Exception as e:
            print(f"[AVISO] La API falló con esta imagen: {e}")
            time.sleep(60)
            continue
        print("Salida cruda:", crudo)

        # Nos quedamos solo con lo que está entre la primera { y la última }
        inicio, fin = crudo.find("{"), crudo.rfind("}")
        crudo = crudo[inicio:fin + 1] if inicio != -1 else crudo

        # Validación: primero que sea JSON, luego que cumpla el esquema
        try:
            datos = PlacaEquipo.model_validate(json.loads(crudo))
        except json.JSONDecodeError as e:
            print(f"[AVISO] La respuesta no es JSON válido: {e}")
            continue
        except ValidationError as e:
            print(f"[AVISO] El JSON no cumple el esquema:\n{e}")
            continue

        real = reales[ruta.name]
        for campo, extraido in datos.model_dump().items():
            ok = normalizar(extraido) == normalizar(real.get(campo))
            aciertos += ok
            total += 1
            print(f"{campo:22} real={real.get(campo)!s:20} extraído={extraido!s:20} {'Sí' if ok else 'No'}")

        time.sleep(60)  # límite gratis de Groq: 1000 tokens de salida por minuto

    if total:
        print(f"\nPorcentaje de aciertos: {aciertos} de {total} campos = {100 * aciertos / total:.1f} %")

# ---------------------------------------------------------------- 1.3
def punto_1_3():
    titulo("1.3 Varias imágenes en una solicitud")
    pregunta = ("Estas imágenes son de dos manuales/equipos relacionados. Compáralos: "
                "¿en qué se diferencian sus especificaciones y cuál recomendarías para una "
                "oficina en Colombia? Responde en una tabla corta y una conclusión.")
    print("Imágenes:", [r.name for r in IMGS_COMPARACION])
    print("Pregunta:", pregunta)
    print("\nRespuesta:\n", preguntar_con_imagenes(pregunta, IMGS_COMPARACION))


# ---------------------------------------------------------------- 1.4
def punto_1_4():
    titulo("1.4 Errores provocados")
    pruebas = {
        "A) Más de 3 imágenes en una solicitud (límite = 3)":
            lambda: preguntar_con_imagenes("Describe las imágenes", [IMG_PREGUNTA] * 4),
        "B) Modelo SIN visión recibiendo una imagen":
            lambda: preguntar_con_imagenes("¿Qué ves?", [IMG_PREGUNTA], modelo=MODELO_SIN_VISION),
        "C) Tipo de imagen mal declarado ('image/gif' con datos que no son imagen)":
            lambda: cargar_llm().invoke([HumanMessage(content=[
                {"type": "text", "text": "¿Qué ves?"},
                {"type": "image", "base64": "ESTO_NO_ES_UNA_IMAGEN", "mime_type": "image/gif"},
            ])]),
    }
    for nombre, prueba in pruebas.items():
        print(f"\n{nombre}")
        try:
            prueba()
            print("   (la API NO devolvió error)")
        except Exception as e:  # la librería de Groq lanza APIStatusError / BadRequestError
            print(f"   Tipo: {type(e).__name__}")
            print(f"   Código HTTP: {getattr(e, 'status_code', '-')}")
            print(f"   Mensaje: {e}")


PUNTOS = {"1.1": punto_1_1, "1.2": punto_1_2, "1.3": punto_1_3, "1.4": punto_1_4}

if __name__ == "__main__":
    elegidos = sys.argv[1:] or list(PUNTOS)
    for p in elegidos:
        PUNTOS[p]()
