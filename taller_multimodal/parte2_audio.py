"""Parte 2 - Audio: transcripción desde archivo (2.1) y respuesta del modelo (2.2).

Basado en audio_input.py de clase, pero leyendo archivos en vez de grabar del micrófono.
Uso:  python parte2_audio.py
Pon los audios en audios/ (wav, mp3, m4a, ogg, webm o mp4; máx. 25 MB cada uno).
"""
import json
import time  # NUEVO

from langchain_core.messages import HumanMessage, SystemMessage

from comun import AUDIOS, MODELO_AUDIO, cargar_llm, groq_client, titulo

# 'prompt' de Whisper: le da contexto y vocabulario técnico -> menos errores en términos raros
VOCABULARIO = ("Soporte técnico: router, firmware, DHCP, dirección IP, puerto Ethernet, "
               "BIOS, driver, impresora, tóner, fuente de poder, voltaje, reinicio de fábrica.")

SISTEMA = ("Eres un asistente experto de soporte técnico. Responde en español, "
           "en máximo 5 pasos claros. Si la pregunta parece mal transcrita, dilo.")


def transcribir(ruta):
    transcripcion = groq_client.audio.transcriptions.create(
        file=(ruta.name, ruta.read_bytes()),
        model=MODELO_AUDIO,
        language="es",
        prompt=VOCABULARIO,
        response_format="verbose_json",  # trae también la duración
        temperature=0,
    )
    return transcripcion.text.strip(), getattr(transcripcion, "duration", None)


llm_model = cargar_llm(max_tokens=800, reasoning_effort="low")  # qwen/qwen3.8-27b, como en clase


def responder(pregunta):
    return llm_model.invoke([SystemMessage(SISTEMA), HumanMessage(pregunta)]).content


if __name__ == "__main__":
    audios = sorted(p for p in AUDIOS.iterdir()
                    if p.suffix.lower() in {".wav", ".mp3", ".m4a", ".ogg", ".webm", ".mp4"})
    resultados = []
    for ruta in audios:
        titulo(f"Audio: {ruta.name}")
        texto, dur = transcribir(ruta)
        print(f"[2.1] Duración: {dur} s\n      Transcripción: {texto}")
        respuesta = responder(texto)
        print(f"\n[2.2] Respuesta del modelo:\n{respuesta}")
        resultados.append({"audio": ruta.name, "duracion_s": dur,
                           "transcripcion": texto, "respuesta": respuesta})
        if ruta != audios[-1]:
            time.sleep(75)  # NUEVO: límite gratis de Groq = 1000 tokens de salida por minuto

    # Guardamos todo para copiarlo fácil a las tablas del informe
    (AUDIOS / "resultados.json").write_text(
        json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nResultados guardados en audios/resultados.json")