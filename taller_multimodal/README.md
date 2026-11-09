# Taller: Entrada multimodal y audio

Dominio: asistente de soporte técnico (placas y manuales de equipos).

## Requisitos

```bash
pip install -r requirements.txt
cp .env.example .env   # y pon tu clave en .env
```

Variables de entorno (archivo `.env`, que **no** se sube):

| Variable | Obligatoria | Por defecto |
|---|---|---|
| `GROQ_API_KEY` | Sí | — |
| `MODELO_VISION` | No | `qwen/qwen3.8-27b` |
| `LLM_MODEL` | No (modelo sin visión, para el error 1.4) | `openai/gpt-oss-20b` |
| `MODELO_AUDIO` | No | `whisper-large-v3-turbo` |

## Parte 1 – Visión

1. Pon al menos 5 imágenes en `imagenes/` (placa1-3.jpg, manual_a.jpg, manual_b.jpg).
2. Llena `imagenes/valores_reales.json` con lo que se lee en cada placa.
3. Ejecuta:

```bash
python parte1_vision.py        # todos los puntos
python parte1_vision.py 1.2    # solo uno (1.1, 1.2, 1.3, 1.4)
```

## Parte 2 – Audio

1. Pon los audios (10-30 s) en `audios/`.
2. Ejecuta `python parte2_audio.py`. Los resultados quedan en `audios/resultados.json`.
