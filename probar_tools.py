"""
probar_tools.py — Prueba las tools del agente SIN el LLM (Avance 3, requisito 1.1).

Llama cada tool con .invoke() y muestra exactamente lo que recibiría el LLM.
No necesita GROQ_API_KEY (buscar_en_web solo necesita TAVILY_API_KEY si quieres
ver una búsqueda real).

    python probar_tools.py
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

import rag_core as rag  # noqa: E402
from agent_tools import crear_tools  # noqa: E402


def titulo(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


engine = rag.RagEngine()
if engine.load_existing() == 0:
    print("Indexando manuales/ (primera vez)…")
    archivos = sorted(p for p in (BASE_DIR / "manuales").iterdir()
                      if p.suffix.lower() in rag.EXTENSIONES_SOPORTADAS)
    engine.index(archivos, progress=lambda f, m: None)
print(f"Índice listo: {engine.count()} fragmentos de {', '.join(engine.indexed_sources())}")

tools = {t.name: t for t in crear_tools(engine)}

titulo('buscar_en_base.invoke({"consulta": "LED rojo fijo qué significa"})')
print(tools["buscar_en_base"].invoke({"consulta": "LED rojo fijo qué significa"}))

titulo('calcular_garantia.invoke({"fecha_compra": "2026-03-15"})')
print(tools["calcular_garantia"].invoke({"fecha_compra": "2026-03-15"}))

titulo('calcular_garantia.invoke({"fecha_compra": "2025-01-02"})   # ya vencida')
print(tools["calcular_garantia"].invoke({"fecha_compra": "2025-01-02"}))

titulo('calcular_garantia.invoke({"fecha_compra": "ayer"})   # argumento inválido')
print(tools["calcular_garantia"].invoke({"fecha_compra": "ayer"}))

titulo('buscar_en_web.invoke({"consulta": "qué es WPA3"})')
if os.getenv("TAVILY_API_KEY"):
    print(tools["buscar_en_web"].invoke({"consulta": "qué es WPA3"}))
else:
    print("(sin TAVILY_API_KEY) ->", tools["buscar_en_web"].invoke({"consulta": "qué es WPA3"}))

sys.exit(0)
