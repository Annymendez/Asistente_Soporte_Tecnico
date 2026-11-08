"""
comparar_costo.py — Costo del pipeline fijo (Avance 2) frente al agente (Avance 3).

Para las MISMAS preguntas de eval/preguntas_eval.json mide el promedio de:
    - llamadas al LLM
    - tokens (entrada + salida)
    - tiempo de respuesta

Los números del agente salen de eval/resultados_agente.respuestas.json (los
guarda evaluar_agente.py). Los del pipeline fijo se miden aquí corriendo
engine.ask() (1 llamada al LLM por pregunta, con top_k=6).

Uso (después de evaluar_agente.py):
    python eval/comparar_costo.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import evaluar_ragas as base

BASE_DIR = base.BASE_DIR
rag = base.rag


def main():
    ap = argparse.ArgumentParser(description="Compara el costo del pipeline fijo y del agente.")
    ap.add_argument("--preguntas", default=str(BASE_DIR / "eval" / "preguntas_eval.json"))
    ap.add_argument("--agente", default=str(BASE_DIR / "eval" / "resultados_agente.respuestas.json"))
    ap.add_argument("--manuales", default=str(BASE_DIR / "manuales"))
    ap.add_argument("--persist-dir", default=str(BASE_DIR / "eval" / "chroma_eval_agente"))
    ap.add_argument("--modelo", default=rag.DEFAULT_GROQ_MODEL)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--pausa", type=float, default=20.0)
    ap.add_argument("--salida", default=str(BASE_DIR / "eval" / "costo_comparacion.csv"))
    args = ap.parse_args()

    if not os.getenv("GROQ_API_KEY"):
        sys.exit("Falta GROQ_API_KEY en el entorno (.env).")
    ruta_agente = Path(args.agente)
    if not ruta_agente.exists():
        sys.exit(f"No existe {ruta_agente}. Corre primero: python eval/evaluar_agente.py")

    preguntas = base.cargar_preguntas(Path(args.preguntas))
    agente = {f["id"]: f for f in json.loads(ruta_agente.read_text(encoding="utf-8"))["filas"]}

    engine = rag.RagEngine(persist_dir=Path(args.persist_dir),
                           llm_factory=lambda modelo, _k: base._groq_paciente(modelo))
    if engine.load_existing() == 0:
        archivos = [p for p in Path(args.manuales).iterdir()
                    if p.suffix.lower() in rag.EXTENSIONES_SOPORTADAS]
        engine.index(archivos)

    filas = []
    for i, p in enumerate(preguntas):
        if p["id"] not in agente:
            continue
        if filas:
            time.sleep(args.pausa)
        print(f"Pipeline fijo — pregunta {i + 1}/{len(preguntas)} (id={p['id']})…", flush=True)
        t0 = time.perf_counter()
        r = engine.ask(p["pregunta"], historial=None, k=args.top_k, model=args.modelo)
        t = time.perf_counter() - t0
        a = agente[p["id"]]
        filas.append({
            "id": p["id"],
            "pipeline_llamadas": 1,
            "pipeline_tokens": r.uso.get("total_tokens", 0),
            "pipeline_tiempo_s": round(t, 2),
            "agente_llamadas": a["llamadas_llm"],
            "agente_tokens": a["tokens"],
            "agente_tiempo_s": a["tiempo_s"],
        })

    import pandas as pd

    df = pd.DataFrame(filas)
    Path(args.salida).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.salida, index=False)

    m = df.mean(numeric_only=True)
    tabla = "\n".join([
        "| Medida (promedio por pregunta) | Pipeline fijo (Avance 2) | Agente (Avance 3) |",
        "|---|---|---|",
        f"| Llamadas al LLM | {m['pipeline_llamadas']:.2f} | {m['agente_llamadas']:.2f} |",
        f"| Tokens | {m['pipeline_tokens']:.0f} | {m['agente_tokens']:.0f} |",
        f"| Tiempo de respuesta (s) | {m['pipeline_tiempo_s']:.1f} | {m['agente_tiempo_s']:.1f} |",
    ])
    md = Path(args.salida).with_suffix(".md")
    md.write_text(tabla + f"\n\nPreguntas usadas: {len(df)} (ids {', '.join(map(str, df['id']))})\n",
                  encoding="utf-8")
    print("\n" + tabla + f"\n\nPreguntas usadas: {len(df)}. Guardado en {args.salida} y {md}")


if __name__ == "__main__":
    main()
