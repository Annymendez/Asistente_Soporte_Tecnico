"""
evaluar_agente.py — Ragas sobre el AGENTE y comparación contra el pipeline fijo (Avance 3).

Corre las mismas preguntas de eval/preguntas_eval.json del Avance 2, pero ahora a
través del agente (agent.py). Los retrieved_contexts que se le dan a Ragas son
los fragmentos que devolvieron las tools del agente (base y, si las usó, web).
Calcula las mismas 4 métricas (faithfulness, answer_relevancy, context_precision,
context_recall) con el mismo juez, y arma la tabla Avance 2 vs agente.

Uso:
    python eval/evaluar_agente.py
    python eval/evaluar_agente.py --top-k 6 --pausa 30

Requiere GROQ_API_KEY (y TAVILY_API_KEY si quieres que el agente pueda usar la web).
Salidas:
    eval/resultados_agente.csv / .png        resultados por pregunta + promedios
    eval/resultados_agente.respuestas.json   respuestas y costo por pregunta (caché)
    eval/comparacion_ragas.md                tabla Avance 2 vs agente para el PDF
    eval/registro_agente.jsonl               registro de uso de tools por pregunta
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

from agent import AgenteSoporte  # noqa: E402

SIN_CONTEXTO = "(el agente no llamó ninguna tool: no recuperó contexto)"

# Nombres de las métricas en los CSV de Ragas -> nombre que se muestra
METRICAS = {
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevancy",
    "llm_context_precision_with_reference": "context_precision",
    "context_precision": "context_precision",
    "llm_context_recall": "context_recall",
    "context_recall": "context_recall",
}


def ejecutar_agente_sobre_preguntas(agente, preguntas, k, model, pausa, hechas=None, al_avanzar=None):
    filas = list(hechas or [])
    ya = {f["id"] for f in filas}
    primera = True
    for i, item in enumerate(preguntas, 1):
        if item["id"] in ya:
            continue
        if not primera and pausa > 0:
            time.sleep(pausa)   # reparte las preguntas para no chocar con el límite de tokens/min de Groq
        primera = False
        print(f"  Pregunta {i}/{len(preguntas)} (id={item['id']})…", flush=True)
        r = agente.responder(item["pregunta"], historial=None, k=k, model=model, sid=f"eval{item['id']}")
        filas.append({
            "id": item["id"],
            "pregunta": item["pregunta"],
            "respuesta": (json.dumps(r.respuesta, ensure_ascii=False)
                          if r.respuesta is not None else r.respuesta_cruda),
            "contextos": r.contextos() or [SIN_CONTEXTO],
            "ground_truth": item["ground_truth"],
            "en_corpus": item["en_corpus"],
            "tools": [t["nombre"] for t in r.tools_usadas],
            "llamadas_llm": r.llamadas_llm,
            "tokens": r.tokens.get("total_tokens", 0),
            "tiempo_s": round(r.tiempo_s, 2),
        })
        if al_avanzar:
            al_avanzar(filas)
    orden = {p["id"]: n for n, p in enumerate(preguntas)}
    return sorted(filas, key=lambda f: orden[f["id"]])


def promedios(df) -> dict[str, float]:
    out = {}
    for col, nombre in METRICAS.items():
        if col in df.columns and nombre not in out:
            out[nombre] = float(df[col].mean())
    return out


def tabla_comparativa(prom_agente: dict, csv_a2: Path, etiqueta_a2: str) -> str:
    import pandas as pd

    prom_a2 = promedios(pd.read_csv(csv_a2)) if csv_a2.exists() else {}
    lineas = [f"| Métrica | Avance 2 (pipeline fijo, {etiqueta_a2}) | Avance 3 (agente) | Diferencia |",
              "|---|---|---|---|"]
    for m in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        a2, a3 = prom_a2.get(m), prom_agente.get(m)
        dif = f"{a3 - a2:+.3f}" if (a2 is not None and a3 is not None) else "—"
        lineas.append(f"| {m} | {'—' if a2 is None else f'{a2:.3f}'} | "
                      f"{'—' if a3 is None else f'{a3:.3f}'} | {dif} |")
    return "\n".join(lineas)


def main():
    ap = argparse.ArgumentParser(description="Evalúa el agente con Ragas.")
    ap.add_argument("--preguntas", default=str(BASE_DIR / "eval" / "preguntas_eval.json"))
    ap.add_argument("--manuales", default=str(BASE_DIR / "manuales"))
    ap.add_argument("--top-k", type=int, default=6, help="fragmentos por búsqueda (tope duro: 6)")
    ap.add_argument("--modelo", default=rag.DEFAULT_GROQ_MODEL)
    ap.add_argument("--modelo-juez", default=None)
    ap.add_argument("--pausa", type=float, default=20.0)
    ap.add_argument("--persist-dir", default=str(BASE_DIR / "eval" / "chroma_eval_agente"))
    ap.add_argument("--salida", default=str(BASE_DIR / "eval" / "resultados_agente.csv"))
    ap.add_argument("--csv-avance2", default=str(BASE_DIR / "eval" / "resultados_before.csv"),
                    help="CSV del Avance 2 contra el que se compara (por defecto top_k=6)")
    ap.add_argument("--etiqueta-avance2", default="top_k=6")
    args = ap.parse_args()

    if not os.getenv("GROQ_API_KEY"):
        sys.exit("Falta GROQ_API_KEY en el entorno (.env).")
    if not os.getenv("TAVILY_API_KEY"):
        print("AVISO: no hay TAVILY_API_KEY; el agente no podrá usar la búsqueda web.\n")

    preguntas = base.cargar_preguntas(Path(args.preguntas))
    print(f"Preguntas de evaluación: {len(preguntas)}")

    engine = rag.RagEngine(persist_dir=Path(args.persist_dir))
    if engine.load_existing() == 0:
        archivos = [p for p in Path(args.manuales).iterdir()
                    if p.suffix.lower() in rag.EXTENSIONES_SOPORTADAS]
        print("Indexando manuales…")
        stats = engine.index(archivos, progress=lambda f, m: print(f"  [{f:.0%}] {m}"))
        print(f"Indexados: {stats.archivos} archivos, {stats.fragmentos} fragmentos.")
    agente = AgenteSoporte(
        engine,
        llm_factory=lambda modelo, _k: base._groq_paciente(modelo),
        log_path=BASE_DIR / "eval" / "registro_agente.jsonl",
    )

    salida_csv = Path(args.salida)
    cache = salida_csv.with_suffix(".respuestas.json")
    params = {"top_k": args.top_k, "modelo": args.modelo, "agente": True}

    def guardar(parciales):
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"params": params, "filas": parciales},
                                    ensure_ascii=False, indent=2), encoding="utf-8")

    hechas = []
    if cache.exists():
        guardado = json.loads(cache.read_text(encoding="utf-8"))
        if guardado.get("params") == params:
            ids = {p["id"] for p in preguntas}
            hechas = [f for f in guardado.get("filas", []) if f["id"] in ids]
    if len(hechas) == len(preguntas):
        filas = hechas
        print(f"Reutilizando las {len(filas)} respuestas guardadas en {cache} (bórralo para regenerarlas).")
    else:
        if hechas:
            print(f"Retomando: {len(hechas)} respuestas ya guardadas.")
        print(f"Ejecutando el AGENTE sobre {len(preguntas) - len(hechas)} preguntas (modelo={args.modelo})…")
        filas = ejecutar_agente_sobre_preguntas(agente, preguntas, args.top_k, args.modelo,
                                                args.pausa, hechas, guardar)
        guardar(filas)

    print("\nConstruyendo dataset de Ragas y evaluando (va de a una llamada, puede tardar 20-40 min)…")
    dataset = base.construir_dataset_ragas(filas)
    resultado = base.ejecutar_evaluacion_ragas(dataset, model=args.modelo_juez or args.modelo)

    df = resultado.to_pandas()
    df.insert(0, "id", [f["id"] for f in filas])
    df.insert(1, "en_corpus", [f["en_corpus"] for f in filas])
    df["tools"] = [", ".join(f["tools"]) or "ninguna" for f in filas]
    df["llamadas_llm"] = [f["llamadas_llm"] for f in filas]
    df["tokens"] = [f["tokens"] for f in filas]
    df["tiempo_s"] = [f["tiempo_s"] for f in filas]
    salida_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(salida_csv, index=False)

    prom = promedios(df)
    print("\n=== Promedios del agente ===")
    for m, v in prom.items():
        print(f"  {m}: {v:.3f}")
    print(f"\n  llamadas al LLM por pregunta (promedio): {df['llamadas_llm'].mean():.2f}")
    print(f"  tokens por pregunta (promedio): {df['tokens'].mean():.0f}")
    print(f"  tiempo por pregunta (promedio): {df['tiempo_s'].mean():.1f} s")
    sin_tool = sum(1 for f in filas if not f["tools"])
    print(f"  preguntas en las que NO llamó ninguna tool: {sin_tool}/{len(filas)}")

    tabla = tabla_comparativa(prom, Path(args.csv_avance2), args.etiqueta_avance2)
    md = BASE_DIR / "eval" / "comparacion_ragas.md"
    md.write_text(tabla + f"\n\nPreguntas evaluadas: {len(filas)}\n", encoding="utf-8")
    print("\n" + tabla + f"\n\n(guardada en {md})")

    base.graficar_resultados(df, salida_csv.with_suffix(".png"))


if __name__ == "__main__":
    main()
