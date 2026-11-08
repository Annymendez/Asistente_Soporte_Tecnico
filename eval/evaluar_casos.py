"""
evaluar_casos.py — Casos de comportamiento del agente (Avance 3, sección 3).

Corre los casos de eval/casos_comportamiento.json. El comportamiento ESPERADO
está escrito en ese archivo ANTES de ejecutar nada. Para cada caso registra qué
tools llamó el agente, con qué argumentos, cuántas llamadas hizo al LLM y si
cumplió lo esperado (verificación automática: revísala a mano antes de
entregar, sobre todo la columna "obtenido").

Uso:
    python eval/evaluar_casos.py
    python eval/evaluar_casos.py --pausa 30

Requiere GROQ_API_KEY y, para el caso de búsqueda web, TAVILY_API_KEY.
Salidas: eval/resultados_casos.csv y eval/resultados_casos.md (tabla lista
para pegar en la plantilla del PDF).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import evaluar_ragas as base  # reutiliza el parche de compatibilidad, .env y el LLM "paciente"

BASE_DIR = base.BASE_DIR
rag = base.rag

from agent import AgenteSoporte  # noqa: E402
from agent_tools import DOMINIOS_WEB_PERMITIDOS, _dominio_permitido  # noqa: E402


def verificar(caso: dict, r) -> list[str]:
    """Devuelve la lista de incumplimientos (vacía = cumple)."""
    fallas = []
    usadas = [t["nombre"] for t in r.tools_usadas]
    texto = (r.respuesta_cruda or "").lower()

    for t in caso.get("tools_esperadas", []):
        if t not in usadas:
            fallas.append(f"no llamó {t}")
    for t in caso.get("tools_prohibidas", []):
        if t in usadas:
            fallas.append(f"llamó {t} (prohibida)")
    for t, minimo in caso.get("min_llamadas_tool", {}).items():
        if usadas.count(t) < minimo:
            fallas.append(f"{t} se llamó {usadas.count(t)} vez/veces (mínimo {minimo})")

    orden = caso.get("orden_tools")
    if orden:
        indices = [usadas.index(t) for t in orden if t in usadas]
        if indices != sorted(indices):
            fallas.append("orden de tools incorrecto (primero la base)")

    for t, esperados in caso.get("args_esperados", {}).items():
        llamadas = [x for x in r.tools_usadas if x["nombre"] == t]
        if not llamadas:
            continue  # ya se reportó arriba
        for arg, valor in esperados.items():
            if not any(str(x["args"].get(arg)) == str(valor) for x in llamadas):
                fallas.append(f"{t}: argumento {arg} != {valor}")

    for alternativas in caso.get("respuesta_debe_contener", []):
        if not any(a.lower() in texto for a in alternativas):
            fallas.append(f"la respuesta no menciona {' / '.join(alternativas)}")
    for prohibido in caso.get("respuesta_no_debe_contener", []):
        if prohibido.lower() in texto:
            fallas.append(f"la respuesta contiene '{prohibido}'")
    if "max_chars_respuesta" in caso and len(r.respuesta_cruda or "") > caso["max_chars_respuesta"]:
        fallas.append(f"respuesta demasiado larga ({len(r.respuesta_cruda)} caracteres)")

    if caso.get("urls_web_permitidas"):
        urls = [w["url"] for w in r.resultados_web]
        if urls and not all(_dominio_permitido(u) for u in urls):
            fallas.append("hay URLs fuera de los sitios permitidos")
    if "requiere_humano" in caso:
        got = bool((r.respuesta or {}).get("requiere_soporte_humano"))
        if got != caso["requiere_humano"]:
            fallas.append(f"requiere_soporte_humano={got} (esperado {caso['requiere_humano']})")
    if r.respuesta is None:
        fallas.append("la respuesta no es un JSON válido")
    return fallas


def describir_obtenido(r) -> str:
    resp = r.respuesta or {}
    texto = (resp.get("diagnostico") or r.respuesta_cruda or "").strip().replace("\n", " ")
    return texto[:220] + ("…" if len(texto) > 220 else "")


def main():
    ap = argparse.ArgumentParser(description="Casos de comportamiento del agente.")
    ap.add_argument("--casos", default=str(BASE_DIR / "eval" / "casos_comportamiento.json"))
    ap.add_argument("--manuales", default=str(BASE_DIR / "manuales"))
    ap.add_argument("--persist-dir", default=str(BASE_DIR / "eval" / "chroma_eval_agente"))
    ap.add_argument("--modelo", default=rag.DEFAULT_GROQ_MODEL)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--pausa", type=float, default=20.0,
                    help="Segundos entre casos (límite de tokens por minuto de Groq)")
    ap.add_argument("--salida", default=str(BASE_DIR / "eval" / "resultados_casos.csv"))
    args = ap.parse_args()

    import os
    if not os.getenv("GROQ_API_KEY"):
        sys.exit("Falta GROQ_API_KEY en el entorno (.env).")
    if not os.getenv("TAVILY_API_KEY"):
        print("AVISO: no hay TAVILY_API_KEY; el caso de búsqueda web no podrá cumplirse.\n")

    casos = json.loads(Path(args.casos).read_text(encoding="utf-8"))

    engine = rag.RagEngine(persist_dir=Path(args.persist_dir))
    if engine.load_existing() == 0:
        archivos = [p for p in Path(args.manuales).iterdir()
                    if p.suffix.lower() in rag.EXTENSIONES_SOPORTADAS]
        print("Indexando manuales…")
        engine.index(archivos)
    agente = AgenteSoporte(
        engine,
        llm_factory=lambda modelo, _k: base._groq_paciente(modelo),
        log_path=BASE_DIR / "eval" / "registro_casos.jsonl",
    )
    print(f"Sitios web permitidos: {', '.join(DOMINIOS_WEB_PERMITIDOS)}\n")

    filas = []
    for i, caso in enumerate(casos):
        if i:
            time.sleep(args.pausa)
        print(f"Caso {caso['id']} — {caso['tipo']}: {caso['pregunta']}")
        r = agente.responder(caso["pregunta"], k=args.top_k, model=args.modelo, sid=f"caso{caso['id']}")
        fallas = verificar(caso, r)
        fila = {
            "#": caso["id"],
            "tipo": caso["tipo"],
            "pregunta": caso["pregunta"],
            "esperado": caso["esperado"],
            "obtenido": describir_obtenido(r),
            "tools": "; ".join(f"{t['nombre']}({json.dumps(t['args'], ensure_ascii=False)})"
                               for t in r.tools_usadas) or "ninguna",
            "llamadas_llm": r.llamadas_llm,
            "tokens": r.tokens.get("total_tokens", 0),
            "tiempo_s": round(r.tiempo_s, 1),
            "cumple": "Sí" if not fallas else "No",
            "observaciones": "; ".join(fallas),
        }
        filas.append(fila)
        print(f"   tools: {fila['tools']}\n   llamadas LLM: {fila['llamadas_llm']} · "
              f"cumple: {fila['cumple']} {fila['observaciones']}\n")

    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    with open(salida, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)

    md = ["| # | Tipo de caso | Pregunta | Comportamiento esperado | Comportamiento obtenido | "
          "Tools usadas | Llam. LLM | ¿Cumple? |", "|---|---|---|---|---|---|---|---|"]
    for x in filas:
        celdas = [x["#"], x["tipo"], x["pregunta"], x["esperado"], x["obtenido"], x["tools"],
                  x["llamadas_llm"], x["cumple"] + (f" ({x['observaciones']})" if x["observaciones"] else "")]
        md.append("| " + " | ".join(str(c).replace("|", "/") for c in celdas) + " |")
    salida.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")

    cumplen = sum(f["cumple"] == "Sí" for f in filas)
    print(f"Cumplen {cumplen}/{len(filas)} casos. CSV: {salida}  ·  Tabla: {salida.with_suffix('.md')}")


if __name__ == "__main__":
    main()
