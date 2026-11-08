"""
agent.py — Agente de soporte técnico con tool calling (Avance 3).

Convierte el RAG del Avance 2 en un agente. En el Avance 2 el programa SIEMPRE
buscaba y pegaba los fragmentos en el prompt (pipeline fijo, 1 llamada al LLM).
Aquí el LLM decide si llama una tool y cuál; el programa la ejecuta y le
devuelve el resultado, en un bucle manual:

    pregunta -> LLM -+-> (sin tool_calls) -> respuesta final
                     |
                     +-> tool_calls -> el programa ejecuta cada tool
                                       -> resultado al LLM -> (otra vuelta)

Garantías que impone el CÓDIGO (no el prompt):
  * MAX_ITERACIONES: tope de vueltas del bucle por pregunta. Si se agota, se
    fuerza una respuesta final sin tools.
  * MAX_LLAMADAS_TOOLS: tope de ejecuciones de tools por pregunta.
  * Cada tool tiene sus propios límites (agent_tools.py).
  * Registro: cada pregunta deja una línea JSON en logs/uso_tools.jsonl con las
    tools llamadas, sus argumentos, las llamadas al LLM, los tokens y el tiempo.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from langchain_core.messages import HumanMessage, ToolMessage

import rag_core as rag
from agent_prompt import construir_mensajes_agente
from agent_tools import crear_tools

BASE_DIR = Path(__file__).resolve().parent
LOG_PATH = Path(os.getenv("AGENT_LOG_PATH", BASE_DIR / "logs" / "uso_tools.jsonl"))

MAX_ITERACIONES = 5        # vueltas máximas del bucle (llamadas al LLM con tools disponibles)
MAX_LLAMADAS_TOOLS = 6     # ejecuciones de tools máximas por pregunta
MAX_CHARS_ARGS_LOG = 300   # recorte de los argumentos en el registro

AVISO_LIMITE = (
    "Has alcanzado el límite de búsquedas permitido. No llames más herramientas: "
    "responde ahora con lo que ya obtuviste, en el formato JSON indicado."
)


@dataclass
class ResultadoAgente:
    pregunta: str
    respuesta: dict | None            # JSON parseado (o None si no se pudo)
    respuesta_cruda: str
    error_parseo: str | None
    tools_usadas: list[dict]          # [{"nombre", "args", "resumen", "duracion_ms"}]
    fragmentos: list[dict]            # fragmentos de la base devueltos por las tools
    resultados_web: list[dict]        # resultados web devueltos por las tools
    llamadas_llm: int
    tokens: dict = field(default_factory=dict)   # input_tokens / output_tokens / total_tokens
    tiempo_s: float = 0.0
    limite_alcanzado: bool = False

    def fuentes_base(self) -> str:
        """'documento (pág. 1, 2) · otro.txt' con lo que devolvieron las tools."""
        por_archivo: dict[str, set[str]] = {}
        for f in self.fragmentos:
            por_archivo.setdefault(f["fuente"], set()).add(f["pagina"])

        def orden(p: str):
            return (0, int(p)) if p.isdigit() else (1, 0)

        partes = []
        for nombre, pags in por_archivo.items():
            con_pag = sorted((p for p in pags if p != "-"), key=orden)
            partes.append(f"{nombre} (pág. {', '.join(con_pag)})" if con_pag else nombre)
        return " · ".join(partes)

    def contextos(self) -> list[str]:
        """Textos que devolvieron las tools (lo que Ragas toma como retrieved_contexts)."""
        textos = [f["texto"] for f in self.fragmentos]
        textos += [r["extracto"] for r in self.resultados_web]
        return textos

    def a_dict(self) -> dict:
        return {
            "tools_usadas": self.tools_usadas,
            "llamadas_llm": self.llamadas_llm,
            "tokens": self.tokens,
            "tiempo_s": round(self.tiempo_s, 2),
            "limite_alcanzado": self.limite_alcanzado,
        }


class AgenteSoporte:
    def __init__(self, engine: rag.RagEngine, llm_factory: Optional[Callable] = None,
                 log_path: Path | str | None = None):
        self.engine = engine
        self._llm_factory = llm_factory      # útil para pruebas sin red y para la evaluación
        self.log_path = Path(log_path) if log_path else LOG_PATH
        self._llm_cache: tuple | None = None

    # -- LLM -----------------------------------------------------------------
    def _crear_llm(self, model: str, api_key: Optional[str]):
        key = api_key or os.getenv("GROQ_API_KEY")
        if self._llm_factory:
            return self._llm_factory(model, key)
        if not key:
            raise ValueError("Falta la API key de Groq. Pégala en Ajustes o defínela "
                             "como GROQ_API_KEY.")
        if self._llm_cache is None or self._llm_cache[0] != (model, key):
            from langchain_groq import ChatGroq

            self._llm_cache = ((model, key),
                               ChatGroq(model=model, temperature=0.0, api_key=key,
                                        timeout=60, max_retries=1))
        return self._llm_cache[1]

    @staticmethod
    def _invocar(llm, mensajes):
        """Groq a veces rechaza una llamada a tool mal formada (tool_use_failed);
        se reintenta una vez antes de rendirse."""
        try:
            return llm.invoke(mensajes)
        except Exception as e:
            if "tool_use_failed" in str(e) or "Failed to call a function" in str(e):
                return llm.invoke(mensajes)
            raise

    # -- bucle del agente ----------------------------------------------------
    def responder(self, pregunta: str, historial: list[dict] | None = None, k: int = 6,
                  model: str = rag.DEFAULT_GROQ_MODEL, api_key: Optional[str] = None,
                  sid: str = "") -> ResultadoAgente:
        if not self.engine.ready:
            raise RuntimeError("Primero indexa los manuales (o carga un índice existente).")

        t0 = time.perf_counter()
        llm = self._crear_llm(model, api_key)
        tools = crear_tools(self.engine, k=k)
        mapa = {t.name: t for t in tools}
        llm_con_tools = llm.bind_tools(tools)

        mensajes = construir_mensajes_agente(pregunta, historial, rag.MAX_TURNOS_HISTORIAL)
        tools_usadas: list[dict] = []
        fragmentos: list[dict] = []
        web: list[dict] = []
        llamadas_llm = 0
        tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        limite = False

        def contar(ai):
            nonlocal llamadas_llm
            llamadas_llm += 1
            uso = getattr(ai, "usage_metadata", None) or {}
            for clave in tokens:
                tokens[clave] += int(uso.get(clave, 0) or 0)

        ai = None
        for _ in range(MAX_ITERACIONES):
            ai = self._invocar(llm_con_tools, mensajes)
            contar(ai)
            mensajes.append(ai)
            if not ai.tool_calls:
                break

            for tc in ai.tool_calls:
                nombre, args = tc["name"], tc.get("args") or {}
                if nombre not in mapa:
                    mensajes.append(ToolMessage(tool_call_id=tc["id"], name=nombre,
                                                content=f"Herramienta desconocida: {nombre}."))
                    continue
                if nombre == "buscar_en_web" and not any(t["nombre"] == "buscar_en_base"
                                                         for t in tools_usadas):
                    # Política "primero la base" impuesta por el CÓDIGO, no solo por el prompt.
                    mensajes.append(ToolMessage(
                        tool_call_id=tc["id"], name=nombre,
                        content="Bloqueado: primero debes consultar buscar_en_base. "
                                "Llámala y solo usa la web si la base no contiene la respuesta."))
                    continue
                if len(tools_usadas) >= MAX_LLAMADAS_TOOLS:
                    mensajes.append(ToolMessage(tool_call_id=tc["id"], name=nombre,
                                                content="Límite de llamadas a herramientas alcanzado."))
                    continue

                ti = time.perf_counter()
                try:
                    salida = mapa[nombre].invoke(tc)       # ToolMessage con content + artifact
                except Exception as e:
                    salida = ToolMessage(tool_call_id=tc["id"], name=nombre,
                                         content=f"Error al ejecutar la herramienta: {type(e).__name__}")
                mensajes.append(salida)

                art = getattr(salida, "artifact", None) or {}
                fragmentos.extend(art.get("fragmentos", []))
                web.extend(art.get("resultados", []))
                tools_usadas.append({
                    "nombre": nombre,
                    "args": {k_: (v[:MAX_CHARS_ARGS_LOG] if isinstance(v, str) else v)
                             for k_, v in args.items()},
                    "resumen": _resumen(nombre, art),
                    "duracion_ms": int((time.perf_counter() - ti) * 1000),
                })
        else:
            # Se agotaron las vueltas con el modelo todavía pidiendo tools:
            # se fuerza una respuesta final SIN tools.
            limite = True
            mensajes.append(HumanMessage(content=AVISO_LIMITE))
            ai = self._invocar(llm, mensajes)
            contar(ai)

        respuesta_cruda = ai.content if isinstance(ai.content, str) else str(ai.content)
        respuesta, error = rag.parsear_json_respuesta(respuesta_cruda)

        resultado = ResultadoAgente(
            pregunta=pregunta, respuesta=respuesta, respuesta_cruda=respuesta_cruda,
            error_parseo=error, tools_usadas=tools_usadas, fragmentos=fragmentos,
            resultados_web=web, llamadas_llm=llamadas_llm, tokens=tokens,
            tiempo_s=time.perf_counter() - t0, limite_alcanzado=limite,
        )
        self._registrar(resultado, sid)
        return resultado

    # -- registro de uso de tools --------------------------------------------
    def _registrar(self, r: ResultadoAgente, sid: str) -> None:
        linea = {
            "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sesion": sid[:8],
            "pregunta": r.pregunta[:300],
            "tools": [{"nombre": t["nombre"], "args": t["args"], "resumen": t["resumen"]}
                      for t in r.tools_usadas],
            "llamadas_llm": r.llamadas_llm,
            "tokens": r.tokens,
            "tiempo_s": round(r.tiempo_s, 2),
            "limite_alcanzado": r.limite_alcanzado,
        }
        texto = json.dumps(linea, ensure_ascii=False)
        print(f"[AGENTE] {texto}", flush=True)       # también visible en los logs de Render
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(texto + "\n")
        except OSError:
            pass  # el registro nunca debe tumbar una respuesta


def _resumen(nombre: str, art: dict) -> str:
    if nombre == "buscar_en_base":
        return f"{len(art.get('fragmentos', []))} fragmentos"
    if nombre == "buscar_en_web":
        return f"{len(art.get('resultados', []))} resultados web"
    if nombre == "calcular_garantia":
        if "error" in art:
            return "error: " + art["error"][:80]
        return f"{'vigente' if art.get('vigente') else 'vencida'}, vence {art.get('vence')}"
    return ""