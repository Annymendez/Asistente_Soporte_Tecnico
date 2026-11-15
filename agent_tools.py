"""
agent_tools.py — Tools del agente de soporte técnico (Avance 3).

El LLM decide cuándo llamar cada tool; el programa la ejecuta y le devuelve el
resultado. Hay tres tools:

    buscar_en_base      (búsqueda)  recuperación del Avance 2 convertida en tool
    calcular_garantia   (tarea)     regla/cálculo exacto que el LLM no debe hacer "de memoria"
    buscar_en_web       (web)       Tavily, solo sitios permitidos, política "primero la base"

Todos los LÍMITES viven aquí, en el código (constantes de abajo), no en el
prompt: el LLM no puede cambiarlos aunque se lo pidan.

Cada tool devuelve (contenido, artefacto):
    - contenido: el texto que ve el LLM.
    - artefacto: datos estructurados (documento, página, URL...) que usan la
      interfaz, el registro de uso y la evaluación con Ragas. El LLM no los ve.
"""
from __future__ import annotations

import calendar
import os
from datetime import date, datetime
from urllib.parse import urlparse

from langchain_core.tools import tool

# ── Límites (escritos en el código, no en el prompt) ─────────────────────────
MAX_FRAGMENTOS_POR_LLAMADA = 6     # fragmentos de la base por cada llamada a buscar_en_base
MAX_CHARS_FRAGMENTO = 500          # longitud máxima de cada fragmento devuelto
MAX_CHARS_CONSULTA = 300           # longitud máxima de la consulta (a la base o a la web)
MAX_RESULTADOS_WEB = 3             # resultados que devuelve buscar_en_web
MAX_CHARS_EXTRACTO_WEB = 300       # longitud máxima del extracto de cada resultado web
TIMEOUT_WEB_SEGUNDOS = 15

# Plazo de la garantía según manuales/politica_garantia_rt900.txt
MESES_GARANTIA = 12

# Lista de sitios permitidos para la búsqueda web. Está fijada aquí: el LLM no
# puede cambiarla. Son organismos de estándares y de seguridad, y los centros
# de soporte de los sistemas operativos con los que se conecta un router.
DOMINIOS_WEB_PERMITIDOS = (
    "wi-fi.org",               # Wi-Fi Alliance: certificación y definiciones WiFi/WPA
    "ieee.org",                # IEEE: estándares 802.11
    "ietf.org",                # IETF: estándares de internet (DNS, DHCP, IP)
    "nist.gov",                # NIST: guías de seguridad
    "cisa.gov",                # CISA: recomendaciones de seguridad para routers y redes
    "support.microsoft.com",   # soporte oficial de Windows
    "support.apple.com",       # soporte oficial de Apple
)

TAVILY_URL = "https://api.tavily.com/search"


# ── Utilidades ────────────────────────────────────────────────────────────────
def _recortar(texto: str, maximo: int) -> str:
    texto = " ".join((texto or "").split())
    return texto if len(texto) <= maximo else texto[: maximo - 1].rstrip() + "…"


def _a_fecha(texto: str) -> date:
    """Acepta AAAA-MM-DD (preferido), DD/MM/AAAA y DD-MM-AAAA."""
    t = (texto or "").strip()
    for formato in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(t, formato).date()
        except ValueError:
            continue
    raise ValueError(f"'{texto}' no es una fecha válida (usa AAAA-MM-DD).")


def _sumar_meses(f: date, meses: int) -> date:
    """Suma meses de calendario; si el día no existe en el mes destino
    (p. ej. 31 de enero + 1 mes) usa el último día de ese mes."""
    indice = f.month - 1 + meses
    anio, mes = f.year + indice // 12, indice % 12 + 1
    return date(anio, mes, min(f.day, calendar.monthrange(anio, mes)[1]))


def _dominio_permitido(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in DOMINIOS_WEB_PERMITIDOS)


# ── Fábrica de tools ──────────────────────────────────────────────────────────
def crear_tools(engine, k: int = MAX_FRAGMENTOS_POR_LLAMADA) -> list:
    """Crea las tools ligadas a un RagEngine ya indexado.

    k: fragmentos por búsqueda; nunca pasa de MAX_FRAGMENTOS_POR_LLAMADA.
    """
    k_efectivo = max(1, min(int(k), MAX_FRAGMENTOS_POR_LLAMADA))

    @tool("buscar_en_base", response_format="content_and_artifact")
    def buscar_en_base(consulta: str):
        """Busca en los manuales del router RT-900 (manual técnico, guía rápida,
        guía de usuario y política de garantía) los fragmentos más relevantes.

        Úsala SIEMPRE que la pregunta trate del producto: instalación, LEDs,
        configuración, WiFi, reinicio, fallas, seguridad o garantía. NO la uses
        para saludos ni conversación casual. Si la pregunta tiene varios temas,
        llámala una vez por tema con una consulta distinta y específica.

        Args:
            consulta: lo que se quiere encontrar, en español y con palabras
                clave (por ejemplo "LED rojo fijo significado").
        """
        consulta = (consulta or "").strip()[:MAX_CHARS_CONSULTA]
        vacio = {"origen": "base", "consulta": consulta, "fragmentos": []}
        if not consulta:
            return "Consulta vacía: indica qué quieres buscar.", vacio
        if not engine.ready:
            return "La base de conocimientos todavía no está lista.", vacio

        docs = engine.buscar(consulta, k=k_efectivo)[:MAX_FRAGMENTOS_POR_LLAMADA]
        fragmentos = [
            {
                "fuente": os.path.basename(str(d.metadata.get("source", "desconocido"))),
                "pagina": _pagina(d),
                "texto": _recortar(d.page_content, MAX_CHARS_FRAGMENTO),
            }
            for d in docs
        ]
        if not fragmentos:
            return "No se encontraron fragmentos relevantes en la base.", vacio

        contenido = "\n\n".join(
            f"[Fuente: {f['fuente']} — Pág. {f['pagina']}]\n{f['texto']}" for f in fragmentos
        )
        return contenido, {"origen": "base", "consulta": consulta, "fragmentos": fragmentos}

    @tool("calcular_garantia", response_format="content_and_artifact")
    def calcular_garantia(fecha_compra: str, fecha_consulta: str = ""):
        """Calcula si la garantía del router RT-900 está vigente y cuándo vence.
        La garantía dura 12 meses desde la fecha de compra.

        Úsala cuando el usuario diga CUÁNDO compró el equipo y pregunte si sigue
        en garantía, cuándo vence o cuántos días le quedan. No hagas ese cálculo
        de memoria. Solo calcula el plazo: las exclusiones de la garantía
        (líquidos, descargas, carcasa abierta) se consultan con buscar_en_base.

        Args:
            fecha_compra: fecha de compra en formato AAAA-MM-DD (convierte tú la
                fecha que diga el usuario, p. ej. "15 de marzo de 2026" -> 2026-03-15).
            fecha_consulta: fecha contra la que se compara, AAAA-MM-DD. Déjala
                vacía para usar la fecha de hoy.
        """
        try:
            compra = _a_fecha(fecha_compra)
            consulta = _a_fecha(fecha_consulta) if (fecha_consulta or "").strip() else date.today()
        except ValueError as e:
            return f"Error: {e}", {"origen": "calculo", "error": str(e)}
        if compra > consulta:
            msg = (f"Error: la fecha de compra ({compra.isoformat()}) es posterior a la "
                   f"fecha de consulta ({consulta.isoformat()}).")
            return msg, {"origen": "calculo", "error": msg}

        vence = _sumar_meses(compra, MESES_GARANTIA)
        dias = (vence - consulta).days
        vigente = dias >= 0
        estado = (f"VIGENTE (quedan {dias} días)" if vigente
                  else f"VENCIDA (venció hace {-dias} días)")
        contenido = (
            f"Fecha de compra: {compra.isoformat()}\n"
            f"Plazo de garantía: {MESES_GARANTIA} meses -> vence el {vence.isoformat()}\n"
            f"Fecha de consulta: {consulta.isoformat()}\n"
            f"Estado: {estado}\n"
            "Nota: solo se calculó el plazo; no se evaluaron las exclusiones de la garantía."
        )
        return contenido, {
            "origen": "calculo", "fecha_compra": compra.isoformat(), "vence": vence.isoformat(),
            "fecha_consulta": consulta.isoformat(), "vigente": vigente, "dias": dias,
        }

    @tool("buscar_en_web", response_format="content_and_artifact")
    def buscar_en_web(consulta: str):
        """Busca información PÚBLICA y ACTUAL en la web, solo en sitios oficiales
        permitidos (Wi-Fi Alliance, IEEE, IETF, NIST, CISA, soporte de Microsoft
        y de Apple).

        Política "primero la base": úsala SOLO si buscar_en_base ya se consultó y
        no contiene la respuesta, y la pregunta trata de información pública o
        actual (un estándar, una definición general, una versión reciente).
        NUNCA la uses para datos o procedimientos del RT-900, ni para saludos.
        La consulta sale a un servicio externo: no incluyas texto copiado de los
        manuales ni datos personales del usuario.

        Args:
            consulta: pregunta corta y general (por ejemplo "qué es WPA3").
        """
        consulta = (consulta or "").strip()[:MAX_CHARS_CONSULTA]
        vacio = {"origen": "web", "consulta": consulta, "resultados": []}
        if not consulta:
            return "Consulta vacía: indica qué quieres buscar.", vacio
        clave = os.getenv("TAVILY_API_KEY", "").strip()
        if not clave:
            return ("Búsqueda web no disponible: falta la variable de entorno "
                    "TAVILY_API_KEY."), vacio

        try:
            import requests

            resp = requests.post(
                TAVILY_URL,
                headers={"Authorization": f"Bearer {clave}"},
                json={
                    "query": consulta,
                    "include_domains": list(DOMINIOS_WEB_PERMITIDOS),
                    "max_results": MAX_RESULTADOS_WEB,
                    "search_depth": "basic",
                    "include_answer": False,
                },
                timeout=TIMEOUT_WEB_SEGUNDOS,
            )
            resp.raise_for_status()
            crudos = resp.json().get("results", [])
        except Exception as e:  # red caída, clave inválida, límite de uso...
            return f"La búsqueda web falló ({type(e).__name__}).", vacio

        # Defensa extra: aunque el servicio devuelva otro dominio, se descarta.
        resultados = [
            {
                "origen": "web",
                "titulo": _recortar(r.get("title", ""), 120),
                "url": r.get("url", ""),
                "extracto": _recortar(r.get("content", ""), MAX_CHARS_EXTRACTO_WEB),
            }
            for r in crudos
            if _dominio_permitido(r.get("url", ""))
        ][:MAX_RESULTADOS_WEB]
        if not resultados:
            return "No hubo resultados en los sitios permitidos.", vacio

        contenido = "\n\n".join(
            f"[web: {r['url']}]\nTítulo: {r['titulo']}\nExtracto: {r['extracto']}" for r in resultados
        )
        return contenido, {"origen": "web", "consulta": consulta, "resultados": resultados}

    return [buscar_en_base, calcular_garantia, buscar_en_web]


def _pagina(doc) -> str:
    """PyPDFLoader numera desde 0; para DOCX/TXT/HTML no hay página: '-'."""
    p = doc.metadata.get("page")
    return str(p + 1) if isinstance(p, int) else "-"
