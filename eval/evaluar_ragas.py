"""
evaluar_ragas.py — Evaluación del pipeline RAG con Ragas (Avance 2).

Ejecuta el conjunto de preguntas de eval/preguntas_eval.json contra el
RagEngine real (ingesta -> chunking -> embeddings -> Chroma -> retrieval ->
generación), y calcula 4 métricas de Ragas:

    faithfulness        -> ¿la respuesta se sostiene en los fragmentos recuperados?
    answer_relevancy     -> ¿la respuesta es relevante respecto a la pregunta?
    context_precision    -> ¿los fragmentos recuperados son relevantes (poco ruido)?
    context_recall        -> ¿se recuperó todo lo necesario para responder (contra el ground_truth)?

Uso:
    python eval/evaluar_ragas.py
    python eval/evaluar_ragas.py --chunk-size 800 --chunk-overlap 100 --top-k 4 --salida eval/resultados_after.csv

Requiere GROQ_API_KEY en el entorno (.env) tanto para el RagEngine (genera las
respuestas) como para el LLM-juez que usa Ragas para calificar faithfulness,
answer_relevancy, context_precision y context_recall.

NOTA DE COMPATIBILIDAD: la versión de ragas instalada (0.4.x) intenta importar
'langchain_community.chat_models.vertexai', un submódulo opcional que las
versiones recientes de langchain-community ya no incluyen. Como no se usa
Vertex AI en este proyecto, se inserta un módulo "falso" en sys.modules ANTES
de importar ragas, para que ese import opcional no rompa el script.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

# ── Parche de compatibilidad (ver nota arriba) ─────────────────────────────
_vertexai_falso = types.ModuleType("langchain_community.chat_models.vertexai")


class _ChatVertexAIFalso:  # pragma: no cover - nunca se instancia, solo evita el ImportError
    pass


_vertexai_falso.ChatVertexAI = _ChatVertexAIFalso
sys.modules.setdefault("langchain_community.chat_models.vertexai", _vertexai_falso)

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import rag_core as rag  # noqa: E402


def cargar_preguntas(ruta: Path) -> list[dict]:
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def ejecutar_rag_sobre_preguntas(
    engine: rag.RagEngine, preguntas: list[dict], k: int, model: str
) -> list[dict]:
    """Corre cada pregunta contra el RagEngine (sin historial: cada pregunta de
    evaluación se trata como una conversación nueva e independiente) y arma
    las filas que espera Ragas: pregunta, respuesta generada, contextos
    recuperados (texto plano de cada fragmento) y ground_truth de referencia."""
    filas = []
    for item in preguntas:
        resultado = engine.ask(item["pregunta"], historial=None, k=k, model=model)
        respuesta_texto = (
            json.dumps(resultado.respuesta, ensure_ascii=False)
            if resultado.respuesta is not None
            else resultado.respuesta_cruda
        )
        filas.append({
            "id": item["id"],
            "pregunta": item["pregunta"],
            "respuesta": respuesta_texto,
            "contextos": [d.page_content for d in resultado.fragmentos],
            "ground_truth": item["ground_truth"],
            "en_corpus": item["en_corpus"],
        })
    return filas


def construir_dataset_ragas(filas: list[dict]):
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample

    muestras = [
        SingleTurnSample(
            user_input=f["pregunta"],
            response=f["respuesta"],
            retrieved_contexts=f["contextos"],
            reference=f["ground_truth"],
        )
        for f in filas
    ]
    return EvaluationDataset(samples=muestras)


def ejecutar_evaluacion_ragas(dataset, model: str):
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        AnswerRelevancy,
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
    )

    llm_juez = LangchainLLMWrapper(ChatGroq(model=model, temperature=0.0))
    embeddings_juez = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=rag.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    )

    metricas = [
        Faithfulness(llm=llm_juez),
        AnswerRelevancy(llm=llm_juez, embeddings=embeddings_juez),
        LLMContextPrecisionWithReference(llm=llm_juez),
        LLMContextRecall(llm=llm_juez),
    ]

    return evaluate(dataset=dataset, metrics=metricas)


def graficar_resultados(df, salida_png: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columnas_metricas = [
        c for c in ["faithfulness", "answer_relevancy", "llm_context_precision_with_reference",
                     "llm_context_recall", "context_precision", "context_recall"]
        if c in df.columns
    ]
    promedios = df[columnas_metricas].mean()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    barras = ax.bar(promedios.index, promedios.values, color="#0a2f5c")
    ax.bar_label(barras, fmt="%.2f")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Puntaje promedio (0-1)")
    ax.set_title("Resultados de evaluación Ragas — Asistente de Soporte Técnico")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(salida_png, dpi=150)
    print(f"Gráfico guardado en: {salida_png}")


def main():
    parser = argparse.ArgumentParser(description="Evalúa el pipeline RAG con Ragas.")
    parser.add_argument("--preguntas", default=str(BASE_DIR / "eval" / "preguntas_eval.json"))
    parser.add_argument("--manuales", default=str(BASE_DIR / "manuales"))
    parser.add_argument("--chunk-size", type=int, default=rag.CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=rag.CHUNK_OVERLAP)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--modelo", default=rag.DEFAULT_GROQ_MODEL)
    parser.add_argument("--embedding-model", default=rag.EMBEDDING_MODEL)
    parser.add_argument("--salida", default=str(BASE_DIR / "eval" / "resultados_eval.csv"))
    parser.add_argument("--persist-dir", default=str(BASE_DIR / "eval" / "chroma_eval"))
    args = parser.parse_args()

    if not os.getenv("GROQ_API_KEY"):
        sys.exit(
            "Falta GROQ_API_KEY en el entorno (.env). Es necesaria tanto para generar "
            "las respuestas del asistente como para el LLM-juez de Ragas."
        )

    # Permite variar chunk_size/chunk_overlap/embedding_model para la iteración de mejora
    # (Avance 2: "modificar un parámetro del pipeline... y comparar antes/después").
    rag.CHUNK_SIZE = args.chunk_size
    rag.CHUNK_OVERLAP = args.chunk_overlap
    rag.EMBEDDING_MODEL = args.embedding_model

    preguntas = cargar_preguntas(Path(args.preguntas))
    print(f"Preguntas de evaluación cargadas: {len(preguntas)} "
          f"({sum(p['en_corpus'] for p in preguntas)} dentro del corpus, "
          f"{sum(not p['en_corpus'] for p in preguntas)} fuera del corpus)")

    print(f"\nIndexando manuales desde {args.manuales} "
          f"(chunk_size={args.chunk_size}, chunk_overlap={args.chunk_overlap}, "
          f"embedding_model={args.embedding_model})…")
    engine = rag.RagEngine(persist_dir=Path(args.persist_dir))
    archivos = [p for p in Path(args.manuales).iterdir()
                if p.suffix.lower() in rag.EXTENSIONES_SOPORTADAS]
    stats = engine.index(archivos, progress=lambda f, m: print(f"  [{f:.0%}] {m}"))
    print(f"Indexados: {stats.archivos} archivos, {stats.fragmentos} fragmentos.")

    print(f"\nEjecutando el RAG sobre las {len(preguntas)} preguntas (top_k={args.top_k})…")
    filas = ejecutar_rag_sobre_preguntas(engine, preguntas, k=args.top_k, model=args.modelo)

    print("\nConstruyendo dataset de Ragas y evaluando "
          "(faithfulness, answer_relevancy, context_precision, context_recall)…")
    dataset = construir_dataset_ragas(filas)
    resultado = ejecutar_evaluacion_ragas(dataset, model=args.modelo)

    df = resultado.to_pandas()
    # Adjuntamos el id y si la pregunta era de corpus o no, para el análisis
    df.insert(0, "id", [f["id"] for f in filas])
    df.insert(1, "en_corpus", [f["en_corpus"] for f in filas])

    salida_csv = Path(args.salida)
    salida_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(salida_csv, index=False)
    print(f"\nResultados detallados guardados en: {salida_csv}")

    columnas_metricas = [c for c in df.columns if c not in (
        "id", "en_corpus", "user_input", "response", "retrieved_contexts", "reference"
    )]
    promedios = df[columnas_metricas].mean().sort_values()
    print("\n=== Promedios ===")
    for metrica, valor in promedios.items():
        print(f"  {metrica}: {valor:.3f}")

    peor = promedios.index[0]
    print(f"\nMétrica más baja: {peor} ({promedios.iloc[0]:.3f})")

    graficar_resultados(df, salida_csv.with_suffix(".png"))


if __name__ == "__main__":
    main()
