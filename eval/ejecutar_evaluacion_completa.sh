#!/usr/bin/env bash
# ejecutar_evaluacion_completa.sh — corre la evaluación ANTES y DESPUÉS de la
# iteración de mejora (Avance 2: "modificar un parámetro del pipeline... y
# comparar antes/después"), y deja los dos CSV + gráficos listos para
# comparar en el PDF de evidencias.
#
# Iteración de mejora elegida: aumentar top_k de 6 a 10 fragmentos
# recuperados. Justificación: con top_k=6 algunas preguntas que requieren
# combinar información de dos secciones distintas del manual (ej. "LED rojo
# + garantía") pueden quedarse sin uno de los fragmentos relevantes, lo que
# baja context_recall. Subir top_k da más margen al retriever sin diluir
# demasiado el contexto (sigue siendo pequeño frente al límite del modelo).
#
# Requisitos: GROQ_API_KEY configurada (en .env o variable de entorno) y
# las dependencias de eval/requirements-eval.txt instaladas.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== Evaluación ANTES (baseline: chunk_size=500, chunk_overlap=50, top_k=6) =="
python eval/evaluar_ragas.py \
  --chunk-size 500 --chunk-overlap 50 --top-k 6 \
  --persist-dir eval/chroma_eval_before \
  --salida eval/resultados_before.csv

echo
echo "== Evaluación DESPUÉS (top_k=10, resto igual) =="
python eval/evaluar_ragas.py \
  --chunk-size 500 --chunk-overlap 50 --top-k 10 \
  --persist-dir eval/chroma_eval_after \
  --salida eval/resultados_after.csv

echo
echo "Listo. Compara eval/resultados_before.csv / .png con eval/resultados_after.csv / .png"
