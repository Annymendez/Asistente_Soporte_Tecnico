import os
import time
from dotenv import load_dotenv

load_dotenv()

from langchain_cohere import CohereEmbeddings

key = os.getenv("COHERE_API_KEY")
print("COHERE_API_KEY presente:", bool(key), "| longitud:", len(key) if key else 0)

print("Creando el objeto CohereEmbeddings...")
emb = CohereEmbeddings(model="embed-multilingual-v3.0", cohere_api_key=key)
print("Objeto creado. Llamando a embed_query (esto SI llama a la API de Cohere por internet)...")

t0 = time.time()
try:
    vector = emb.embed_query("hola, esto es una prueba")
    print(f"OK: respuesta recibida en {time.time() - t0:.2f} segundos.")
    print("Longitud del vector:", len(vector))
except Exception as e:
    print(f"ERROR despues de {time.time() - t0:.2f} segundos:")
    print(type(e).__name__, "-", e)