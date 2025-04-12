import os
import json
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv


class DataExtractor:
    def __init__(self, source: str = None, chunksize: int = 100000):
        """
        Inicializa el extractor.
        Parámetro:
            source: Ruta al archivo local de datos (CSV o JSON). En esta entrega, tras
                    extraer los datos vía API, se utilizará este archivo para cargar los datos.
            chunksize: Tamaño de cada chunk para la lectura en caso de archivos grandes.
        """
        self.source = source
        self.data = None
        self.chunksize = chunksize
        load_dotenv()  # Carga las variables de entorno (por ejemplo, RAPIDAPI_KEY)

    def load_data_api(self, query: str, max_results: int = 100,
                      output_file: str = "tweets_from_api.csv") -> pd.DataFrame:
        """
        Conecta con la API de Twitter a través de RapidAPI para extraer tweets en tiempo real,
        y almacena los resultados en un archivo local (formato CSV).

        Parámetros:
            query: Palabra clave o hashtag a buscar.
            max_results: Número máximo de tweets a extraer.
            output_file: Nombre base del archivo donde se almacenarán los datos extraídos (CSV).

        Proceso:
            - Realiza la llamada a la API utilizando requests.
            - Verifica el éxito de la petición (código de estado 200) y extrae el contenido en formato JSON.
            - Adapta los nombres de columnas mínimas (por ejemplo, 'author_id' a 'user_name', 'created_at' a 'date').
            - Guarda los datos en el archivo local (en formato CSV).

        Devuelve:
            DataFrame con los datos extraídos y almacenados.
        """
        rapidapi_key = os.getenv("RAPIDAPI_KEY")
        url = "https://twitter154.p.rapidapi.com/search/search"

        # Configuración inicial: se solicita 5 tweets por llamada (según lo permitido por la API)
        payload = {
            "query": query,
            "limit": 5,
            "section": "top",
            "language": "en",
            "min_likes": 20,
            "min_retweets": 20,
            "start_date": "2022-01-01"
        }

        headers = {
            "x-rapidapi-key": rapidapi_key,
            "x-rapidapi-host": "twitter154.p.rapidapi.com",
            "Content-Type": "application/json"
        }

        collected_tweets = []
        while len(collected_tweets) < max_results:
            response = requests.post(url, json=payload, headers=headers)
            if response.status_code != 200:
                print(f"Error {response.status_code}: {response.text}")
                break

            # Extraer el contenido en formato JSON
            response_data = response.json()

            # Se asume que la lista de tweets viene bajo la clave "results"
            if "results" in response_data:
                tweets = response_data["results"]
            else:
                print("La respuesta no contiene la clave 'results'.")
                break

            if not tweets:
                print("No se obtuvieron resultados en esta llamada.")
                break

            collected_tweets.extend(tweets)
            print(f"Recolectados {len(collected_tweets)} tweets hasta ahora...")

            # Verifica si existe token para paginación
            if "continuation_token" in response_data:
                payload["continuation_token"] = response_data["continuation_token"]
            else:
                break

        # Limitar la lista a max_results (por si se han recolectado más)
        collected_tweets = collected_tweets[:max_results]

        # Crear un DataFrame con los datos recolectados
        df = pd.DataFrame(collected_tweets)

        # Adaptar los nombres de las columnas mínimas
        df.rename(columns={
            "author_id": "user_name",
            "created_at": "date",
            # Aquí podrías agregar más mapeos según las necesidades del proyecto.
        }, inplace=True)

        self.data = df  # Almacena los datos en el atributo self.data

        # Guardar los datos en un archivo local (CSV)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_path = "../data/bronze"
        os.makedirs(dir_path, exist_ok=True)
        file_path = f"{dir_path}/{output_file.rstrip('.csv')}_{query}_{timestamp}.csv"
        df.to_csv(file_path, index=False)
        print(f"Datos guardados en: {file_path}")

        return self.data


# --- MAIN para probar load_data_api() ---
if __name__ == "__main__":
    # No se requiere 'source' cuando se extraen datos vía API.
    extractor = DataExtractor()
    search_term = input("Ingrese el término de búsqueda para la API: ")
    # Por ejemplo, queremos obtener hasta 100 tweets:
    df_api = extractor.load_data_api(query=search_term, max_results=100, output_file="tweets_from_api.csv")
    if df_api is not None:
        print("\nMuestra de datos extraídos:")
        print(df_api.head())
