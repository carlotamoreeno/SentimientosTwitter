import os
import json
import re
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import gensim
from gensim import corpora
from gensim.models import LdaModel
from textblob import TextBlob
import spacy
from spacy.lang.en import English
import heapq
from collections import defaultdict
import networkx as nx
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


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
        load_dotenv()  # Carga las variables de entorno 

        # Crear directorios para la arquitectura medallón si no existen
        self.bronze_dir = "../data/bronze"
        self.silver_dir = "../data/silver"
        self.gold_dir = "../data/gold"
        os.makedirs(self.bronze_dir, exist_ok=True)
        os.makedirs(self.silver_dir, exist_ok=True)
        os.makedirs(self.gold_dir, exist_ok=True)

        # Descargar los recursos necesarios de NLTK si no están disponibles

        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

        # Añadir esta nueva sección
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab')

        # Cargar el modelo de spaCy
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except:
            # Si no está instalado, mostrar instrucciones
            print(
                "Es necesario instalar el modelo de spaCy en_core_web_sm. Ejecuta: python -m spacy download en_core_web_sm")
            self.nlp = None

    # Esta función realiza la conexión a la API de Twitter mediante RapidAPI
   
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

        # Por defecto se solicitan 5 tweets por llamada (según lo permitido por la API)
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
            "x-rapidapi-key": rapidapi_key,  # el token esta en .env
            "x-rapidapi-host": "twitter154.p.rapidapi.com",
            "Content-Type": "application/json"
        }

        collected_tweets = []
        while len(collected_tweets) < max_results:
            response = requests.post(url, json=payload, headers=headers)
            if response.status_code != 200:
                print(f"Error {response.status_code}: {response.text}")
                break

            # Extrae el contenido en formato JSON
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
        }, inplace=True)

        self.data = df  # Almacena los datos en el atributo self.data

        # Guardar los datos en un archivo local (CSV) en la capa Bronze
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"{self.bronze_dir}/{output_file.rstrip('.csv')}_{query}_{timestamp}.csv"
        df.to_csv(file_path, index=False)
        print(f"Datos guardados en: {file_path}")

        # Guardar la ruta al archivo como fuente
        self.source = file_path

        return self.data

    # Método load_data existente...
    def load_data(self) -> pd.DataFrame:
        """
        Carga los datos desde el archivo local especificado en self.source.

        Devuelve:
            DataFrame con los datos cargados.
        """
        if self.source is None:
            raise ValueError("No se ha especificado un archivo fuente (self.source es None)")

        # Determinar la extensión del archivo
        file_ext = os.path.splitext(self.source)[1].lower()

        try:
            if file_ext == '.csv':
                # Usar chunksize para archivos grandes
                if os.path.getsize(self.source) > 100 * 1024 * 1024:  # Si es mayor a 100MB
                    self.data = pd.read_csv(self.source, chunksize=self.chunksize)
                else:
                    self.data = pd.read_csv(self.source)
            elif file_ext == '.json':
                with open(self.source, 'r', encoding='utf-8') as f:
                    self.data = pd.DataFrame(json.load(f))
            else:
                raise ValueError(f"Formato de archivo no soportado: {file_ext}")

            print(f"Datos cargados correctamente desde {self.source}")
            return self.data

        except Exception as e:
            print(f"Error al cargar los datos: {str(e)}")
            return None

    # Método clean_text 
    def clean_text(self, text: str) -> str:
        """
        Limpia y normaliza el texto.

        Pasos:
        - Convertir a minúsculas.
        - Eliminar URLs.
        - Conservar hashtags y eliminar espacios redundantes.

        Devuelve:
            El texto limpio.
        """
        if not isinstance(text, str):
            return ""

        # Convertir a minúsculas
        text = text.lower()

        # Eliminar URLs manteniendo el texto
        text = re.sub(r'https?://\S+', '', text)

        # Eliminar menciones (@usuario) comentamos esta linea para que no se eliminen las menciones
        #text = re.sub(r'@\w+', '', text)

        # Eliminar caracteres especiales excepto hashtags y letras
        text = re.sub(r'[^\w\s#]', ' ', text)

        # Eliminar espacios redundantes
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    # Método extract_hashtags
    def extract_hashtags(self, text: str) -> list:
        """
        Extrae y devuelve una lista de hashtags presentes en el texto.

        Implementación:
        - Utiliza expresiones regulares para encontrar palabras que comiencen con '#'.

        Devuelve:
            Lista de hashtags.
        """
        if not isinstance(text, str):
            return []

        # Buscar patrones que comiencen con # seguido de caracteres alfanuméricos
        hashtags = re.findall(r'#\w+', text)

        # Eliminar posibles duplicados y convertir a minúsculas
        hashtags = [tag.lower() for tag in hashtags]

        return hashtags

    # Método analytics_hashtags_extended
    def analytics_hashtags_extended(self) -> dict:
        """
        Realiza un análisis avanzado de hashtags sobre los datos cargados en self.data.

        Pasos:
        1. Normaliza la columna 'text' y almacena el resultado en 'clean_text'.
        2. Extrae hashtags de 'clean_text' y crea la columna 'hashtags'.
        3. Convierte la columna 'date' a tipo datetime (solo fecha).
        4. Explota la columna 'hashtags' para contar cada ocurrencia.
        5. Calcula:
           - Frecuencia global de cada hashtag ('overall').
           - Frecuencia de hashtags por usuario ('by_user').
           - Evolución de la frecuencia por fecha ('by_date').

        Devuelve:
            Diccionario con DataFrames en claves: 'overall', 'by_user', 'by_date'.
        """
        if self.data is None:
            raise ValueError("No hay datos cargados. Utilice load_data() o load_data_api() primero.")

        # Crear una copia para no modificar los datos originales
        df = self.data.copy()

        # 1. Normalizar el texto
        df['clean_text'] = df['text'].apply(self.clean_text)

        # 2. Extraer hashtags
        df['hashtags'] = df['clean_text'].apply(self.extract_hashtags)

        # 3. Convertir la columna 'date' a datetime
        if 'creation_date' in df.columns:
            # Tratamos de convertir creation_date a datetime, manejando diferentes formatos posibles
            try:
                df['date'] = pd.to_datetime(df['creation_date'], format='%a %b %d %H:%M:%S %z %Y', errors='coerce')
                df['date'] = df['date'].dt.date  # Extraer solo la fecha
            except:
                print("No se pudo convertir la columna 'creation_date' a formato datetime.")
        elif 'date' in df.columns:
            try:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df['date'] = df['date'].dt.date  # Extraer solo la fecha
            except:
                print("No se pudo convertir la columna 'date' a formato datetime.")
        else:
            print("No se encontró una columna de fecha ('date' o 'creation_date').")
            df['date'] = datetime.now().date()

        # 4. Explotar la columna 'hashtags' para contar cada ocurrencia
        # Crear un DataFrame separado para el análisis de hashtags
        all_hashtags = []
        for idx, row in df.iterrows():
            user = row['user']
            if isinstance(user, str) and user.startswith('{'):
                try:
                    user_dict = eval(user.replace('None', 'null').replace('True', 'true').replace('False', 'false'))
                    username = user_dict.get('username', 'unknown')
                except:
                    username = 'unknown'
            else:
                username = 'unknown'

            date = row['date']
            for hashtag in row['hashtags']:
                all_hashtags.append({
                    'hashtag': hashtag,
                    'user': username,
                    'date': date
                })

        hashtags_df = pd.DataFrame(all_hashtags)

        # 5. Calcular las frecuencias
        # Frecuencia global de cada hashtag
        if len(hashtags_df) > 0:
            overall = hashtags_df['hashtag'].value_counts().reset_index()
            overall.columns = ['hashtag', 'frequency']

            # Frecuencia de hashtags por usuario
            by_user = hashtags_df.groupby(['user', 'hashtag']).size().reset_index(name='frequency')
            by_user = by_user.sort_values(['user', 'frequency'], ascending=[True, False])

            # Evolución de la frecuencia por fecha
            by_date = hashtags_df.groupby(['date', 'hashtag']).size().reset_index(name='frequency')
            by_date = by_date.sort_values(['date', 'frequency'], ascending=[True, False])

            # Guardar los resultados en un archivo CSV en la capa Silver
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            silver_file_path = f"{self.silver_dir}/processed_data_{timestamp}.csv"
            df.to_csv(silver_file_path, index=False)
            print(f"Datos procesados guardados en: {silver_file_path}")

            # Guardar los análisis específicos
            overall.to_csv(f"{self.silver_dir}/hashtags_overall_{timestamp}.csv", index=False)
            by_user.to_csv(f"{self.silver_dir}/hashtags_by_user_{timestamp}.csv", index=False)
            by_date.to_csv(f"{self.silver_dir}/hashtags_by_date_{timestamp}.csv", index=False)

            print(f"Análisis de hashtags guardado en la capa Silver")

            return {'overall': overall, 'by_user': by_user, 'by_date': by_date}
        else:
            print("No se encontraron hashtags en los datos.")
            # Guardar los datos procesados aunque no haya hashtags
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            silver_file_path = f"{self.silver_dir}/processed_data_{timestamp}.csv"
            df.to_csv(silver_file_path, index=False)
            print(f"Datos procesados guardados en: {silver_file_path}")

            return {'overall': pd.DataFrame(), 'by_user': pd.DataFrame(), 'by_date': pd.DataFrame()}

    # Método generate_hashtag_wordcloud
    def generate_hashtag_wordcloud(self, overall_df: pd.DataFrame = None, max_words: int = 100,
                                   figsize: tuple = (10, 6)) -> None:
        """
        Genera y muestra una WordCloud basada en el análisis global de hashtags.

        Parámetros:
            overall_df: DataFrame con columnas ['hashtag', 'frequency']. Si es None, se calcula.
            max_words: Número máximo de palabras en la WordCloud.
            figsize: Tamaño de la figura.
        """
        if overall_df is None:
            # Si no se proporciona un DataFrame, usar analytics_hashtags_extended para obtenerlo
            try:
                analysis_result = self.analytics_hashtags_extended()
                overall_df = analysis_result['overall']
            except Exception as e:
                print(f"Error al generar el análisis de hashtags: {str(e)}")
                return

        if len(overall_df) == 0:
            print("No hay datos de hashtags para generar una nube de palabras.")
            return

        # Crear un diccionario de frecuencias
        word_freq = dict(zip(overall_df['hashtag'], overall_df['frequency']))

        # Generar la WordCloud
        try:
            wordcloud = WordCloud(
                width=1200,
                height=800,
                max_words=max_words,
                background_color='white',
                colormap='viridis',
                contour_width=1,
                contour_color='steelblue'
            ).generate_from_frequencies(word_freq)

            # Mostrar la WordCloud
            plt.figure(figsize=figsize)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.tight_layout(pad=0)

            # Guardar la figura en la capa Silver
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            wordcloud_path = f"{self.silver_dir}/hashtags_wordcloud_{timestamp}.png"
            plt.savefig(wordcloud_path, bbox_inches='tight', dpi=300)
            plt.close()

            print(f"Nube de palabras de hashtags guardada en: {wordcloud_path}")

        except Exception as e:
            print(f"Error al generar la nube de palabras: {str(e)}")

    # Método para análisis de tópicos con LDA (Nuevo)
    def model_topics(self, num_topics: int = 5, passes: int = 10) -> list:
        """
        Aplica el modelo LDA para descubrir tópicos en el corpus.

        Pasos:
        1. Asegurarse de que la columna 'clean_text' existe (se debe llamar previamente a clean_text).
        2. Tokeniza la columna 'clean_text' (división simple en palabras).
        3. Crea un diccionario y un corpus (bag-of-words) a partir de los tokens.
        4. Entrena el modelo LDA con los parámetros especificados.
        5. Extrae y muestra los tópicos en formato lista (cada tópico es una lista de palabras).

        Devuelve:
            Lista de tópicos, por ejemplo: [['word1', 'word2', ...], ['word3', 'word4', ...], ...]
        """
        if self.data is None:
            raise ValueError("No hay datos cargados.Porfavor utilice load_data() o load_data_api() primero.")

        # Verificar si existe la columna 'clean_text', y si no, crearla
        if 'clean_text' not in self.data.columns:
            self.data['clean_text'] = self.data['text'].apply(self.clean_text)

        # Obtener stopwords en inglés
        stop_words = set(stopwords.words('english'))

        # Tokenización y limpieza
        def preprocess_text(text):
            if not isinstance(text, str):
                return []

            tokens = word_tokenize(text)
            # Eliminar stopwords y tokens cortos
            tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
            return tokens

        # Aplicar preprocesamiento a cada texto
        tokenized_texts = self.data['clean_text'].apply(preprocess_text)

        # Filtrar textos vacíos
        tokenized_texts = [tokens for tokens in tokenized_texts if len(tokens) > 0]

        if len(tokenized_texts) == 0:
            print("No hay suficientes textos con tokens para modelar tópicos.")
            return []

        # Crear diccionario y corpus para el modelo LDA
        dictionary = corpora.Dictionary(tokenized_texts)
        corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

        # Entrenar el modelo LDA
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=passes,
            alpha='auto',
            random_state=42
        )

        # Obtener los tópicos con sus palabras más relevantes
        topics = []
        for topic_id in range(num_topics):
            # Obtener las 10 palabras más importantes de cada tópico
            topic_words = lda_model.show_topic(topic_id, topn=10)
            topics.append([word for word, prob in topic_words])

        # Guardar resultados en formato JSON en la capa Gold
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        topics_data = {f"topic_{i + 1}": words for i, words in enumerate(topics)}

        with open(f"{self.gold_dir}/lda_topics_{timestamp}.json", 'w') as f:
            json.dump(topics_data, f, indent=4)

        # Crear visualización de tópicos
        fig, axes = plt.subplots(1, num_topics, figsize=(20, 5))
        if num_topics == 1:
            axes = [axes]  # Convertir a lista para manejar un solo tópico

        for i, topic_words in enumerate(topics):
            word_probs = [lda_model.show_topic(i, topn=10)[j][1] for j in range(10)]
            axes[i].bar(topic_words, word_probs)
            axes[i].set_title(f'Tópico {i + 1}')
            axes[i].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(f"{self.gold_dir}/lda_topics_visualization_{timestamp}.png")
        plt.close()

        print(f"Análisis de tópicos LDA completado y guardado en la capa Gold")

        return topics

    # Método para análisis de sentimiento
    def analyze_sentiment(self, method: str = 'textblob') -> pd.DataFrame:
        """
        Analiza el sentimiento de cada tweet utilizando el método especificado.

        Parámetros:
            method: 'textblob' o 'spacy'. Si se elige 'spacy', se usará spacytextblob.

        Proceso:
            - Para cada 'clean_text', calcula la polaridad y subjetividad.
            - Almacena los resultados en las columnas 'sentiment_polarity' y 'sentiment_subjectivity'.

        Devuelve:
            DataFrame actualizado con las nuevas columnas de sentimiento.
        """
        if self.data is None:
            raise ValueError("No hay datos cargados. Utilice load_data() o load_data_api() primero.")

        # Verificar si existe la columna 'clean_text', y si no, crearla
        if 'clean_text' not in self.data.columns:
            self.data['clean_text'] = self.data['text'].apply(self.clean_text)

        # Crear una copia del DataFrame para trabajar
        df = self.data.copy()

        # Función para analizar sentimiento con TextBlob
        def get_sentiment_textblob(text):
            if not isinstance(text, str) or text == "":
                return {'polarity': 0.0, 'subjectivity': 0.0}

            analysis = TextBlob(text)
            return {
                'polarity': analysis.sentiment.polarity,
                'subjectivity': analysis.sentiment.subjectivity
            }

        # Función para analizar sentimiento con spaCy
        def get_sentiment_spacy(text):
            if not isinstance(text, str) or text == "" or self.nlp is None:
                return {'polarity': 0.0, 'subjectivity': 0.0}

            
            analysis = TextBlob(text)
            return {
                'polarity': analysis.sentiment.polarity,
                'subjectivity': analysis.sentiment.subjectivity
            }

        # Seleccionar método y aplicar análisis
        sentiment_func = get_sentiment_textblob if method == 'textblob' else get_sentiment_spacy
        sentiments = df['clean_text'].apply(sentiment_func)

        # Extraer polaridad y subjetividad
        df['sentiment_polarity'] = sentiments.apply(lambda x: x['polarity'])
        df['sentiment_subjectivity'] = sentiments.apply(lambda x: x['subjectivity'])

        # Categorizar el sentimiento
        def get_sentiment_category(polarity):
            if polarity > 0.1:
                return 'positive'
            elif polarity < -0.1:
                return 'negative'
            else:
                return 'neutral'

        df['sentiment_category'] = df['sentiment_polarity'].apply(get_sentiment_category)

        # Guardar resultados en la capa Silver
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sentiment_file = f"{self.silver_dir}/tweets_sentiment_{timestamp}.csv"
        df.to_csv(sentiment_file, index=False)

        # Crear visualizaciones de sentimiento
        # 1. Distribución de polaridad
        plt.figure(figsize=(10, 6))
        plt.hist(df['sentiment_polarity'], bins=20, color='skyblue', edgecolor='black')
        plt.title('La distribucionn de Polaridad del Sentimiento')
        plt.xlabel('Polaridad')
        plt.ylabel('Frecuencia')
        plt.savefig(f"{self.gold_dir}/sentiment_polarity_distribution_{timestamp}.png")
        plt.close()

        # 2. Distribución de categorías de sentimiento
        sentiment_counts = df['sentiment_category'].value_counts()
        plt.figure(figsize=(8, 8))
        plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%',
                colors=['green', 'gray', 'red'])
        plt.title('Distribución de Categorías de Sentimiento')
        plt.savefig(f"{self.gold_dir}/sentiment_categories_distribution_{timestamp}.png")
        plt.close()

        # Actualizar el DataFrame principal
        self.data = df

        print(f"Análisis de sentimiento completado y guardado en la capa Silver")
        print(f"Visualizaciones guardadas en la capa Gold")

        return self.data

    # Método para parsing y resumen de textos
    def parse_and_summarize(self, summary_ratio: float = 0.3) -> str:
        """
        Realiza un análisis de parsing y genera un resumen extractivo del corpus.

        Pasos:
        1. Concatena todos los textos limpios.
        2. Divide el texto concatenado en oraciones.
        3. Calcula una puntuación para cada oración basándose en la frecuencia de palabras (excluyendo stopwords).
        4. Selecciona las oraciones con mayor puntuación según el ratio especificado.
        5. Devuelve el resumen formado por las oraciones seleccionadas, manteniendo el orden original.

        Parámetros:
            summary_ratio: Proporción de oraciones a retener (ej. 0.3 para el 30%).

        Devuelve:
            Un string con el resumen generado.
        """
        if self.data is None:
            raise ValueError("No hay datos cargados. Porfavor utilice load_data() o load_data_api() primero.")

        # Verificar si existe la columna 'clean_text', y si no, crearla
        if 'clean_text' not in self.data.columns:
            self.data['clean_text'] = self.data['text'].apply(self.clean_text)

        # Filtrar textos no vacíos
        valid_texts = [text for text in self.data['clean_text'] if isinstance(text, str) and text.strip()]

        if not valid_texts:
            return "No hay suficientes textos para generar un resumen."

        # Concatenar todos los textos
        all_text = " ".join(valid_texts)

        # Dividir en oraciones
        sentences = sent_tokenize(all_text)

        if len(sentences) <= 1:
            return all_text

        # Obtener stopwords
        stop_words = set(stopwords.words('english'))

        # Calcular frecuencia de palabras
        word_frequencies = {}
        for sentence in sentences:
            for word in word_tokenize(sentence.lower()):
                if word not in stop_words and word.isalnum():
                    if word not in word_frequencies:
                        word_frequencies[word] = 1
                    else:
                        word_frequencies[word] += 1

        # Normalizar las frecuencias
        max_frequency = max(word_frequencies.values()) if word_frequencies else 1
        normalized_word_frequencies = {word: freq / max_frequency for word, freq in word_frequencies.items()}

        # Calcular la puntuación de cada oración
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            for word in word_tokenize(sentence.lower()):
                if word in normalized_word_frequencies:
                    if i not in sentence_scores:
                        sentence_scores[i] = normalized_word_frequencies[word]
                    else:
                        sentence_scores[i] += normalized_word_frequencies[word]

        # Determinar cuántas oraciones incluir en el resumen
        num_sentences = max(1, int(len(sentences) * summary_ratio))

        # Seleccionar las oraciones con mayor puntuación
        selected_indices = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
        selected_indices.sort()  # Mantener el orden original

        # Construir el resumen
        summary = ' '.join([sentences[i] for i in selected_indices])

        # Guardar el resumen en la capa Gold
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = f"{self.gold_dir}/corpus_summary_{timestamp}.txt"

        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)

        print(f"Resumen extractivo generado y guardado en: {summary_file}")

        return summary

    def generate_advanced_visualizations(self) -> None:
        """
        Genera visualizaciones avanzadas a partir de los análisis de sentimiento y tópicos.
        Guarda todas las visualizaciones en la capa Gold.
        """
        if self.data is None or len(self.data) == 0:
            print("No hay datos para generar visualizaciones.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Asegurarse de que existen las columnas necesarias
        if 'sentiment_polarity' not in self.data.columns or 'sentiment_category' not in self.data.columns:
            print("No se ha realizado análisis de sentimiento. Ejecutando analyze_sentiment()...")
            self.analyze_sentiment()

        # 1. Visualización de sentimiento a lo largo del tiempo
        if 'date' in self.data.columns:
            try:
                # Convertir a datetime si es necesario
                if not pd.api.types.is_datetime64_any_dtype(self.data['date']):
                    self.data['date'] = pd.to_datetime(self.data['date'], errors='coerce')

                # Agrupar por fecha y calcular la polaridad media
                sentiment_over_time = self.data.groupby(self.data['date'].dt.date)[
                    'sentiment_polarity'].mean().reset_index()

                plt.figure(figsize=(12, 6))
                plt.plot(sentiment_over_time['date'], sentiment_over_time['sentiment_polarity'], marker='o',
                         linestyle='-')
                plt.title('Evolución del Sentimiento a lo largo del tiempo')
                plt.xlabel('Fecha')
                plt.ylabel('Polaridad media')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(f"{self.gold_dir}/sentiment_over_time_{timestamp}.png")
                plt.close()

                print(f"Visualización de sentimiento a lo largo del tiempo guardada en la capa Gold")
            except Exception as e:
                print(f"Error al generar visualización de sentimiento temporal: {str(e)}")

        # 2. Relación entre subjetividad y polaridad
        try:
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(
                self.data['sentiment_subjectivity'],
                self.data['sentiment_polarity'],
                c=self.data['sentiment_polarity'].apply(lambda x: 'green' if x > 0 else ('red' if x < 0 else 'gray')),
                alpha=0.6,
                s=100
            )
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.axvline(x=0.5, color='black', linestyle='-', alpha=0.3)
            plt.title('Relación entre Subjetividad y Polaridad')
            plt.xlabel('Subjetividad')
            plt.ylabel('Polaridad')
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{self.gold_dir}/subjectivity_vs_polarity_{timestamp}.png")
            plt.close()

            print(f"Visualización de subjetividad vs polaridad guardada en la capa Gold")
        except Exception as e:
            print(f"Error al generar visualización de subjetividad vs polaridad: {str(e)}")

        # 3. Wordcloud separada por sentimiento
        try:
            # Generar nubes de palabras separadas por sentimiento
            for sentiment in ['positive', 'neutral', 'negative']:
                # Filtrar textos por categoría de sentimiento
                filtered_texts = ' '.join(
                    self.data[self.data['sentiment_category'] == sentiment]['clean_text'].dropna())

                if filtered_texts.strip():
                    # Configurar colores según sentimiento
                    if sentiment == 'positive':
                        colormap = 'Greens'
                    elif sentiment == 'negative':
                        colormap = 'Reds'
                    else:
                        colormap = 'Greys'

                    # Crear la nube de palabras
                    wordcloud = WordCloud(
                        width=800,
                        height=500,
                        max_words=100,
                        background_color='white',
                        colormap=colormap,
                        contour_width=1,
                        contour_color='steelblue'
                    ).generate(filtered_texts)

                    plt.figure(figsize=(10, 6))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    plt.title(f'Palabras más frecuentes en tweets {sentiment.capitalize()}')
                    plt.tight_layout(pad=0)
                    plt.savefig(f"{self.gold_dir}/wordcloud_{sentiment}_{timestamp}.png")
                    plt.close()

                    print(f"Nube de palabras para sentimiento {sentiment} guardada en la capa Gold")
        except Exception as e:
            print(f"Error al generar nubes de palabras por sentimiento: {str(e)}")

        print("Visualizaciones avanzadas completadas y guardadas en la capa Gold")

    def generate_final_report(self) -> str:
        """
        Genera un informe final en formato markdown con todos los resultados del análisis.
        Guarda el informe en la capa Gold.

        Devuelve:
            Ruta al archivo de informe.
        """
        if self.data is None:
            raise ValueError("No hay datos para generar el informe.")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"{self.gold_dir}/informe_final_{timestamp}.md"

        # Realizar análisis básicos si no se han realizado
        if 'clean_text' not in self.data.columns:
            self.data['clean_text'] = self.data['text'].apply(self.clean_text)

        if 'sentiment_polarity' not in self.data.columns:
            self.analyze_sentiment()

        # Realizar análisis de tópicos
        topics = self.model_topics(num_topics=5)

        # Obtener estadísticas generales
        tweet_count = len(self.data)

        # Estadísticas de sentimiento
        positive_count = len(self.data[self.data['sentiment_category'] == 'positive'])
        neutral_count = len(self.data[self.data['sentiment_category'] == 'neutral'])
        negative_count = len(self.data[self.data['sentiment_category'] == 'negative'])

        # Extraer resumen del corpus
        corpus_summary = self.parse_and_summarize(summary_ratio=0.2)

        # Generar contenido del informe
        report_content = f"""# Informe de Análisis de Tweets

## Resumen Ejecutivo

Este informe presenta el análisis de {tweet_count} tweets extraídos mediante API. 
El análisis incluye exploración de hashtags, análisis de sentimiento, modelado de tópicos 
y generación de resúmenes extractivos.

## Estadísticas Generales

- **Total de tweets analizados**: {tweet_count}
- **Distribución de sentimiento**:
  - Positivo: {positive_count} ({positive_count / tweet_count * 100:.1f}%)
  - Neutral: {neutral_count} ({neutral_count / tweet_count * 100:.1f}%)
  - Negativo: {negative_count} ({negative_count / tweet_count * 100:.1f}%)

## Análisis de Tópicos

Se identificaron los siguientes tópicos principales en el corpus:

"""

        # Agregar tópicos al informe
        for i, topic_words in enumerate(topics):
            topic_str = ", ".join(topic_words)
            report_content += f"### Tópico {i + 1}\n\n{topic_str}\n\n"

        # Agregar resumen del corpus
        report_content += f"""## Resumen Extractivo del Corpus

{corpus_summary}

## Imágenes y Visualizaciones

Las visualizaciones generadas se pueden encontrar en la carpeta 'gold' con el siguiente timestamp: {timestamp}.

## Metodología

1. **Extracción de datos**: Los tweets fueron extraídos mediante la API de Twitter a través de RapidAPI.
2. **Preprocesamiento**: Los textos fueron normalizados, eliminando URLs, menciones y caracteres especiales.
3. **Análisis de hashtags**: Se identificaron y analizaron los hashtags más frecuentes.
4. **Análisis de sentimiento**: Se utilizó TextBlob para determinar la polaridad y subjetividad de cada tweet.
5. **Modelado de tópicos**: Se aplicó el algoritmo LDA (Latent Dirichlet Allocation) para descubrir los temas principales.
6. **Generación de resúmenes**: Se utilizó un método extractivo basado en frecuencia de palabras para generar un resumen del corpus.

## Conclusiones

El análisis muestra tendencias importantes en los tweets analizados, tanto en términos de sentimiento como en los tópicos identificados.
Las visualizaciones proporcionan una representación gráfica de estos resultados para facilitar su interpretación.

---
Informe generado automáticamente el {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

        # Guardar el informe
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"Informe final generado y guardado en: {report_file}")

        return report_file

    # --- ANÁLISIS DE REDES CON NETWORKX ---
    def build_interaction_graph(self) -> nx.DiGraph:
        """
        Construye un grafo de interacciones a partir de los datos.
        Se asume que self.data tiene las columnas 'user' y 'text' para extraer interacciones.
        
        Las interacciones se basan en:
        - Menciones explícitas (@usuario) dentro de los tweets
        - Menciones implícitas (nombres de usuario o términos relevantes)
        - Referencias a otros usuarios
        
        Devuelve:
            Un grafo dirigido (nx.DiGraph) donde los nodos son usuarios y las aristas 
            representan menciones o referencias de un usuario a otro.
        """
        if self.data is None:
            raise ValueError("No hay datos cargados. Utilice load_data() o load_data_api() primero.")
        
        # Verificar que existen las columnas necesarias
        if 'user' not in self.data.columns:
            print("No se encontró la columna 'user' en los datos.")
            
        if 'text' not in self.data.columns:
            print("No se encontró la columna 'text' en los datos.")
            
        # Crear un grafo dirigido
        G = nx.DiGraph()
        
        # Función para extraer menciones explícitas
        def extract_mentions(text):
            if not isinstance(text, str):
                return []
            # Buscar patrones que comiencen con @ seguido de caracteres alfanuméricos
            mentions = re.findall(r'@\w+', text.lower())
            # Eliminar el @ y convertir a minúsculas
            return [mention[1:].lower() for mention in mentions]
        
        # Función para detectar menciones implícitas (sin @)
        def extract_implicit_mentions(text, known_users):
            if not isinstance(text, str) or not known_users:
                return []
            
            text_lower = text.lower()
            implicit_mentions = []
            
            # Buscar nombres de usuario conocidos en el texto
            for user in known_users:
                # Evitar falsos positivos con nombres muy cortos
                if len(user) <= 3:
                    continue
                    
                # Buscar el nombre de usuario sin @ como palabra separada
                pattern = r'\b' + re.escape(user) + r'\b'
                if re.search(pattern, text_lower):
                    implicit_mentions.append(user)
                    
            return implicit_mentions
            
        # Función para extraer el nombre de usuario
        def get_username(user_data):
            try:
                # Si es un diccionario, extraer directamente
                if isinstance(user_data, dict):
                    return user_data.get('username', '').lower() or 'unknown'
                
                # Si es un string, ver si es JSON o un diccionario serializado como string
                if isinstance(user_data, str):
                    # Caso 1: Es un diccionario con formato Python serializado como string
                    if user_data.startswith('{') and '}' in user_data:
                        try:
                            # Reemplazar valores Python por valores JSON
                            # Usamos ast.literal_eval que es más seguro que eval
                            import ast
                            safe_str = user_data.replace('None', 'null').replace('True', 'true').replace('False', 'false')
                            # Convertir de nuevo a formato Python para ast.literal_eval
                            safe_str = safe_str.replace('null', 'None').replace('true', 'True').replace('false', 'False')
                            user_dict = ast.literal_eval(safe_str)
                            return user_dict.get('username', '').lower() or 'unknown'
                        except Exception as e:
                            print(f"Error al evaluar diccionario de usuario: {str(e)}")
                            
                    # Caso 2: Intentar parsear como JSON
                    try:
                        user_dict = json.loads(user_data)
                        return user_dict.get('username', '').lower() or 'unknown'
                    except:
                        pass
            except Exception as e:
                print(f"Error al procesar usuario: {str(e)}")
            
            return 'unknown'
            
        # Función para extraer hashtags
        def extract_hashtags(text):
            if not isinstance(text, str):
                return []
            # Buscar patrones que comiencen con # seguido de caracteres alfanuméricos
            hashtags = re.findall(r'#\w+', text.lower())
            # Eliminar el # y convertir a minúsculas
            return [hashtag[1:].lower() for hashtag in hashtags]
        
        print("Se esta construyendo elgrafo de interacciones...")
        
        # Inspeccionar algunos registros para debug
        print(f"Muestra de datos, los primeros 3 registros:")
        for i, (_, row) in enumerate(self.data.head(3).iterrows()):
            print(f"\nRegistro {i+1}:")
            print(f"- Tipo de 'user': {type(row.get('user', 'No encontrado'))}")
            user_sample = str(row.get('user', 'No encontrado'))[:100] + "..." if len(str(row.get('user', 'No encontrado'))) > 100 else str(row.get('user', 'No encontrado'))
            print(f"- Muestra de 'user': {user_sample}")
            username = get_username(row.get('user', {}))
            print(f"- Username extraído: {username}")
            
            text_sample = str(row.get('text', 'No encontrado'))[:100] + "..." if len(str(row.get('text', 'No encontrado'))) > 100 else str(row.get('text', 'No encontrado'))
            print(f"- Muestra de 'text': {text_sample}")
            mentions = extract_mentions(row.get('text', ''))
            print(f"- Menciones explícitas: {mentions}")
        
        # Primera pasada: recopilar todos los usuarios
        known_users = set()
        usernames_by_id = {}  # Para mapear IDs a usernames
        
        for _, row in self.data.iterrows():
            try:
                # Extraer el username del campo 'user'
                username = get_username(row.get('user', {}))
                
                if username != 'unknown':
                    known_users.add(username)
                    
                # También extraer el user_id si está disponible
                if isinstance(row.get('user', {}), str) and row.get('user', {}).startswith('{'):
                    try:
                        import ast
                        safe_str = row.get('user', '{}').replace('None', 'null').replace('True', 'true').replace('False', 'false')
                        safe_str = safe_str.replace('null', 'None').replace('true', 'True').replace('false', 'False')
                        user_dict = ast.literal_eval(safe_str)
                        user_id = user_dict.get('user_id', '')
                        if user_id and username != 'unknown':
                            usernames_by_id[user_id] = username
                    except:
                        pass
                
            except Exception as e:
                print(f"Error al extraer usuario: {str(e)}")
                
        print(f"Se identificaron {len(known_users)} usuarios distintos.")
        
        # Términos relevantes del dominio para buscar menciones implícitas
        domain_terms = [
            'rafaelnadal', 'rafa', 'nadal', 'nadalfc', 'rafanadal',
            'elonmusk', 'elon', 'musk', 'tesla',
        ]
        
        # Segunda pasada: construir el grafo con menciones explícitas e implícitas
        for _, row in self.data.iterrows():
            try:
                # Obtener el usuario que emitió el tweet
                source_user = get_username(row.get('user', {}))
                
                # Si no se pudo determinar el usuario fuente, continuar con el siguiente tweet
                if source_user == 'unknown':
                    continue
                
                # Añadir el nodo al grafo (si no existe)
                if not G.has_node(source_user):
                    G.add_node(source_user)
                
                # Extraer menciones del texto
                text = row.get('text', '')
                
                # 1. Menciones explícitas (@usuario)
                explicit_mentions = extract_mentions(text)
                
                # 2. Menciones implícitas (nombres sin @)
                implicit_mentions = extract_implicit_mentions(text, known_users.union(domain_terms))
                
                # 3. Hashtags como nodos adicionales
                hashtags = [f"#{tag}" for tag in extract_hashtags(text)]
                
                # Combinar todas las menciones (usuarios y hashtags)
                all_mentions = set(explicit_mentions + implicit_mentions)
                
                # Añadir aristas por cada mención
                for target_user in all_mentions:
                    # No añadir auto-menciones
                    if source_user != target_user:
                        # Añadir el nodo destino si no existe
                        if not G.has_node(target_user):
                            G.add_node(target_user)
                        
                        # Añadir la arista o incrementar su peso si ya existe
                        if G.has_edge(source_user, target_user):
                            G[source_user][target_user]['weight'] += 1
                        else:
                            G.add_edge(source_user, target_user, weight=1)
                
                # Añadir hashtags como nodos
                for hashtag in hashtags:
                    if not G.has_node(hashtag):
                        G.add_node(hashtag, type='hashtag')
                    
                    # Conectar usuario con hashtag
                    if G.has_edge(source_user, hashtag):
                        G[source_user][hashtag]['weight'] += 1
                    else:
                        G.add_edge(source_user, hashtag, weight=1)
                
            except Exception as e:
                print(f"Error al procesar un tweet: {str(e)}")
                continue
        
        # Si el grafo quedó muy pequeño, intentar crear conexiones adicionales basadas en el tema común
        if len(G.edges()) < 5:
            print("Pocas interacciones detectadas. Añadiendo conexiones basadas en tema común...")
            
            # Crear un nodo central para el tema principal (según dataset)
            central_topic = "rafa_nadal" if any("nadal" in k for k in known_users) else "elon_musk"
            G.add_node(central_topic, type='topic')
            
            # Conectar todos los usuarios al tema central
            for user in known_users:
                if not G.has_edge(user, central_topic):
                    G.add_edge(user, central_topic, weight=0.5)
                if not G.has_edge(central_topic, user):
                    G.add_edge(central_topic, user, weight=0.5)
        
        print(f"Grafo construido con {len(G.nodes())} nodos y {len(G.edges())} aristas.")
        
        # Guardar el grafo en formato GraphML en la capa Gold
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        graph_file = f"{self.gold_dir}/interaction_graph_{timestamp}.graphml"
        try:
            nx.write_graphml(G, graph_file)
            print(f"Grafo de interacciones guardado en: {graph_file}")
        except Exception as e:
            print(f"Error al guardar el grafo: {str(e)}")
        
        return G
    
    def analyze_network(self, G: nx.DiGraph = None) -> dict:
        """
        Calcula métricas de red y detecta comunidades utilizando el algoritmo de Louvain.
        Imprime estadísticas y genera visualizaciones.
        
        Parámetros:
            G: Grafo dirigido de networkx. Si es None, se llama a build_interaction_graph.
            
        Devuelve:
            Un diccionario con las métricas calculadas y resultados del análisis.
        """
        if G is None:
            G = self.build_interaction_graph()
        
        if len(G.nodes()) == 0:
            print("El grafo está vacío. No hay interacciones para analizar.")
            return {}
        
        # Convertir a grafo no dirigido para algunas métricas y detección de comunidades
        UG = G.to_undirected()
        
        # Calcular métricas básicas
        metrics = {}
        
        # Número de nodos y aristas
        metrics['num_nodes'] = len(G.nodes())
        metrics['num_edges'] = len(G.edges())
        
        # Grado (in, out, total)
        in_degree = dict(G.in_degree(weight='weight'))
        out_degree = dict(G.out_degree(weight='weight'))
        metrics['max_in_degree'] = max(in_degree.items(), key=lambda x: x[1]) if in_degree else ('none', 0)
        metrics['max_out_degree'] = max(out_degree.items(), key=lambda x: x[1]) if out_degree else ('none', 0)
        
        # Calcular centralidades
        try:
            # Centralidad de grado
            degree_centrality = nx.degree_centrality(G)
            metrics['top_degree_centrality'] = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:3]
            
            # Centralidad de intermediación (betweenness)
            betweenness_centrality = nx.betweenness_centrality(G, weight='weight')
            metrics['top_betweenness_centrality'] = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:3]
            
            # Centralidad de cercanía (closeness)
            # Puede fallar en grafos no conectados, usamos try/except
            try:
                closeness_centrality = nx.closeness_centrality(G)
                metrics['top_closeness_centrality'] = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:3]
            except:
                metrics['top_closeness_centrality'] = []
            
            # Centralidad de eigenvector
            try:
                eigenvector_centrality = nx.eigenvector_centrality(G, weight='weight')
                metrics['top_eigenvector_centrality'] = sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)[:3]
            except:
                # Puede fallar en grafos muy grandes o no convergentes
                metrics['top_eigenvector_centrality'] = []
        except Exception as e:
            print(f"Error al calcular centralidades: {str(e)}")
        
        # Detección de comunidades (en el grafo no dirigido)
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(UG)
            
            # Contar nodos por comunidad
            community_counts = {}
            for node, community_id in partition.items():
                if community_id not in community_counts:
                    community_counts[community_id] = []
                community_counts[community_id].append(node)
            
            # Ordenar comunidades por tamaño
            sorted_communities = sorted(community_counts.items(), key=lambda x: len(x[1]), reverse=True)
            metrics['communities'] = sorted_communities
            metrics['num_communities'] = len(sorted_communities)
            
            # Las comunidades más grandes
            metrics['largest_communities'] = sorted_communities[:3] if len(sorted_communities) >= 3 else sorted_communities
        except ImportError:
            print("No se pudo importar python-louvain. Instálelo con: pip install python-louvain")
            metrics['communities'] = []
            metrics['num_communities'] = 0
            metrics['largest_communities'] = []
        except Exception as e:
            print(f"Error en la detección de comunidades: {str(e)}")
            metrics['communities'] = []
            metrics['num_communities'] = 0
            metrics['largest_communities'] = []
        
        # Visualizar el grafo
        try:
            plt.figure(figsize=(12, 10))
            
            # Posicionamiento usando Fruchterman-Reingold
            pos = nx.spring_layout(G, k=0.3)
            
            # Calcular tamaños de nodo basados en centralidad de grado
            node_size = [v * 3000 + 100 for v in degree_centrality.values()]
            
            # Dibujar nodos
            nx.draw_networkx_nodes(G, pos, 
                                  node_size=node_size,
                                  node_color="skyblue", 
                                  alpha=0.8)
            
            # Dibujar aristas con transparencia proporcional al peso
            edge_width = [data['weight'] * 0.2 for _, _, data in G.edges(data=True)]
            nx.draw_networkx_edges(G, pos, 
                                  width=edge_width,
                                  alpha=0.4, 
                                  edge_color="gray",
                                  arrows=True,
                                  arrowsize=10)
            
            # Añadir etiquetas solo para los nodos más centrales
            top_nodes = [node for node, _ in metrics['top_degree_centrality']]
            labels = {node: node for node in top_nodes}
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)
            
            plt.title(f"Grafo de interacciones ({len(G.nodes())} usuarios, {len(G.edges())} interacciones)")
            plt.axis('off')
            
            # Guardar la visualización
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            viz_file = f"{self.gold_dir}/network_visualization_{timestamp}.png"
            plt.tight_layout()
            plt.savefig(viz_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Visualización de la red guardada en: {viz_file}")
            
            # Si hay detección de comunidades, crear visualización con colores por comunidad
            if metrics['communities']:
                plt.figure(figsize=(12, 10))
                
                # Colores por comunidad
                cmap = plt.get_cmap('tab20')
                colors = [cmap(i) for i in range(min(20, len(metrics['communities'])))]
                
                # Asignar colores a nodos según su comunidad
                node_colors = [colors[partition[node] % len(colors)] for node in G.nodes()]
                
                # Dibujar con colores por comunidad
                nx.draw_networkx_nodes(G, pos, 
                                      node_size=node_size,
                                      node_color=node_colors, 
                                      alpha=0.8)
                
                # Dibujar aristas
                nx.draw_networkx_edges(G, pos, 
                                      width=edge_width,
                                      alpha=0.3, 
                                      edge_color="gray",
                                      arrows=True,
                                      arrowsize=10)
                
                # Etiquetas para nodos principales
                nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)
                
                plt.title(f"Comunidades en la red de interacciones ({metrics['num_communities']} comunidades)")
                plt.axis('off')
                
                # Guardar visualización de comunidades
                community_viz_file = f"{self.gold_dir}/network_communities_{timestamp}.png"
                plt.tight_layout()
                plt.savefig(community_viz_file, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Visualización de comunidades guardada en: {community_viz_file}")
                
        except Exception as e:
            print(f"Error al generar visualización: {str(e)}")
        
        # Guardar métricas en JSON
        try:
            # Filtrar datos serializables para JSON
            json_metrics = {
                'num_nodes': metrics['num_nodes'],
                'num_edges': metrics['num_edges'],
                'max_in_degree': {'user': metrics['max_in_degree'][0], 'value': metrics['max_in_degree'][1]},
                'max_out_degree': {'user': metrics['max_out_degree'][0], 'value': metrics['max_out_degree'][1]},
                'num_communities': metrics['num_communities'],
                'top_degree_centrality': [{'user': u, 'value': v} for u, v in metrics.get('top_degree_centrality', [])],
                'top_betweenness_centrality': [{'user': u, 'value': v} for u, v in metrics.get('top_betweenness_centrality', [])]
            }
            
            # Guardar en JSON
            metrics_file = f"{self.gold_dir}/network_metrics_{timestamp}.json"
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(json_metrics, f, indent=4)
            print(f"Métricas de red guardadas en: {metrics_file}")
        except Exception as e:
            print(f"Error al guardar métricas: {str(e)}")
        
        return metrics
    
    # --- IMPLEMENTACIÓN DE LLMs Y CHAT INTERACTIVO ---
    def generate_prompt_from_network(self, G: nx.DiGraph = None, metrics: dict = None) -> str:
        """
        Genera un prompt para el LLM utilizando insights del análisis de la red.
        Se extraen métricas como el top 3 de nodos por centralidad y el hashtag más frecuente.
        
        Parámetros:
            G: Grafo dirigido de networkx. Si es None, se usa el último grafo generado.
            metrics: Diccionario con métricas de red. Si es None, se calcula usando analyze_network.
            
        Devuelve:
            Un string con el prompt generado.
        """
        # Si no se proporcionan métricas ni grafo, realizar el análisis
        if metrics is None:
            if G is None:
                # Intentar cargar un grafo guardado o crear uno nuevo
                try:
                    metrics = self.analyze_network()
                except Exception as e:
                    print(f"Error al analizar la red: {str(e)}")
                    return "No se pudieron obtener métricas de la red para generar el prompt."
        
        # Obtener los hashtags más frecuentes si están disponibles
        top_hashtags = []
        try:
            hashtag_analysis = self.analytics_hashtags_extended()
            if 'overall' in hashtag_analysis and not hashtag_analysis['overall'].empty:
                top_hashtags = hashtag_analysis['overall'].head(5)['hashtag'].tolist()
        except Exception as e:
            print(f"Error al obtener hashtags: {str(e)}")
        
        # Crear el prompt
        prompt = "Eres un experto en análisis de redes sociales y comportamiento en línea. "
        prompt += "He analizado una red de interacciones en Twitter y te proporciono los siguientes datos:\n\n"
        
        # Añadir métricas básicas
        prompt += f"- La red tiene {metrics.get('num_nodes', 'desconocido')} usuarios y {metrics.get('num_edges', 'desconocido')} interacciones.\n"
        
        # Añadir información sobre comunidades
        if 'num_communities' in metrics and metrics['num_communities'] > 0:
            prompt += f"- Se han detectado {metrics.get('num_communities', 0)} comunidades distintas en la red.\n"
            
            # Información sobre las comunidades más grandes
            if 'largest_communities' in metrics and metrics['largest_communities']:
                prompt += "- Las comunidades más grandes son:\n"
                for i, (comm_id, members) in enumerate(metrics['largest_communities'][:3]):
                    prompt += f"  * Comunidad {i+1}: {len(members)} miembros\n"
        
        # Añadir usuarios más influyentes
        if 'top_degree_centrality' in metrics and metrics['top_degree_centrality']:
            prompt += "\nUsuarios con mayor centralidad de grado (los más conectados):\n"
            for user, value in metrics['top_degree_centrality']:
                prompt += f"- @{user}: {value:.4f}\n"
        
        if 'top_betweenness_centrality' in metrics and metrics['top_betweenness_centrality']:
            prompt += "\nUsuarios con mayor centralidad de intermediación (puentes entre comunidades):\n"
            for user, value in metrics['top_betweenness_centrality']:
                prompt += f"- @{user}: {value:.4f}\n"
        
        # Añadir usuario con más seguidores y más activo
        if 'max_in_degree' in metrics:
            prompt += f"\nUsuario más mencionado: @{metrics['max_in_degree'][0]} con {metrics['max_in_degree'][1]} menciones\n"
        
        if 'max_out_degree' in metrics:
            prompt += f"Usuario que más menciona a otros: @{metrics['max_out_degree'][0]} con {metrics['max_out_degree'][1]} menciones a otros\n"
        
        # Añadir hashtags populares
        if top_hashtags:
            prompt += "\nHashtags más utilizados en los tweets analizados:\n"
            for hashtag in top_hashtags:
                prompt += f"- #{hashtag}\n"
        
        # Solicitar análisis
        prompt += "\nBasándote en estos datos, por favor:\n"
        prompt += "1. Identifica patrones de interacción importantes en esta red social.\n"
        prompt += "2. ¿Qué tipo de estructura de red parece ser (centralizada, descentralizada, distribuida)?\n"
        prompt += "3. ¿Qué roles parecen tener los usuarios más centrales?\n"
        prompt += "4. ¿Qué temas o conversaciones podrían estar ocurriendo basados en los hashtags y la estructura de la red?\n"
        prompt += "5. Proporciona recomendaciones para alguien que quiera aumentar su influencia en esta red.\n"
        
        # Guardar el prompt generado en un archivo
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prompt_file = f"{self.gold_dir}/llm_prompt_{timestamp}.txt"
            with open(prompt_file, 'w', encoding='utf-8') as f:
                f.write(prompt)
            print(f"Prompt guardado en: {prompt_file}")
        except Exception as e:
            print(f"Error al guardar el prompt: {str(e)}")
        
        return prompt
    
    def chat_local_llm(self, prompt: str = None, model_name: str = "google/gemma-2-2b-it"):
        """
        Levanta un modelo LLM preentrenado (gemma-2-2b-it por defecto) o utiliza la API de OpenAI
        para generar respuestas basadas en el prompt.
        
        Parámetros:
            prompt: String con el prompt inicial. Si es None, se solicitará al usuario.
            model_name: Nombre del modelo a utilizar. Puede ser:
                       - un modelo de HuggingFace (ej: "google/gemma-2-2b-it")
                       - un modelo de OpenAI (ej: "gpt-4o", "gpt-4", "gpt-3.5-turbo")
            
        Devuelve:
            La respuesta generada por el modelo.
        """
        try:
            # Si no se proporciona un prompt, generar uno a partir del análisis de red
            if prompt is None:
                try:
                    print("Generando prompt a partir del análisis de red...")
                    prompt = self.generate_prompt_from_network()
                except Exception as e:
                    print(f"Error al generar prompt: {str(e)}")
                    prompt = input("Por favor, ingrese un prompt para el modelo: ")
            
            print("\nPrompt para el modelo:")
            print("-" * 50)
            print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
            print("-" * 50)
            
            # Verificar si es un modelo de OpenAI
            is_openai_model = model_name.lower() in ["gpt-4o", "gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"]
            
            if is_openai_model:
                # Usar la API de OpenAI
                import openai
                
                # Obtener la clave de API desde el archivo .env
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    print("AVISO: No se encontró OPENAI_API_KEY en el archivo .env")
                    api_key = input("Ingrese su clave de API de OpenAI: ")
                    os.environ["OPENAI_API_KEY"] = api_key
                else:
                    print("OPENAI_API_KEY encontrada en el archivo .env")
                
                # Configurar el cliente
                client = openai.OpenAI(api_key=api_key)
                
                print(f"\nUsando modelo de OpenAI: {model_name}")
                print("Generando respuesta, esto puede tomar unos momentos...")
                
                # Llamar a la API de OpenAI
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "Eres un experto en análisis de redes sociales y comportamiento en línea."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=2000
                )
                
                # Extraer la respuesta del modelo
                response_text = response.choices[0].message.content
                
            else:
                # Usar un modelo local de HuggingFace
                # Crear directorio para cachear modelos si no existe
                os.makedirs("models_cache", exist_ok=True)
                
                print(f"Cargando modelo {model_name}...")
                # Cargar tokenizer y modelo
                tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="models_cache")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    cache_dir="models_cache",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                
                # Verificar disponibilidad de GPU
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"Usando dispositivo: {device}")
                
                if device == "cpu":
                    print("ADVERTENCIA: Estás usando CPU para inferencia. El proceso puede ser muy lento.")
                
                # Generar respuesta
                print("\nGenerando respuesta, esto puede tomar un tiempo...")
                inputs = tokenizer(prompt, return_tensors="pt")
                if device == "cuda":
                    inputs = inputs.to(device)
                
                # Configuración de generación
                gen_config = {
                    "max_new_tokens": 1024,
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "repetition_penalty": 1.15
                }
                
                # Generar respuesta
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        **gen_config
                    )
                
                # Decodificar y obtener solo la respuesta (no el prompt)
                full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response_text = full_response[len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):].strip()
            
            # Guardar respuesta independientemente del modelo usado
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            response_file = f"{self.gold_dir}/llm_response_{timestamp}.txt"
            with open(response_file, 'w', encoding='utf-8') as f:
                f.write(f"PROMPT:\n{prompt}\n\nRESPUESTA:\n{response_text}")
            print(f"\nRespuesta guardada en: {response_file}")
            
            # Mostrar respuesta
            print("\nRespuesta del modelo:")
            print("=" * 50)
            print(response_text)
            print("=" * 50)
            
            return response_text
            
        except Exception as e:
            print(f"Error al usar el modelo LLM: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"
            
# --- MAIN para probar las fases Bronze, Silver y Gold ---
if __name__ == "__main__":
    # Crear una instancia del extractor
    extractor = DataExtractor()

    # Preguntar al usuario qué acción quiere realizar
    print("Seleccione una opción:")
    print("1. Extraer datos de la API de Twitter (fase Bronze)")
    print("2. Cargar datos existentes y procesar (fases Silver y Gold)")
    print("3. Análisis de red social con NetworkX")
    print("4. Generar prompt para LLM y obtener análisis")
    option = input("Opción (1/2/3/4): ")

    if option == "1":
        # Fase Bronze: Extraer datos de la API
        search_term = input("Ingrese el término de búsqueda para la API: ")
        max_results = int(input("Ingrese el número máximo de resultados a obtener (ej. 100): ") or "100")

        # Extraer datos
        df_api = extractor.load_data_api(query=search_term, max_results=max_results)

        if df_api is not None:
            print("\nMuestra de datos extraídos (primeros 5 registros):")
            print(df_api.head())

            # Preguntar si quiere continuar con las fases Silver y Gold
            continue_next = input("\n¿Desea continuar con el procesamiento Silver y Gold? (s/n): ")
            if continue_next.lower() != 's':
                print("Proceso finalizado en la fase Bronze.")
                exit()

    if option in ["2", "3", "4"] or (option == "1" and continue_next.lower() == 's'):
        # Fase Silver y Gold: Cargar y procesar datos
        if option in ["2", "3", "4"]:
            # Si viene de la opción 2, 3 o 4, necesitamos cargar los datos desde un archivo
            bronze_files = [f for f in os.listdir(extractor.bronze_dir) if f.endswith('.csv')]

            if not bronze_files:
                print(f"No se encontraron archivos CSV en {extractor.bronze_dir}.")
                exit()

            # Mostrar los archivos disponibles
            print("\nArchivos disponibles en Bronze:")
            for i, file in enumerate(bronze_files):
                print(f"{i + 1}. {file}")

            # Seleccionar archivo
            file_index = int(input("\nSeleccione el número del archivo a procesar: ")) - 1
            if file_index < 0 or file_index >= len(bronze_files):
                print("Índice inválido.")
                exit()

            file_path = f"{extractor.bronze_dir}/{bronze_files[file_index]}"
            extractor.source = file_path

            # Cargar los datos
            df = extractor.load_data()
            if df is None:
                print("Error al cargar los datos.")
                exit()

        if option in ["2", "1"]:
            # Fase Silver: Mostrar información básica del dataset
            print("\n--- FASE SILVER: PROCESAMIENTO Y LIMPIEZA ---")
            print("\nInformación del dataset:")
            print(f"- Número de registros: {len(extractor.data)}")
            print(f"- Columnas disponibles: {extractor.data.columns.tolist()}")

            # Mostrar ejemplo de limpieza de texto
            if 'text' in extractor.data.columns and len(extractor.data) > 0:
                sample_text = extractor.data['text'].iloc[0]
                clean_sample = extractor.clean_text(sample_text)
                print("\nEjemplo de limpieza de texto:")
                print(f"Original: {sample_text[:100]}..." if len(sample_text) > 100 else f"Original: {sample_text}")
                print(f"Limpio: {clean_sample[:100]}..." if len(clean_sample) > 100 else f"Limpio: {clean_sample}")

            # Realizar análisis de hashtags
            print("\nRealizando análisis de hashtags...")
            hashtag_analysis = extractor.analytics_hashtags_extended()

            # Mostrar resultados del análisis
            overall_hashtags = hashtag_analysis['overall']
            if not overall_hashtags.empty:
                print("\nTop 10 hashtags más frecuentes:")
                print(overall_hashtags.head(10))

                # Generar nube de palabras
                print("\nGenerando nube de palabras de hashtags...")
                extractor.generate_hashtag_wordcloud(overall_hashtags)
            else:
                print("No se encontraron hashtags en los datos.")

            # Fase Gold: Análisis avanzado
            print("\n--- FASE GOLD: ANÁLISIS AVANZADO ---")

            # Análisis de sentimiento
            print("\nRealizando análisis de sentimiento...")
            extractor.analyze_sentiment()

            # Mostrar distribución de sentimiento
            sentiment_counts = extractor.data['sentiment_category'].value_counts()
            print("\nDistribución de sentimiento:")
            for sentiment, count in sentiment_counts.items():
                percentage = count / len(extractor.data) * 100
                print(f"- {sentiment.capitalize()}: {count} tweets ({percentage:.1f}%)")

            # Modelado de tópicos
            print("\nRealizando modelado de tópicos con LDA...")
            topics = extractor.model_topics(num_topics=5)

            # Mostrar tópicos
            print("\nTópicos identificados:")
            for i, topic_words in enumerate(topics):
                print(f"Tópico {i + 1}: {', '.join(topic_words)}")

            # Generar resumen extractivo
            print("\nGenerando resumen extractivo del corpus...")
            summary = extractor.parse_and_summarize(summary_ratio=0.2)

            print("\nResumen del corpus:")
            print(summary[:300] + "..." if len(summary) > 300 else summary)

            # Generar visualizaciones avanzadas
            print("\nGenerando visualizaciones avanzadas...")
            extractor.generate_advanced_visualizations()

            # Generar informe final
            print("\nGenerando informe final...")
            report_path = extractor.generate_final_report()

            print("\nProceso completado:")
            print(f"- Fase Bronze: Datos extraídos guardados en {extractor.bronze_dir}")
            print(f"- Fase Silver: Datos procesados guardados en {extractor.silver_dir}")
            print(f"- Fase Gold: Resultados de análisis guardados en {extractor.gold_dir}")
            print(f"- Informe final: {report_path}")
        
        # Opción 3: Análisis de Red Social con NetworkX
        if option == "3":
            print("\n--- ANÁLISIS DE RED SOCIAL CON NETWORKX ---")
            
            # Construir el grafo de interacciones
            print("\nConstruyendo grafo de interacciones...")
            G = extractor.build_interaction_graph()
            
            if len(G.nodes()) == 0:
                print("No se encontraron interacciones para analizar. Asegúrese de que los datos contienen menciones (@usuario).")
                exit()
            
            print(f"\nGrafo construido con {len(G.nodes())} usuarios y {len(G.edges())} interacciones")
            
            # Analizar la red
            print("\nAnalizando red y calculando métricas...")
            metrics = extractor.analyze_network(G)
            
            # Mostrar métricas principales
            print("\nMétricas principales de la red:")
            print(f"- Número de nodos (usuarios): {metrics.get('num_nodes', 0)}")
            print(f"- Número de aristas (interacciones): {metrics.get('num_edges', 0)}")
            
            if 'num_communities' in metrics and metrics['num_communities'] > 0:
                print(f"- Número de comunidades detectadas: {metrics['num_communities']}")
                print("\nComunidades más grandes:")
                for i, (comm_id, members) in enumerate(metrics.get('largest_communities', [])[:3]):
                    print(f"  * Comunidad {i+1}: {len(members)} miembros")
            
            # Mostrar usuarios más influyentes
            if 'top_degree_centrality' in metrics and metrics['top_degree_centrality']:
                print("\nUsuarios con mayor centralidad de grado (más conectados):")
                for user, value in metrics['top_degree_centrality']:
                    print(f"- @{user}: {value:.4f}")
            
            if 'max_in_degree' in metrics:
                print(f"\nUsuario más mencionado: @{metrics['max_in_degree'][0]} con {metrics['max_in_degree'][1]} menciones")
            
            if 'max_out_degree' in metrics:
                print(f"Usuario que más menciona a otros: @{metrics['max_out_degree'][0]} con {metrics['max_out_degree'][1]} menciones")
            
            # Preguntar si desea continuar con el análisis LLM
            continue_llm = input("\n¿Desea continuar con el análisis mediante LLM? (s/n): ")
            if continue_llm.lower() == 's':
                # Pasar a la opción 4
                option = "4"
            else:
                print("\nProceso de análisis de red completado.")
                exit()
                
        # Opción 4: Generar prompt para LLM y obtener análisis
        if option == "4":
            print("\n--- ANÁLISIS CON MODELO DE LENGUAJE (LLM) ---")
            
            # Verificar si ya tenemos métricas de red
            if 'metrics' not in locals():
                print("\nSe esta generando análisis de red para crear el prompt...")
                metrics = extractor.analyze_network()
            
            # Generar prompt para LLM
            print("\nSe est generando prompt para el modelo de lenguaje...")
            prompt = extractor.generate_prompt_from_network(metrics=metrics)
            
            # Preguntar si desea ejecutar el modelo LLM
            run_model = input("\n¿Desea ejecutar el modelo de lenguaje? (Puede ser lento en CPU) (s/n): ")
            if run_model.lower() == 's':
                # Configuración del modelo
                model_name = "gpt-4o"
                print(f"\nUsando modelo {model_name}...")
                
                # Ejecutar el modelo con el prompt
                extractor.chat_local_llm(prompt=prompt, model_name=model_name)
            else:
                # Mostrar el prompt generado
                print("\nPrompt generado (puede usarlo manualmente en cualquier LLM):")
                print("-" * 50)
                print(prompt)
                print("-" * 50)
            
            print("\nProceso completado.")