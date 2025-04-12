# Análisis de Sentimiento y Tendencias en Redes Sociales

## Descripción del Proyecto

Este proyecto implementa un sistema completo para el análisis de tendencias y sentimientos en redes sociales, específicamente en Twitter, mediante la integración con APIs y la aplicación de técnicas avanzadas de procesamiento de lenguaje natural (NLP). La solución se basa en una arquitectura de medallón (Bronze, Silver, Gold) para el procesamiento y refinamiento progresivo de los datos.

El objetivo principal es extraer datos de Twitter en tiempo real, procesarlos, y aplicar diversas técnicas de análisis textual para descubrir patrones, tendencias, sentimientos y tópicos de interés en las conversaciones analizadas.

## Los Objetivos Específicos

- Extraer tweets en tiempo real mediante la API de Twitter (RapidAPI)
- Procesar y limpiar los datos extraídos
- Analizar hashtags y su distribución
- Descubrir tópicos mediante modelado LDA
- Realizar análisis de sentimiento
- Generar resúmenes automáticos del corpus
- Visualizar los resultados de los análisis

## Arquitectura del Proyecto

El proyecto sigue una arquitectura de medallón con tres capas:

### 1. Bronze (Datos en crudo)
- Datos extraídos directamente de la API
- Sin procesamiento, en formato original
- Almacenamiento de dichos datos en CSV

### 2. Silver (Datos procesados)
- Datos limpios y normalizados
- Estructura consistente
- Enriquecimiento básico (extracción de hashtags, análisis de sentimiento)

### 3. Gold (Datos analíticos)
- Insights y resultados de análisis avanzados
- Visualizaciones
- Informes y resúmenes
- Resultados de modelado

## Requisitos y Dependencias

### Librerías Principales
- pandas: Manipulación de datos tabulares
- numpy: Operaciones numéricas
- requests: Conexión a APIs
- nltk: Procesamiento de lenguaje natural
- gensim: Modelado de tópicos
- wordcloud: Generación de nubes de palabras
- matplotlib: Visualizaciones
- textblob: Análisis de sentimiento
- spacy: Procesamiento y parsing de textos
- python-dotenv: Gestión de variables de entorno

### Instalación de Dependencias

```bash
pip install pandas numpy requests nltk gensim wordcloud matplotlib textblob spacy python-dotenv
python -m nltk.downloader punkt stopwords
python -m spacy download en_core_web_sm
```

## Estructura de la Clase DataExtractor

La clase `DataExtractor` centraliza toda la funcionalidad del proyecto:

### Métodos Requeridos por el Enunciado

1. **__init__(source, chunksize)**: Inicializa el extractor con la fuente de datos y configura el entorno.

2. **load_data_api(query, max_results, output_file)**: Conecta con la API de Twitter mediante RapidAPI para extraer tweets en tiempo real según los parámetros de búsqueda especificados.

3. **load_data()**: Carga datos desde un archivo local en formato CSV o JSON.

4. **clean_text(text)**: Limpia y normaliza el texto, eliminando URLs, menciones y caracteres especiales.

5. **extract_hashtags(text)**: Extrae hashtags presentes en el texto mediante expresiones regulares.

6. **analytics_hashtags_extended()**: Realiza un análisis avanzado de hashtags, incluyendo frecuencia global, por usuario y evolución temporal.

7. **generate_hashtag_wordcloud(overall_df, max_words, figsize)**: Genera una nube de palabras a partir del análisis de hashtags.

8. **model_topics(num_topics, passes)**: Aplica el modelo LDA para descubrir tópicos en el corpus de tweets.

9. **analyze_sentiment(method)**: Analiza el sentimiento de cada tweet utilizando TextBlob o spaCy.

10. **parse_and_summarize(summary_ratio)**: Realiza un análisis de parsing y genera un resumen extractivo del corpus.

### Uso de Métodos Adicionales 

Para esta sección me he ayudado de internet para la parte de Gold, es por ello que se han implementado:

11. **generate_advanced_visualizations()**: Genera visualizaciones avanzadas a partir de los análisis de sentimiento y tópicos. Este método fue implementado como una extensión para proporcionar insights más detallados y visuales, aunque no era un requisito explícito del enunciado. Me inspiré en técnicas de visualización de datos de la biblioteca Seaborn y en prácticas recomendadas de dashboards en ciencia de datos.

12. **generate_final_report()**: Genera un informe final en formato markdown con todos los resultados del análisis. Este método fue desarrollado como una mejora adicional para consolidar los hallazgos en un formato legible y estructurado. La idea surge de la práctica común en proyectos de ciencia de datos de generar informes automáticos, similar a lo que ofrecen herramientas como Jupyter Notebooks.

## Flujo de Trabajo

El flujo de trabajo completo se divide en tres fases principales:

### Fase Bronze: Extracción de Datos
1. Configuración de parámetros de búsqueda (palabra clave, número máximo de resultados)
2. Conexión a la API de Twitter mediante RapidAPI (Las contraseñas para acceder a la API se encuentarn en .env)
3. Extracción y almacenamiento de tweets en un archivo CSV

### Fase Silver: Procesamiento y Limpieza
1. Carga de datos desde la fase Bronze
2. Limpieza y normalización de textos
3. Extracción de hashtags y análisis de frecuencia
4. Generación de nubes de palabras

### Fase Gold: Análisis Avanzado
1. Modelado de tópicos con LDA
2. Análisis de sentimiento mediante TextBlob/spaCy
3. Parsing y generación de resúmenes extractivos
4. Visualizaciones avanzadas (evolución de sentimiento, relación entre subjetividad y polaridad)
5. Generación de informe final en markdown

## Detalles de Implementación

### Extracción de Datos (API de Twitter)
- Utiliza la biblioteca `requests` para conectarse a RapidAPI
- Implementa paginación para obtener más tweets de los permitidos por llamada (La API permite como máximo 5)
- Maneja tokens de continuación para navegación entre páginas de resultados
- Gestión segura de claves de API mediante variables de entorno (archivo .env)

### Procesamiento y Limpieza de Textos
- Conversión a minúsculas
- Eliminación de URLs mediante expresiones regulares
- Eliminación de menciones (@usuario)
- Preservación de hashtags para análisis posterior
- Normalización de espacios y caracteres especiales

### Análisis de Hashtags
- Extracción mediante expresiones regulares
- Agrupación por frecuencia global, usuario y fecha
- Generación de nubes de palabras con tamaño proporcional a la frecuencia

### Modelado de Tópicos (LDA)
- Preprocesamiento especializado (eliminación de stopwords, tokens cortos)
- Tokenización mediante NLTK
- Creación de diccionario y corpus en formato bag-of-words
- Entrenamiento del modelo LDA con parámetros configurables
- Extracción y visualización de los tópicos descubiertos

### Análisis de Sentimiento
- Cálculo de polaridad (positivo/negativo) y subjetividad
- Categorización en positivo, negativo o neutral
- Visualización de distribución de sentimientos
- Soporte para múltiples métodos (TextBlob, spaCy)

### Parsing y Resumen
- Concatenación de textos limpios
- Segmentación en oraciones mediante NLTK
- Puntuación de oraciones basada en frecuencia de palabras
- Selección de oraciones más representativas según ratio configurado
- Generación de resumen manteniendo el orden original

### Visualizaciones Avanzadas (Adicional)
- Evolución del sentimiento a lo largo del tiempo
- Relación entre subjetividad y polaridad
- Nubes de palabras separadas por categoría de sentimiento

### Generación de Informe (Adicional)
- Creación de estructura markdown con resultados clave
- Inclusión de estadísticas generales
- Desglose de tópicos descubiertos
- Incorporación del resumen extractivo
- Referencias a visualizaciones generadas

## Uso del Programa

El programa incluye una interfaz de línea de comandos interactiva que permite elegir entre:

1. **Extraer datos de la API de Twitter (fase Bronze)**
   - Solicita término de búsqueda y número máximo de resultados
   - Extrae los tweets y los almacena en la capa Bronze
   - Muestra una muestra de los datos extraídos

2. **Cargar datos existentes y procesar (fases Silver y Gold)**
   - Muestra los archivos disponibles en la capa Bronze
   - Permite seleccionar un archivo específico para procesar
   - Ejecuta el análisis completo (hashtags, sentimiento, tópicos, resúmenes)
   - Genera visualizaciones y almacena resultados en las capas correspondientes


## Conclusiones y Posibles Mejoras

Este proyecto la verdad que me ha parecido super interesante ya que le veo un uso muy cotidiano y que se podría utilizar para multiples sectores.

La verdad sea dicha que me ha parecido un poco complejo ya que en algunas secciones como por ejemplo terminando la capa silver e implementando la capa gold me he encontrado con muchos errores y he tenido que buscar alternativas para poder solucionarlo.

---

## Referencias

- Documentación de RapidAPI: https://rapidapi.com/hub
- TextBlob: https://textblob.readthedocs.io/
- Gensim LDA: https://radimrehurek.com/gensim/models/ldamodel.html
- NLTK: https://www.nltk.org/
- spaCy: https://spacy.io/
- Arquitectura Medallón: https://www.databricks.com/glossary/medallion-architecture
