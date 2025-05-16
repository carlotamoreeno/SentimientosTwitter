# Análisis de Sentimiento y Redes Sociales en Twitter con LLMs

## ¿De qué va esta práctica?

En esta práctica hemos ampliado el proyecto de análisis de sentimientos en Twitter para incluir dos funcionalidades nuevas:

1. **Análisis de Redes Sociales**: He construido grafos que muestran cómo interactúan los usuarios mencionándose entre sí, descubriendo quiénes son los más influyentes y qué comunidades se forman.

2. **Análisis con Modelos de Lenguaje (LLMs)**: Utilizando la información de las redes sociales, alimento un modelo de lenguaje (como el Gemma de Google o en mi caso GPT-4o de OpenAI) para que interprete los patrones de interacción y dé insights valiosos. Como no podia utilizar un modelo en local y como ya tenia acceso a la API de OpenAI, he pasado a utilizar el modelo 4o que es potente, rapido y su coste es minimo.

El proyecto entero funciona como una especie de "pipeline de datos": Extraemos tweets en tiempo real, los procesamos, analizamos, y finalmente obtenemos conclusiones mediante técnicas avanzadas.

## ¿Qué he aprendido e implementado?

### Análisis de Redes con NetworkX 🕸️

He implementado un sistema que:

- **Construye grafos de interacción**: Cada nodo es un usuario y las conexiones representan menciones entre ellos
- **Identifica usuarios clave**: Mediante métricas como centralidad de grado, intermediación y cercanía
- **Detecta comunidades**: Utilizando el algoritmo de Louvain para encontrar grupos de usuarios que interactúan más entre sí
- **Visualiza la red**: Con nodos de tamaño proporcional a su importancia y colores para identificar comunidades

El reto más grande fue detectar menciones implícitas y explícitas, teniendo que interpretar cómo los usuarios se refieren a otros incluso sin usar el símbolo "@" por si no había.

### Integración con Modelos de Lenguaje 🤖

La parte que más me ha gustado ha sido conectar el análisis de redes con modelos de lenguaje:

- **Generación automática de prompts**: El sistema formula preguntas basadas en los patrones detectados en la red
- **Flexibilidad de modelos**: Puedo elegir entre usar el modelo Gemma-2-2b-it localmente o conectar con GPT-4o a través de la API de OpenAI
- **Análisis interpretativo**: El LLM proporciona insights sobre la estructura de la red, los roles de los usuarios y recomendaciones estratégicas

Me fue complicado optimizar los prompts para que el modelo realmente entendiera el contexto y proporcionara análisis útiles en lugar de respuestas genéricas.

## Arquitectura: Bronze, Silver y Gold 

Todo el proyecto sigue la arquitectura medallón que implementé en las prácticas anteriores:

1. **Bronze**: Datos crudos extraídos de la API de Twitter
2. **Silver**: Datos procesados, limpios y estructurados
3. **Gold**: Visualizaciones, informes y resultados de análisis avanzados

Ahora en la capa Gold también se guardan:
- Grafos de interacción en formato GraphML
- Visualizaciones de la red y comunidades
- Métricas de red en formato JSON
- Prompts generados y respuestas de los LLMs

## Mis mayores dificultades y soluciones

1. **Extracción de menciones**: Al principio no lograba detectar suficientes conexiones entre usuarios. Tuve que implementar un sistema que no solo busca el símbolo "@" sino que también detecta menciones implícitas.

2. **Integración con LLMs locales**: Ejecutar modelos como Gemma en CPU no me fue posible y por eso opté a utilizar la API de OpenAI.

3. **Visualización de redes complejas**: Las redes con muchos nodos se volvían ilegibles y por eso implementé filtros y escalado de nodos para mejorar la visualización.

## Cómo ejecutar el proyecto

El sistema tiene un menú interactivo con 4 opciones:

1. **Extraer datos de Twitter**: Conecta con la API y guarda tweets sobre un tema
2. **Procesar datos existentes**: Limpia y analiza datos ya extraídos
3. **Análisis de red social**: Construye y analiza grafos de interacción
4. **Análisis con LLM**: Genera prompts y obtiene interpretaciones avanzadas

Para usar la API de OpenAI, añadí soporte para configurar la clave a través del archivo .env:
```
RAPIDAPI_KEY="tu_clave_aqui"
OPENAI_API_KEY="tu_clave_aqui"
```

## Conclusiones y reflexiones personales

Este proyecto me ha permitido entender mucho mejor cómo funcionan las interacciones en redes sociales y cómo los LLMs pueden ayudar a interpretar grandes volúmenes de datos.

Me sorprendió descubrir que incluso con pocos datos se pueden identificar patrones de influencia claros.

Creo que este tipo de análisis tiene aplicaciones muy grandes como mencionamos en clase como puede ser en marketing, investigación social, detección de tendencias y gestión de crisis.

## Requisitos del sistema

Para ejecutar este proyecto se necesitan las siguientes dependencias:
- Python 3.8+
- Pandas, NumPy, Matplotlib
- NLTK, Gensim, TextBlob
- NetworkX, Community (python-louvain)
- Transformers, Torch
- SpaCy
- Requests, Python-dotenv

# Librerías Principales

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

El archivo requirements.txt contiene todas las dependencias exactas.
