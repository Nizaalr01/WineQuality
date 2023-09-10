# Proceso de Análisis de Datos y Modelado en Python

En este documento, se describe el proceso de análisis de datos y modelado en Python utilizando un conjunto de datos de calidad de vino tinto y blanco. Se están utilizando tres modelos de aprendizaje automático: Regresión Logística, Árbol de Decisión y Random Forest, para predecir si un vino es de buena o mala calidad. El objetivo principal es entrenar los modelos en un dataset de vino tinto y posteriormente comprobar si se pueden basar en las características del tinto para clasificar la calidad del vino blanco.

## Carga de Datos

Los datos utilizados en este proyecto provienen del siguiente enlace: [Wine Quality Dataset](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset/code). El conjunto de datos contiene las siguientes variables de entrada basadas en pruebas fisicoquímicas:

1. Fixed acidity
2. Volatile acidity
3. Citric acid
4. Residual sugar
5. Chlorides
6. Free sulfur dioxide
7. Total sulfur dioxide
8. Density
9. pH
10. Sulphates
11. Alcohol

Y una variable de salida basada en datos sensoriales:

12. Quality (puntuación entre 0 y 10)

Se descargaron dos conjuntos de datos: uno para vino tinto y otro para vino blanco. Luego, se realizaron las siguientes tareas:

- Importación de bibliotecas necesarias, incluyendo pandas, pandas_profiling, seaborn y scikit-learn.
- Carga de datos en DataFrames.
- Realización de un análisis exploratorio inicial, que incluyó estadísticas descriptivas y la eliminación de valores duplicados en el conjunto de datos de vino tinto.

## Exploración Inicial

Antes de comenzar con el modelado, se realizó un análisis exploratorio de datos (EDA) en ambos conjuntos de datos (vino tinto y blanco) para comprender la naturaleza de las variables y cualquier posible desafío. Algunos de los aspectos destacados incluyen:

- Estadísticas descriptivas para cada variable, proporcionando una visión general de su distribución y dispersión.
- Detección y eliminación de valores duplicados en el conjunto de datos de vino tinto para mejorar la calidad de los modelos.

**Clasificación de Calidad**

La clasificación de "buena" y "mala" calidad se basó en el análisis del boxplot de calidad y alcohol. Se observó un codo claro en la distribución de la calidad alrededor del valor 6. Por lo tanto, se etiquetaron como "1" aquellos vinos con una puntuación de calidad mayor que 6, considerándolos de "buena" calidad, y como "0" aquellos con una puntuación menor o igual a 6, considerándolos de "mala" calidad.

## Preprocesamiento de Datos

### División de Datos

Los datos se dividieron en conjuntos de entrenamiento y prueba, asignando el 70% de los datos para el entrenamiento y el 30% para las pruebas. Esta división es crucial para evaluar el rendimiento de los modelos.

## Modelado

### Regresión Logística

#### Modelo Default

El primer paso en la implementación de la Regresión Logística fue utilizar el modelo default proporcionado por la librería. Este modelo proporcionó una precisión inicial del 89.46%.

#### Optimización

Llevé a cabo una búsqueda exhaustiva de hiperparámetros para el modelo de Regresión Logística mediante GridSearchCV. La cuadrícula de hiperparámetros definida incluyó valores para el parámetro de regularización 'C' y el tipo de regularización 'penalty'. Posteriormente, configuré el modelo de Regresión Logística con valores específicos de hiperparámetros, como el solucionador 'liblinear', la tolerancia y el número máximo de iteraciones. Luego, empleé GridSearchCV para encontrar la combinación óptima de hiperparámetros mediante validación cruzada en 5 divisiones, utilizando 'accuracy' como métrica de evaluación. Una vez encontrado el mejor modelo, lo entrené en el conjunto de entrenamiento y realicé predicciones en el conjunto de prueba. Finalmente, calculé y presenté la precisión en el conjunto de prueba. 
Aumentó al 89.71%.

### Árbol de Decisión

#### Modelo Default

El Árbol de Decisión se implementó inicialmente con los parámetros predeterminados de la librería. Esto resultó en una precisión inicial del 82.35%.

#### Optimización

Realicé ajustes significativos para optimizar su rendimiento. Implementé una búsqueda de hiperparámetros utilizando GridSearchCV, lo que me permitió encontrar la combinación óptima de valores de hiperparámetros, incluyendo el criterio de división, la profundidad máxima del árbol, el número mínimo de muestras para dividir un nodo y el número mínimo de muestras en un nodo hoja. Esta búsqueda exhaustiva ayudó a maximizar la precisión del modelo en la clasificación de la calidad del vino. Además, calculé y presenté métricas adicionales, como la matriz de confusión y el informe de clasificación.
Aumentando la precisión al 85.78%.

### Random Forest

#### Modelo Default

El modelo de Random Forest se implementó por primera vez con los valores predeterminados de la librería. Inicialmente, obtuvo una precisión del 90.69%.

#### Optimización

Agregué una búsqueda de hiperparámetros utilizando GridSearchCV para encontrar la combinación óptima de valores de hiperparámetros que mejoren la precisión del modelo Random Forest. También calculé y presenté métricas adicionales, como la matriz de confusión y el informe de clasificación, para obtener una evaluación más completa del rendimiento del modelo. Además, incorporé la visualización de la importancia de las características, lo que permite identificar qué atributos influyen más en las predicciones del modelo. 
Alcanzando una precisión del 90.93%.

## Evaluación en un Nuevo Conjunto de Datos

Se probaron los tres modelos entrenados en un nuevo conjunto de datos de vino blanco:

- Se aplicó la misma transformación a la variable objetivo "quality" en el conjunto de vino blanco.
- Se realizaron predicciones utilizando los modelos y se calcularon las precisiones en el conjunto de datos de vino blanco.
- Se mostraron las matrices de confusión y los puntajes F1 para cada modelo en el conjunto de datos de vino blanco.

## Resultados y Conclusiones

### Regresión Logística

- La Regresión Logística entrenada en el conjunto de datos de vino tinto alcanzó una precisión del 89.71% después de la optimización de hiperparámetros.
- Al aplicar este modelo al conjunto de datos de vino blanco, se obtuvo una precisión del 78.83% en la clasificación de calidad del vino blanco.
- El modelo de Regresión Logística tiene un puntaje F1 de 0.13 en el conjunto de datos de vino blanco.
- Esto sugiere que el modelo de Regresión Logística funciona bien en el conjunto de datos de vino tinto, pero su rendimiento se reduce ligeramente en el conjunto de datos de vino blanco.

### Árbol de Decisión

- El Árbol de Decisión entrenado en el conjunto de datos de vino tinto alcanzó una precisión del 85.78% después de la optimización de hiperparámetros.
- Al aplicar este modelo al conjunto de datos de vino blanco, se obtuvo una precisión del 76.85% en la clasificación de calidad del vino blanco.
- El modelo de Árbol de Decisión tiene un puntaje F1 de 0.26 en el conjunto de datos de vino blanco.
- Aunque el Árbol de Decisión funciona razonablemente bien en el conjunto de datos de vino tinto, su rendimiento se reduce significativamente en el conjunto de datos de vino blanco.

### Random Forest

- El modelo de Random Forest entrenado en el conjunto de datos de vino tinto alcanzó una precisión del 90.93% después de la optimización de hiperparámetros.
- Al aplicar este modelo al conjunto de datos de vino blanco, se obtuvo una precisión del 78.26% en la clasificación de calidad del vino blanco.
- El modelo de Random Forest tiene un puntaje F1 de 0.09 en el conjunto de datos de vino blanco.
- Si bien el modelo de Random Forest funcionó muy bien en el conjunto de datos de vino tinto, también muestra una disminución en el rendimiento en el conjunto de datos de vino blanco.

## Conclusiones Generales

- Los modelos entrenados en el conjunto de datos de vino tinto tienen un rendimiento generalmente mejor que en el conjunto de datos de vino blanco. Esto sugiere que las características y relaciones en los dos conjuntos de datos pueden diferir.
- La Regresión Logística y el Árbol de Decisión tienen un rendimiento más bajo en el conjunto de datos de vino blanco en comparación con el Random Forest.
- El Random Forest parece ser el modelo más robusto en esta tarea de clasificación de calidad del vino, aunque su puntaje F1 en el conjunto de datos de vino blanco es bajo.
- Se pueden considerar futuras investigaciones para mejorar el rendimiento de los modelos en el conjunto de datos de vino blanco, como la adquisición de más datos o la exploración de características adicionales.
