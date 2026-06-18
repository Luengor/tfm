# Prefacio
## Título
Hola, soy... y voy a defender mi trabajo ...

## Índice 
La presentación sigue la misma estructura que el artículo escrito:
Intro con arte
Conjunto de datos
Metodología, se explica el sistema implementado
Resultados
Conclusiones

# Introducción
## Contexto
Empezamos con un poco de contexto.

El grafiti no autorizado es un fenómeno urbano conocido por todos y que
requiere un esfuerzo constante por parte de las autoridades para su control.

El seguimiento y clasificación manual de los grafitis es un proceso costoso e
inconsistente, lo que hace que la automatización de cualquier parte de este
proceso sea de gran interés.

## Problema y objetivo
Para ser más concretos, el problema que nos ocupa es: dado un banco de
fotografías de grafitis, ¿cómo podemos agruparlo por similitud de forma
automática?

Para esto, se desarrolla un sistema modular que permite la combinación de
diferentes técnicas de segmentación, extracción de características, técnicas de
reducción de dimensionalidad y algoritmos de agrupamiento. Con este sistema,
se busca encontrar la combinación de técnicas que mejor se adapte a este problema,
evaluando la calidad y coste de cada elemento del sistema.

## Trabajos relacionados
No existe mucha literatura sobre la aplicación de técnicas informáticas al
estudio de grafitis. Algunos de los pocos trabajos que se han encontrado
cubren puntos similares a este:
 - Fogaca trata la segmentación de grafitis y la clasificación de los mismos en
   grafiti o no grafiti.
 - Tokuda propone un sistema para detectar grafitis automáticamente desde
   Street View para cuantificar la presencia de grafitis en las ciudades.
 - Miguel Gracía desarrollan un sistema para agrupar grafitis según sus
   colores dominantes.
Este trabajo se diferencia de los anteriores en que se centra en la agrupación
de grafitis por similitud, con un énfasis en la evaluación de la calidad y
coste de cada elemento del sistema.

# Datos
## Conjunto de datos
Para el proyecto, disponemos de un banco principal de 6600 fotografías de
grafiti en Salamanca, capturadas principalmente en 2022. Las fotografías son
bastante distintas entre si: hay bastante variedad de perspectivas, iluminación
y escenas, algunos grafitis son sobre muros, otros sobre papeleras o postes.
Muchas de las fotografías contienen varios grafiti.

Además, se cuenta con un banco auxiliar de más de 1000 fotografías de Cuenca,
cedidas por la iniciativa StopGrafiti. Estas se utilizan solo para el ajuste
fino para evitar el solapamiento entre datos de entrenamiento y evaluación.

## Anotaciones y subconjuntos
Ninguno de los bancos cuenta con anotaciones, por lo que ha sido necesario
crear varios conjuntos anotados para poder evaluar la calidad de los
resultados. Puesto que muchas de las fotografías contienen varios grafitis, se
ha decidido recortar los grafitis de las fotografías y crear un conjunto de
datos de grafitis anotados.

Los recortes se anotan según dos listas de etiquetas: una de ellas para el
estilo del grafiti, que contiene las 4 categorías mostradas aquí y otra
para el "autor" del grafiti, que contiene unos \~80 autores distintos.
Aquí en la tabla se muestran la procedencia y tamaño de cada uno de los
conjuntos utilizados.

# Metodología
## Sistema: dos fases acopladas por el almacenamiento
Ahora pasamos a la metodología. El sistema desarrollado consta de dos fases
bien diferenciadas, que se acoplan mediante el paso por el almacenamiento.

La primera fase, la ingesta, se realiza de forma incremental y consiste en la
segmentación y obtención de la representación numérica de cada grafiti para
luego almacenarlos.

La segunda fase, la búsqueda por similitud, se realiza de forma puntual y
consiste en la obtención de la representación numérica de cada grafiti para
agruparlos y buscar los más similares.

A continuación se explican con más detalle cada una de las fases.

## Segmentación
La etapa de segmentación consiste en obtener de manera automática los recortes
de los grafitis a partir de las fotografías. Para esto, se utiliza un modelo de
YOLO ajustado sobre algunos de los recortes anotados para detectar
específicamente grafitis.

La fase procede como se muestra en el diagrama: los grafitis se detectan, se
fusionan los recortes que cumplen ciertos criterios, se añade un ligero margen
a los recortes y se almacenan en el banco de grafitis.

Esta fase es opcional, y en el caso de prescindir de ella se hace uso de la
fotografía completa.

## Extracción de características
Los recortes de grafiti se alimentan a una red neuronal de la que luego se
extrae una representación numérica de cada grafiti. Para esto, se utilizan
distintos tipos de redes neuronales pre-entrenadas como ResNet, MobileNet o 
transformers como DINOv2 o CLIP.

## Extracción de características: ajuste al dominio
Para mejorar la calidad de las representaciones numéricas, se realiza un ajuste
fino de algunas de las redes neuronales. Esto se hace entrenando una pequeña
red neuronal que se situa al final de la red pre-entrenada y que se entrena
sobre los conjuntos anotados de grafitis. Se entrenan 4 cabezas: 2 para estilo
y 2 para autoría, una en MobileNet y otra en DINOv2.

## Reducción de dimensionalidad
Los vectores de características obtenidos por la red neuronal llegan a contener
hasta 2048 dimensiones. Esto hace que el coste de agrupamiento y búsqueda por
similitud sea muy alto, además de que la calidad de los resultados se ve
afectada por un suavizado en la distancia entre vectores, conocido como la
maldición de la dimensionalidad como se puede ver en el gráfico.

Para esto, la etapa opcional de reducción reduce el tamaño de estos vectores a
uno más manejable mediante distintas técnicas como PCA, Isomap o UMAP.

## Agrupamiento

Por último, para el agrupamiento de los grafitis, se utilizan distintos algoritmos
de agrupamiento de varias familias: ...

# Experimentos y resultados
## Diseño experimental
Pasando a los experimentos, se han realizado varios barridos de combinaciones
de técnicas, intentando siempre variar un solo eje para poder evaluar el efecto
de cada técnica de forma aislada. Podemos distinguir tres grupos de
experimentos: los que evalúan la calidad general de cada componente del
sistema, los que evalúan el coste de cada componente y los experimentos
supervisados.

Como métricas de calidad, distinguimos entre métricas intrínsecas, que evalúan
la calidad de los resultados sin necesidad de anotaciones, que son las que se
utilizarán para la mayoría de experimentos y métricas extrínsecas, que comparan
los resultados con anotaciones. Para el coste, se mide el tiempo de ejecución y
el consumo de memoria.

## Resultados: segmentador (E5)
Siguiendo el flujo de la metodología, en este experimento se evalúa la calidad
del segmentador en el pipeline completo. Aunque los resultados del segmentador
son aparentemente buenos viendo las fotografías segmentadas, el efecto que
tiene sobre el sistema completo es negativo.

El uso de segmentador empeora la calidad de los resultados, se use la
configuración de segmentador que se use. Viendo los resultados del
entrenamiento del segmentador, recall X y mAP X, se puede deducir que la razón
de este efecto es la variabilidad de los recortes obtenidos. El modelo pierde 1
de cada 4 grafitis, y los recortes obtenidos no suelen ajustarse correctamente
al grafiti. Esta variabilidad extra provoca naturalmente un aumento del ruido,
una peor agrupación y una ingesta mucho más costosa.

Por esto, se decide prescindir del segmentador y trabajar con las fotografías
completas.

## Resultados: extractores de características (E1)

## Resultados: validación supervisada (E8) y elección del extractor

## Resultados: reducción de dimensionalidad (E2)

## Resultados: algoritmo de agrupamiento (E3)

## Resultados: coste del sistema completo (E7)

## Resultados: búsqueda por similitud (E6, E9)

# Conclusiones
## Conclusiones

