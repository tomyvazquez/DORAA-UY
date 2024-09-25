# DORAA-UY (Despacho Óptimo de Reactiva mediante Aprendizaje Automático)

Este repositorio forma parte del proyecto de tesis desarrollado en la Facultad de Ingeniería de la UdelaR. El objetivo principal es demostrar cómo las técnicas de aprendizaje automático pueden ser aplicadas para resolver de manera eficiente el problema de Despacho Óptimo de Potencia Reactiva (ORPD) en la red eléctrica uruguaya.

El proyecto busca optimizar el uso de los recursos de la red, mejorando tanto su estabilidad como su eficiencia energética. Como punto de partida, se utilizan las redes de prueba IEEE 30 e IEEE 118, que luego se extienden al modelo de la red eléctrica uruguaya con datos reales proporcionados por el Despacho Nacional de Cargas (DNC).

Este repositorio incluye el código necesario para entrenar modelos de aprendizaje automático que optimizan el ORPD para cada una de las redes. La base de datos y el modelo de la red uruguaya se encuentran disponibles para su descarga [aquí](link a Drive).

A continuación, se brindan instrucciones detalladas sobre cómo entrenar los modelos y una descripción más extensa del proyecto. Para más información técnica y teórica, consultar la documentación completa de la tesis en el siguiente [enlace](link a la documentación).

HABLAR DE QUE EN LOS ULTIMOS AÑOS ESTO COBRO MUCHA IMPORTANCIA Y BLA BLA
DECIR QUE SE PROPONE EXPLORAR TANTO SUP COMO UNSUP Y FCNN Y GNN Y POR QUE
DECIR: A CONTINUACION SE INCLUYE UNA BREVE DESCRIPCION DEL PROYECTO, PARA MAS INFO CONSULTAR LA DOC
DECIR QUE SE TOMA COMO PUNTO DE PARTIDA EL TRABAJO DE DAMIAN OWERKO

### Despacho Óptimo de Potencia Reactiva en la Red Eléctrica Uruguaya utilizando Aprendizaje Automático

Este proyecto se centra en la aplicación de estrategias de aprendizaje automático para resolver el Despacho Óptimo de Potencia Reactiva (ORPD) en la red eléctrica uruguaya, con el objetivo de minimizar pérdidas y asegurar la estabilidad del sistema. Para ello, se utilizan redes neuronales, tanto completamente conectadas (FCNN) como redes neuronales sobre grafos (GNN), aprovechando la capacidad de estas últimas para modelar la estructura de la red eléctrica como un grafo.

Se ataca el problema mediante aprendizaje supervisado y no supervisado. Para el código se utiliza principalmente la librería PandaPower, que es una herramienta de código abierto desarrollada en Python para el análisis de sistemas de energía eléctrica. Entre sus capacidades destacan la ejecución de flujos de carga y la optimización del flujo de potencia (OPF) donde utiliza el optimizador de PyPower por detras. Además permite correr la optimización del despacho de potencia reactiva (ORPD), lo cual es fundamental en esta investigación. Para esto, PandaPower hace uso de PowerModels, optimizador que desarrollado en Julia. Por lo tanto, al correr la optimización, se activa Julia y se corre el algoritmo de optimización.

Además, el proyecto incluye estudios sobre redes de prueba IEEE30 e IEEE118 para comparar las estrategias con casos más generales y analizar la representatividad de los datos sintéticos respecto a los datos reales de la red uruguaya. Uno de los principales aportes de este trabajo es la creación de una base de datos histórica real de la red eléctrica uruguaya, la cual será puesta a disposición pública para fomentar más investigaciones en este ámbito.

#### Planteo del problema de Optimización

El problema de Despacho Óptimo de Potencia Reactiva (ORPD) se plantea como un problema de optimización sujeto a restricciones. Existen varias formulaciones posibles para este problema, pero el objetivo principal es minimizar las pérdidas eléctricas en la red, sin exceder los límites de seguridad y operatividad de la misma. Las restricciones del problema representan las limitaciones físicas que los componentes de la red imponen sobre los valores que pueden tomar las variables. La primera restricción es la ecuación de flujo de potencia, que establece la relación entre los voltajes y las potencias inyectadas en toda la red; por lo tanto, cualquier solución propuesta debe cumplirla. Además, existen restricciones a nivel de barras, donde se limita tanto la magnitud como los ángulos del voltaje. Por otro lado, se establece un límite físico para la corriente que una línea o transformador puede soportar de manera sostenida, el cual debe cumplirse para la corriente que circula en ambos sentidos. En cuanto a las potencias, hay restricciones en la capacidad de generar potencia reactiva tanto para los generadores controladores de voltaje como para los compensadores de reactiva. Finalmente, se imponen restricciones sobre la potencia activa y reactiva del generador de referencia.


### Redes IEEE
Las redes de potencia IEEE son un conjunto de redes de transmisión presentadas como casos de estudio públicamente disponibles por la Universidad de Illinois a principios de los años 90's. Estas consisten en 7 redes que varían en tamaño desde 14 hasta 300 barras. Estas redes proporcionan un entorno controlado y bien documentado para probar y evaluar tanto algoritmos como técnicas de optimización antes de su implementación en la red uruguaya. A su vez, permiten explorar diferentes arquitecturas y estrategias, facilitando la identificación de mejores prácticas y posibles desafíos que podrían surgir en la aplicación a la red uruguaya. Debido a la popularidad de estas redes, la librería PandaPower ya presenta una implementación hecha para ellas, lo que facilita aún más su uso en simulaciones y análisis de redes eléctricas.

En este trabajo se utilizan dos de estas redes; la IEEE30 e IEEE118, que cuentan con 30 y 118 barras respectivamente. La razón de utilizar la IEEE30 es contar con una red de tamaño chico, la cual dado su pequeño tamaño, hace que sea posible interpretar los resultados más intuitivamente, lo que es muy útil a la hora de detectar posibles causas de errores ante imprevistos. Por otro lado, la red IEEE118 es elegida ya que la cantidad de barras es similar a la red uruguaya, la cual tiene 107 barras. De esta forma se asegura que una estrategia operativa para la red 30 mantiene su validez para una red de tamaño similar a la uruguaya.

#### Red Eléctrica Uruguaya

El modelo de la red está armado en PandaPower, el cual se puede cargar mediante:

`net = pp.from_pickle('uru_net.p')`

Este modelo corresponde a una simplificación de la red eléctrica uruguaya, el cual posee un total de 107 barras, de las cuales 95 son en 150 kV y 12 en 500 kV. A su vez se tienen 144 líneas, de las cuales 14 conectan barras de 500 kV y 130 conectan barras de 150 kV. Todos las barras de 500 tienen asociados un transformador, que las conecta a una barra del mismo nombre pero a 150, por lo que hay también 12 transformadores. Cuenta además con 43 generadores, de los cuales 15 son controladores de voltaje, 27 son estáticos y hay 1 generador de referencia. Finalmente, se tienen 55 cargas y 6 compensadores de reactiva.

#### Definición de entradas y salidas
Definir que el objetivo es dado un estado, encontrar los voltajes de los generadores y los q de los shunts

#### Alternativas para la resolución del problema

En este trabajo, se abordan dos metodologías diferentes para la resolución de este problema. En primer lugar, se propone explorar algoritmos de aprendizaje supervisado. Para esto, se utiliza una base de datos con distintos estados de la red junto con sus respectivas variables de control óptimas, halladas previamente mediante el solver de PandaPower. De esta manera, se entrenan modelos FCNN y GNN para aprender este mapeo de entradas y salidas. La limitación que tendrán estos modelos, es que aprenderan a encontrar los mismos óptimos que un optimizador tradicional, por lo cual se esperan resultados con igual o peor desempeño que este en términos de pérdidas en la red. De todas maneras, el entrenar un modelo tiene la ventaja de que los recursos de cómputo se utilizan offline, reduciendo considerablemente tiempos de inferencia. En este caso se utiliza como función de pérdida se utiliza el error cuadrático medio.

Por otro lado y a modo de exploración de distintos resultados, se implementa un modelo de aprendizaje no supervisado, donde la función de pérdida es el lagrangiano del problema de optimización. En este caso, el modelo no intenta copiar salidas de un optimizador, sino que iterativamente va aprendiendo a minimizar las pérdidas en la red, cumpliendo con las restricciones del problema.

En el código se puede ver la implementación de los distintos modelos. DECIR EN QUE DIRECTORIO ESTA CADA COSA 


#### Arquitecturas
Las dos arquitecturas a probar en este trabajo son las redes neuronales completamente conectadas y las redes neuronales para datos en grafos. Estas consisten de una secuencia de capas lineales o convolucionales, intercaladas con funciones de activación. En este trabajo, se utiliza como función de activación Leaky ReLU.
Además de las capas lineales y la no linealidad, se agrega una capa de normalización por lotes (o batch normalization). Esta consiste en agregar un proceso de normalización a la salida de las capas ocultas de la red neuronal, lo cual permite estabilizar y acelerar el proceso de entrenamiento.

Finalmente se tiene el uso de máscara a la salida del predictor. La salida óptima consiste en los voltajes en generadores controlables y reactiva en los compensadores para todas las barras, rellenando con 0 aquellas barras que no tienen conectado este elemento. Por lo tanto, utilizar una máscara que multiplique la salida del predictor, llevando a cero las salidas que son siempre nulas, evita que este tenga que utilizar parte de su capacidad en aprender a llevar estas entradas a cero, potencialmente empeorando el desempeño.


### Datos
#### IEEE 
En cuanto a las redes IEEE, ninguna de las dos ofrece un histórico de generación o demanda de potencia para entrenar los modelos. Esto implica que para trabajar con estas es necesario generar una base de datos sintética. Los datos que son necesarios simular son aquellos que se toman como entrada al problema. Se deben generar valores de potencia activa y reactiva demandada, y valores de potencia activa generada (se omiten los generadores estáticos ya que esta red no presenta generadores estáticos).

Como metodología utilizada para la generación de datos sintéticos se realiza un proceso  similar a los utilizados en otros trabajos que abordan este problema con aprendizaje automático. Esta consiste en, para cada nodo, tomar valores nominales de potencia activa y reactiva de todos los nodos. Estos valores nominales son información prevista por la red. Luego, a partir de estos valores, se genera una distribución de generación/demanda, que consiste en una uniforme entre un 0.7 y 1.3 del valor de referencia. A partir de estos valores se halla el óptimo mediante la función `net.acopf()

#### Resultados


### Reproducción de resultados
En las carpetas .... ejecutar el codigo `train.py` indicando qué tipo de entrenamiento

#### Creaación del environment
- Instalación de Julia con herramienta PyCall. Seguir las instrucciones en este [link])(https://pandapower.readthedocs.io/en/v2.6.0/opf/powermodels.html)
- Crear un entorno para python e instalar todas las dependencias mediante el siguiente comando:
` pip install requirements.txt`
