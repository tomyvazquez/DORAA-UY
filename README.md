# DORAA-UY (Despacho Óptimo de Reactiva mediante Aprendizaje Automático)

Este repositorio forma parte del proyecto de tesis desarrollado en la Facultad de Ingeniería de la UdelaR. El objetivo principal es demostrar cómo las técnicas de aprendizaje automático pueden ser aplicadas para resolver de manera eficiente el problema de Despacho Óptimo de Potencia Reactiva (ORPD) en la red eléctrica uruguaya.

El proyecto busca optimizar el uso de los recursos de la red, mejorando tanto su estabilidad como su eficiencia energética. Como punto de partida, se utilizan las redes de prueba IEEE 30 e IEEE 118, que luego se extienden al modelo de la red eléctrica uruguaya con datos reales proporcionados por el Despacho Nacional de Cargas (DNC).

Este repositorio incluye el código necesario para entrenar modelos de aprendizaje automático que optimizan el ORPD para cada una de las redes. La base de datos y el modelo de la red uruguaya se encuentran disponibles para su descarga [aquí](link a Drive).

A continuación, se brindan instrucciones detalladas sobre cómo entrenar los modelos y una descripción más extensa del proyecto. Para más información técnica y teórica, consultar la documentación completa de la tesis en el siguiente [enlace](link a la documentación).

### Despacho Óptimo de Potencia Reactiva en la Red Eléctrica Uruguaya utilizando Aprendizaje Automático

Este proyecto se centra en la aplicación de estrategias de aprendizaje automático para resolver el Despacho Óptimo de Potencia Reactiva (ORPD) en la red eléctrica uruguaya, con el objetivo de minimizar pérdidas y asegurar la estabilidad del sistema. Para ello, se utilizan redes neuronales, tanto completamente conectadas (FCNN) como redes neuronales sobre grafos (GNN), aprovechando la capacidad de estas últimas para modelar la estructura de la red eléctrica como un grafo.

Se ataca el problema mediante aprendizaje supervisado y no supervisado. Para el código se utiliza principalmente la librería PandaPower, que es una herramienta de código abierto desarrollada en Python para el análisis de sistemas de energía eléctrica. Entre sus capacidades destacan la ejecución de flujos de carga y la optimización del flujo de potencia (OPF) donde utiliza el optimizador de PyPower por detras. Además permite correr la optimización del despacho de potencia reactiva (ORPD), lo cual es fundamental en esta investigación. Para esto, PandaPower hace uso de PowerModels, optimizador que desarrollado en Julia. Por lo tanto, al correr la optimización, se activa Julia y se corre el algoritmo de optimización.

Además, el proyecto incluye estudios sobre redes de prueba IEEE30 e IEEE118 para comparar las estrategias con casos más generales y analizar la representatividad de los datos sintéticos respecto a los datos reales de la red uruguaya. Uno de los principales aportes de este trabajo es la creación de una base de datos histórica real de la red eléctrica uruguaya, la cual será puesta a disposición pública para fomentar más investigaciones en este ámbito.

#### Planteo del problema de Optimización

El problema de Despacho Óptimo de Reactiva se plantea como un problema de optimización bajo restricciones. Existen varias alternativas para el planteo de dicho problema, pero lo que se busca es minimizar las pérdidas eléctricas en la red, sujeto a no sobrepasar límites se seguridad y funcionamiento de la red.

#### Cómo se conforma el modelo de la red

Decir que se teinen buses, cargas, generadores, sgen

#### Alternativas para la resolución del problema

En primer lugar, se ataca el problema mediante aprendizaje supervisado. Para esto, se dispone de un conjunto de datos de entrenamiento con sus respectivas salidas objetivo. Los datos constan de configuraciones de la red (

### Reproducción de resultados
En las carpetas .... ejecutar el codigo `train.py` indicando qué tipo de entrenamiento

#### Creaación del environment
- Instalación de Julia con herramienta PyCall. Seguir las instrucciones en este [link])(https://pandapower.readthedocs.io/en/v2.6.0/opf/powermodels.html)
- Crear un entorno para python e instalar todas las dependencias mediante el siguiente comando:
` pip install requirements.txt`
