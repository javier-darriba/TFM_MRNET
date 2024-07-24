# TFM_MRNET

## Descripción
Este repositorio alberga las implementaciones secuenciales y paralelas del proyecto MRNET. Incluye versiones que aprovechan la computación paralela mediante multihilo y CUDA, basadas en las bibliotecas FEAST y ParallelFST, esta última diseñada para optimizar la inferencia de redes genéticas. Estas implementaciones están preparadas para funcionar tanto en sistemas multinúcleo de memoria compartida como en GPUs de NVIDIA.

## Prerrequisitos
Para compilar y ejecutar los programas en este repositorio, necesitarás las siguientes herramientas con las versiones especificadas (puede que funcionen versiones anteriores si son compatibles):

- g++ 10.1.0
- nvcc 11.2

## Compilación
- **Secuencial:** Para compilar la versión secuencial del código, puedes usar el siguiente comando:

  g++ -o mrnet main.cpp
- **OpenMP:**

  Para la versión que paraleliza los calculos internos de mRMR:

  g++ -fopenmp -o mrnet inner_loop.cpp

  Para la versión que paraleliza el bucle externo MRNET:

  g++ -fopenmp -o mrnet outer_loop.cpp

- **CUDA:**

  Para una GPU A100:

  nvcc -gencode=arch=compute_80,code=sm_80 -o mrnet your_source_file.cu

  Para una GPU T4:

  nvcc -gencode=arch=compute_75,code=sm_75 -o mrnet your_source_file.cu

## Uso
Para ejecutar el programa, es necesario proporcionar una ruta al archivo de entrada que
contenga los datos de la matriz de características, utilizando el argumento -i, y especificar la
ruta donde se desea generar el archivo de salida para los resultados, utilizando el argumento
-o.

De manera opcional, el usuario también tiene la posibilidad de incluir el argumento -t.
Este argumento permite que el programa muestre por pantalla el tiempo total de ejecución,
así como los tiempos parciales de varias secciones específicas del programa.

### Formato de entrada
El archivo de entrada debe estar formateado de la siguiente manera:
- La primera fila debe incluir las etiquetas de las muestras.
- Las filas subsecuentes deben comenzar con la etiqueta de la característica, seguida por los valores numéricos correspondientes a cada muestra.
- Si las filas contienen un elemento adicional antes de la etiqueta de la característica, se proporciona un script en Python para eliminar este primer elemento de cada fila.

### Formato de salida
El archivo de salida contiene una matriz de dimensiones número_de_características x número_de_características, donde cada fila representa los resultados del análisis mRMR de esa característica contra las demás, mostrando sus puntuaciones. Las características se listan en el mismo orden que en el archivo de entrada. Las comparaciones de una característica consigo misma se marcan con "NA".

### Comandos para ejecución
**Programa principal:**
  
  ./mrnet -i ruta/a/fichero_entrada.txt -o ruta/a/fichero_salida.txt [-t]

**Ajuste del archivo de entrada:**
  
  python remove_first_element.py ruta/a/fichero_entrada.txt
