# red-neuronal-lineal-
Trabajo Práctico de la materia Álgebra Lineal Computacional, 2do Cuatri 2025

En este trabajo se comparan cuatro metodologías para obtener el vector de pesos W en un problema de clasificación binaria: Householder (HH), Gram-Schmidt (GS), Ecuaciones Normales (EN) y la Descomposición en Valores Singulares (SVD). Todas estas técnicas provienen de la misma formulación de mínimos cuadrados, pero difieren en la manera en que construyen la pseudo-inversa de X, la base ortonormal o resuelven el sistema, lo que puede generar variaciones en su desempeño numérico. La implementación de las mismas están detalladas en el archivo 'alc.py' adjunto.

El objetivo es evaluar cómo cada método clasifica las observaciones del conjunto de validación y determinar si existen diferencias relevantes entre ellos. Para esto, se analizan dos tipos de resultados: la Accuracy, que mide la proporción total de aciertos, y la matriz de confusión, que permite observar cómo se distribuyen los errores y si alguna clase es más difícil de predecir.

Este análisis no solo permite identificar si un método obtiene un mejor rendimiento, sino también explorar la estabilidad del problema y la sensibilidad de la solución frente a distintas estrategias de descomposición y resolución. De esta forma, el estudio funciona como un pequeño benchmark que muestra si, en la práctica, estas técnicas teóricamente equivalentes producen resultados similares o si aparecen diferencias significativas que deban tenerse en cuenta en aplicaciones reales.
