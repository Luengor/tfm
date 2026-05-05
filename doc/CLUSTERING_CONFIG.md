# Guía de Configuración de Algoritmos de Clustering

Este documento detalla los algoritmos de clustering implementados en el proyecto, sus parámetros configurables a través de los archivos de `whitelist` o `grid`, y su perfil de coste computacional.

Todos los algoritmos se encuentran en `src/src/cluster/cluster.py` y heredan de `ClusteringBase`.

---

## 1. K-Means (`kmeans`)
Algoritmo basado en centroides que particiona los datos en $k$ grupos globulares.

*   **Paradigma:** Partición.
*   **Parámetros:**
    *   `n_clusters` (int, default: `5`): El número de clústeres a generar.
*   **Coste:** $O(N \cdot K \cdot I \cdot D)$ - Muy eficiente.
    *   $N$: número de muestras, $K$: clústeres, $I$: iteraciones, $D$: dimensiones.

## 2. DBSCAN (`dbscan`)
Algoritmo basado en densidad que identifica grupos de cualquier forma y detecta ruido.

*   **Paradigma:** Densidad.
*   **Parámetros:**
    *   `eps` (float, opcional): Radio máximo entre dos muestras para que se consideren vecinas. Si no se proporciona, se calcula automáticamente usando el método del "codo" (elbow method) sobre las distancias de los vecinos más cercanos.
    *   `min_samples` (int, default: `5`): Número mínimo de muestras en un vecindario para que un punto sea considerado "core point".
*   **Coste:** $O(N \log N)$ (promedio) hasta $O(N^2)$ (peor caso).

## 3. HDBSCAN (`hdbscan`)
Extensión jerárquica de DBSCAN que permite encontrar clústeres de diferentes densidades.

*   **Paradigma:** Densidad / Jerárquico.
*   **Parámetros:**
    *   `min_cluster_size` (int, default: `5`): El tamaño mínimo de un grupo para ser considerado clúster.
    *   `max_cluster_size` (int, opcional): El tamaño máximo de un grupo.
*   **Coste:** $O(N \log N)$.

## 4. OPTICS (`optics`)
Similar a DBSCAN pero mejor para manejar bases de datos con densidades variables.

*   **Paradigma:** Densidad.
*   **Parámetros:**
    *   `min_samples` (int, default: `5`): Número de muestras en un vecindario.
    *   `max_eps` (float, default: `inf`): Distancia máxima para considerar vecinos.
    *   `metric` (str, default: `"cosine"`): Métrica de distancia (ej: `"euclidean"`, `"cosine"`).
*   **Coste:** $O(N \log N)$ hasta $O(N^2)$.

## 5. Agglomerative Clustering (`agglomerative`)
Clustering jerárquico ascendente que fusiona grupos sucesivamente.

*   **Paradigma:** Jerárquico.
*   **Parámetros:**
    *   `n_clusters` (int, default: `5`): El número de clústeres final.
    *   `linkage` (str, default: `"ward"`): Criterio de fusión (`"ward"`, `"complete"`, `"average"`, `"single"`).
*   **Coste:** $O(N^2)$ (típicamente).

## 6. Spectral Clustering (`spectral`)
Utiliza la descomposición en valores propios de la matriz de afinidad para reducir dimensiones antes de aplicar K-Means.

*   **Paradigma:** Gráficos / Espectral.
*   **Parámetros:**
    *   `n_clusters` (int, default: `5`): Número de clústeres.
    *   `affinity` (str, default: `"nearest_neighbors"`): Método para construir la matriz de afinidad.
*   **Coste:** $O(N^3)$ - Muy alto para datasets grandes.

## 7. Gaussian Mixture Models (`gmm`)
Modelo probabilístico que asume que los datos son generados por una mezcla de distribuciones Gaussianas.

*   **Paradigma:** Probabilístico.
*   **Parámetros:**
    *   `n_clusters` (int, default: `5`): Número de componentes (clústeres). Internamente mapeado a `n_components`.
    *   `covariance_type` (str, default: `"full"`): Tipo de parámetros de covarianza (`"full"`, `"tied"`, `"diag"`, `"spherical"`).
*   **Coste:** Similar a K-Means pero con más parámetros por clúster.

## 8. Affinity Propagation (`affinity_propagation`)
Envía mensajes entre pares de muestras hasta que convergen en un conjunto de "ejemplares".

*   **Paradigma:** Basado en ejemplares.
*   **Parámetros:**
    *   `damping` (float, default: `0.5`): Factor de amortiguación (entre 0.5 y 1.0) para evitar oscilaciones.
    *   `preference` (float, opcional): Preferencia de cada punto para ser ejemplar. Controla indirectamente el número de clústeres.
*   **Coste:** $O(N^2 \cdot T)$ donde $T$ es el número de iteraciones.

---

## Ejemplo de uso en `grid.json`

```json
{
  "clustering": [
    {
      "type": "kmeans",
      "params": { "n_clusters": 10 }
    },
    {
      "type": "dbscan",
      "params": { "min_samples": 3 }
    },
    {
      "type": "gmm",
      "params": { "n_clusters": 5, "covariance_type": "diag" }
    }
  ]
}
```
