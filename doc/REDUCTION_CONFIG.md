# Guía de Configuración de Algoritmos de Reducción de Dimensiones

Este documento detalla los algoritmos de reducción de dimensiones implementados, sus parámetros y su perfil de coste.

Todos los algoritmos se encuentran en `src/src/reduction/` y heredan de `ReductionBase`.

---

## 1. PCA (`pca`)
Reducción de dimensiones lineal mediante la maximización de la varianza explicada.

*   **Paradigma:** Proyección Lineal.
*   **Parámetros:**
    *   `n_components` (int, default: `50`): Número de dimensiones finales.
*   **Coste:** $O(D^2 \cdot N + D^3)$. Muy eficiente para reducir de cientos de dimensiones a decenas.

## 2. UMAP (`umap`)
Técnica de reducción no lineal basada en topología que preserva tanto la estructura local como la global.

*   **Paradigma:** Manifold Learning (No lineal).
*   **Parámetros:**
    *   `n_components` (int, default: `2`): Dimensiones finales.
    *   `n_neighbors` (int, default: `15`): Tamaño del vecindario local.
    *   `min_dist` (float, default: `0.1`): Controla cómo de empaquetados están los puntos.
    *   `metric` (str, default: `"cosine"`): Métrica de distancia.
*   **Coste:** $O(N \cdot \log N)$. El más rápido entre los no lineales modernos.

## 3. Isomap (`isomap`)
Algoritmo de Manifold Learning que preserva las distancias geodésicas entre todos los puntos.

*   **Paradigma:** Manifold Learning (Geodésico).
*   **Parámetros:**
    *   `n_components` (int, default: `2`): Dimensiones finales.
    *   `n_neighbors` (int, default: `5`): Número de vecinos para construir el grafo.
*   **Coste:** $O(N^3)$ (debido al cálculo de caminos más cortos). Útil para entender estructuras geométricas complejas.

## 4. Kernel PCA (`kernel_pca`)
Extensión no lineal de PCA que utiliza funciones kernel para proyectar datos en espacios de mayor dimensión.

*   **Paradigma:** Proyección No Lineal (Kernel Trick).
*   **Parámetros:**
    *   `n_components` (int, default: `50`): Dimensiones finales.
    *   `kernel` (str, default: `"rbf"`): Tipo de kernel (`"linear"`, `"poly"`, `"rbf"`, `"sigmoid"`, `"cosine"`).
*   **Coste:** $O(N^3)$. Excelente para datos que no son linealmente separables.

## 5. Identity (`identity`)
No realiza ninguna reducción, devuelve los embeddings originales.

*   **Paradigma:** Ninguno (Baseline).
*   **Coste:** $O(1)$. Útil como grupo de control en los experimentos.

---

## Ejemplo de uso en `grid.json`

```json
{
  "reduction": [
    { "type": "pca", "params": { "n_components": 128 } },
    { "type": "umap", "params": { "n_components": 2 } },
    { "type": "isomap", "params": { "n_components": 2, "n_neighbors": 10 } }
  ]
}
```
