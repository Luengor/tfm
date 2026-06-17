# Nota de preparación: correlación silueta ↔ ARI

Guion para defender la subsección "Correlación interna-externa" (§ resultados) y
responder preguntas del jurado sobre por qué se usa la silueta y por qué se
eligió la cabeza de estilo de DINOv2.

---

## La idea en una frase

La silueta sirve para **ordenar configuraciones dentro de una misma
representación y a granularidad de estilo**, pero **no sirve para elegir el
extractor** ni para tareas de grano fino (autoría). No es una contradicción: son
preguntas distintas.

---

## Los dos regímenes (no confundirlos)

| Régimen | Qué se compara | Resultado | Lectura |
|---|---|---|---|
| **A — mismo dato, por celda** | silueta y ARI calculados sobre los *mismos* recortes de E8a | Pearson **+0.78** (k fijo, n=48), **+0.58** (todas, n=63) | La silueta ordena configuraciones a granularidad de estilo |
| **B — entre datasets** | silueta de E1 (set completo, sin etiquetas) vs ARI de E8a | Spearman **−0.62** | El liderazgo intrínseco en E1 NO se traslada a la tarea supervisada |

- El **+0.78 reproduce exacto** desde los JSON. Número sólido.
- El régimen B (−0.62) es el "flip" que parecía contradicción: mobilenet/resnet
  ganan en silueta sobre el set completo, pero pierden en ARI de estilo. **No
  contradice A** — son cosas distintas, y el documento ya lo separa.

## Por qué pasa el flip (la frase clave)

> La silueta mide **cohesión de apariencia** (textura, color, composición), no
> **identidad estilística**.

Las CNN de ImageNet producen blobs geométricamente limpios sobre rasgos visuales
de bajo nivel → silueta alta, pero el eje que separan no es el estilo → ARI baja.
La cabeza de estilo separa por estilo → ARI alta, aunque su geometría global sea
menos compacta en el set heterogéneo completo.

**Regla mental:** la silueta premia *que haya* estructura; el ARI pregunta si es
*la estructura correcta*. Un extractor genérico tiene mucha estructura, de la
clase equivocada.

## Por qué en autoría (E8b) el signo se invierte (−0.54) y no importa

Argumento **estructural**, no estadístico: con 87 clases, el único modo de sacar
ARI no trivial es k≈87. Pero 87 clústeres diminutos **no pueden ser cohesivos** →
silueta ≈ 0 por construcción. Las celdas de silueta alta son particiones
degeneradas de HDBSCAN (k=2,3) con ARI nulo. Las dos métricas piden un k
incompatible cuando la verdad es de grano fino. → la silueta no "falla", es que
mide algo geométricamente incompatible con la tarea a esa granularidad.

(El que "todo el ARI vive en banda de azar" es solo el motivo de descartar el
número; el verdadero motor del −0.54 es la incompatibilidad de k. Liderar con
esto si preguntan.)

## ¿CH resuelve el problema de la silueta? — NO

Sobre los mismos datos de E1: silueta vs ARI = −0.62; **CH vs ARI = +0.33**. CH
cambia de signo pero:

1. +0.33 sobre **n=8** = débil, casi ruido.
2. CH sigue siendo **ciego a las etiquetas** — mide geometría, no estilo. En E1
   completo coronó a **yolon** (CH=42), que no tiene ventaja de estilo.
3. CH es no acotado, depende de escala/dimensión, sesga hacia clústeres convexos
   y hacia más clústeres → aún menos comparable entre extractores.

**Conclusión:** ningún métrica intrínseca lo resuelve, porque el defecto no es la
fórmula sino que **toda métrica sin etiquetas puntúa geometría en el espacio
propio del embedding, ciega a si ese eje es el concepto objetivo**. La solución
es el subconjunto etiquetado (§8).

## La consecuencia práctica (el remate)

Se eligió la **cabeza de estilo de DINOv2** — uno de los *peores* por métricas
intrínsecas en E1 — por su liderazgo en E8a (ARI 0.722). Haber elegido por el
ranking intrínseco de E1 habría dado peor rendimiento en la tarea que motiva el
trabajo. **Esto justifica todo el §8: por eso hace falta la validación
supervisada.**

---

## Posibles preguntas del jurado y respuesta corta

- **"¿Por qué Pearson y no un coeficiente de rango si hablan de ordenar?"**
  Tienen razón; Spearman es más adecuado para una afirmación de orden y da
  magnitud similar (~+0.5/+0.7). [Pendiente: cambiar en el texto.]
- **"p < 10⁻⁷ con celdas que comparten extractores no es independiente."**
  Correcto, hay pseudorreplicación; el p está inflado. Lo defendible es el
  tamaño del efecto (r), no el p. [Pendiente: quitar los p del texto.]
- **"¿El +0.78 no lo arrastra el eje del extractor?"** En parte sí: las cabezas
  de estilo están alto-alto. La correlación es más fuerte *entre* extractores que
  para afinar el agrupador con un extractor fijo. Sigue siendo válido para
  ordenar, pero conviene no sobrevenderlo.

## Tres correcciones pendientes en results.tex (si hay tiempo antes de presentar)

1. Reportar Spearman junto a Pearson (afirmación de orden).
2. Quitar/suavizar los `p < 10^{-7}` (pseudorreplicación).
3. Una frase: la correlación es más fuerte entre extractores que dentro de uno.
