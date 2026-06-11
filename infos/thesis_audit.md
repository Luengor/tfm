# Auditoría de secciones del TFM (documento LaTeX)

Auditoría de la memoria del TFM *«Caracterización del desempeño y el coste
computacional en algoritmos de similitud en bancos de imágenes de grafiti»*.
Cada sección se juzga contra (a) las guías de escritura de `CLAUDE.md` y
`infos/LATEX_GUIDE.md` y (b) los objetivos y la estructura recomendada de la
propuesta oficial (`doc/enunciado.md`).

**Alcance:** análisis únicamente. No se modifica ningún `.tex`. Los números de
línea son orientativos, referidos al estado actual del documento (rama de
trabajo); todos los hallazgos se han verificado también contra `origin/main`,
donde persisten.

## Resumen ejecutivo

La estructura del documento **se ajusta a la propuesta**: están las seis
secciones recomendadas por `enunciado.md` —Introducción, Estado de la cuestión,
Materiales y métodos, Metodología, Resultados y discusión, Conclusiones— en el
orden sugerido y con la nomenclatura esperada. Se cumplen los objetivos
centrales (evaluación no supervisada + supervisada sobre subconjunto etiquetado
+ caracterización del coste computacional), se respeta el alcance estrictamente
técnico que la propuesta exige, y resultados y discusión van unidos tal como
prefiere la propuesta. **No hay hallazgos críticos** (ninguna sección requerida
falta ni contradice los objetivos).

Los problemas detectados son de jerarquía/estructura interna, consistencia
numérica y pulido editorial.

### Tabla resumen

| Severidad | Nº | Hallazgos |
|---|---|---|
| Critical | 0 | — |
| High | 1 | H1 |
| Medium | 4 | M1, M2, M3, M4 |
| Low | 6 | L1, L2, L3, L4, L5, L6 |
| **Total** | **11** | |

| Id | Ubicación | Tipo | Sev. | Problema (resumen) |
|---|---|---|---|---|
| H1 | `tools.tex` §Búsqueda por similitud | move/change | High | Subsección mal nivelada que «traga» a Agrupamiento/Reducción/Métricas, que deberían colgar de «Marco conceptual». |
| M1 | `methods.tex` §Visión general | change | Medium | «cuatro etapas» vs «cinco etapas» vs las 5 del resumen: recuento inconsistente. |
| M2 | `tools.tex` §Conjunto de datos | change | Medium | «el entrenamiento de estilo contiene recortes del banco principal» omite que es mixto Salamanca+Cuenca; contradice su propia tabla. |
| M3 | `tools.tex` (~l. 297) | remove | Medium | Comentario `% TODO` olvidado en el fuente. |
| M4 | `conclusion.tex` §Resultados principales | add | Medium | No recapitula objetivo por objetivo los seis objetivos específicos de la intro. |
| L1 | `tools.tex`, `conclusion.tex` | change | Low | Prefijos de `\label` incoherentes con el nivel de seccionado. |
| L2 | `intro.tex` §Estructura del documento | change | Low | Llama «herramientas y tecnologías» a la sección titulada «Materiales y métodos». |
| L3 | `tools.tex` §título | change | Low | «Materiales y métodos» casi no contiene metodología (está en la sección siguiente). |
| L4 | `tools.tex` §Búsqueda por similitud | move | Low | Métricas de recuperación (P@k, R@k, MRR, MAP@k) definidas fuera de «Métricas de evaluación». |
| L5 | `tools.tex` `tab:subconjuntos` | change | Low | Usa `table`+`tabular` en vez del `longtable` recomendado por la guía (y usado por el resto). |
| L6 | `tools.tex` §Conjunto de datos (~l. 78) | change | Low | Describe los recortes de autoría de Salamanca como «entrenamiento» cuando son el conjunto de *evaluación*. |

---

## Critical

Ninguno. Todas las secciones que la propuesta exige están presentes, en el
orden recomendado, y ninguna contradice los objetivos declarados.

---

## High

### H1 — Subsección «Búsqueda por similitud» mal nivelada en «Materiales y métodos»

- **Ubicación:** `doc/doc/tools.tex`, §`Búsqueda por similitud`
  (`\subsection`, `\label{subsubsec:tools-similarity-search}`, ~l. 168–169),
  dentro de la sección «Materiales y métodos».
- **Tipo:** move / change (estructura).
- **Severidad:** High.
- **Problema:** «Búsqueda por similitud» está declarada como `\subsection`, al
  mismo nivel que «Marco conceptual» y «Componentes implementados». En
  consecuencia, las tres subsubsecciones que la siguen —«Agrupamiento no
  supervisado», «Reducción de dimensionalidad» y «Métricas de evaluación»—
  quedan **anidadas bajo «Búsqueda por similitud»** en el índice. Esos tres
  temas son fundamentos conceptuales que pertenecen, como hermanos de
  «Representación vectorial y métricas de distancia», a **«Marco conceptual»**.
  Tal y como está, el índice presenta una jerarquía lógicamente incorrecta: la
  reducción de dimensionalidad o las métricas de calidad no son sub-temas de la
  búsqueda por similitud. Además, la búsqueda por similitud no es uno de los
  cinco componentes intercambiables de la arquitectura (es una característica de
  los *backends* de almacenamiento), por lo que tampoco encaja como subsección
  de primer nivel paralela a «Componentes implementados».
- **Corrección sugerida:** rebajar «Búsqueda por similitud» a `\subsubsection`
  dentro de «Marco conceptual» (o moverla tras los tres temas conceptuales), y
  **reparentar** «Agrupamiento no supervisado», «Reducción de dimensionalidad»
  y «Métricas de evaluación» como subsubsecciones de «Marco conceptual», junto a
  «Representación vectorial y métricas de distancia». De paso, alinear el prefijo
  de la etiqueta con el nivel real (ver L1).

---

## Medium

### M1 — Recuento de etapas inconsistente (cuatro vs cinco)

- **Ubicación:** `doc/doc/methods.tex`, §`Visión general del flujo de trabajo`
  (l. 7 «cuatro etapas operativas» y l. 64 «Cada una de las cinco etapas»);
  contrástese con `resumen.tex` (l. 23–24), que enumera cinco etapas
  (segmentación, extracción, almacenamiento, reducción, agrupamiento).
- **Tipo:** change (consistencia).
- **Severidad:** Medium.
- **Problema:** el mismo párrafo de apertura habla de «cuatro etapas
  operativas» y, pocas líneas después, de «las cinco etapas» como componentes
  intercambiables. El lector no puede saber si el sistema tiene cuatro o cinco
  etapas. La discrepancia nace de agrupar recuperación+reducción+agrupamiento en
  una sola «etapa operativa» mientras que los componentes intercambiables son
  cinco (segmentador, *embedding*, almacenamiento, reducción, agrupamiento).
- **Corrección sugerida:** unificar el discurso. Por ejemplo, hablar siempre de
  los **cinco componentes intercambiables** y describir aparte las **dos fases**
  (ingesta y agrupamiento), evitando el número «cuatro» o aclarando
  explícitamente que esas cuatro etapas operativas agrupan a los cinco
  componentes.

### M2 — La procedencia del entrenamiento de estilo contradice su propia tabla

- **Ubicación:** `doc/doc/tools.tex`, §`Conjunto de datos`, l. 73 («el
  entrenamiento de estilo contiene recortes del banco principal, mientras que el
  de autoría se restringe a StopGrafiti»).
- **Tipo:** change (consistencia).
- **Severidad:** Medium.
- **Problema:** la frase, por contraste con autoría, da a entender que el
  entrenamiento de estilo procede **solo** del banco principal (Salamanca). Sin
  embargo, su propia `tab:subconjuntos` declara «Estilo (entr.) … StopGrafiti +
  Salamanca … 385», y tanto `methods.tex` §Evaluación extrínseca («agregando
  recortes de Salamanca y de Cuenca … al entrenamiento») como `results.tex` E8a
  («conjunto mixto de 385 recortes, 110 Salamanca + 275 Cuenca») confirman que
  es **mixto**. La redacción induce a error por omisión.
- **Corrección sugerida:** reformular para reflejar que el entrenamiento de
  estilo es mixto (Salamanca + Cuenca/StopGrafiti) y que el contraste real con
  autoría es la *disyunción de identidades/ciudades*, no la exclusividad del
  banco principal.

### M3 — Comentario `% TODO` olvidado en el fuente

- **Ubicación:** `doc/doc/tools.tex`, ~l. 297 (`% TODO: extend this a bit too`
  / «a lo mejor extender esto un poco tb»), justo antes del párrafo de métricas
  extrínsecas en §`Métricas de evaluación`.
- **Tipo:** remove.
- **Severidad:** Medium (riesgo de quedar en la versión entregada).
- **Problema:** marca de trabajo pendiente embebida en el documento. Señala
  además que el párrafo de métricas extrínsecas podría estar incompleto respecto
  a la intención del autor.
- **Corrección sugerida:** resolver el TODO (extender la descripción de las
  métricas extrínsecas ARI/NMI/F1 por pares si procede) y eliminar el
  comentario.

### M4 — Las conclusiones no recapitulan objetivo por objetivo

- **Ubicación:** `doc/doc/conclusion.tex`, §`Resultados principales`; en
  relación con `intro.tex` §`Objetivos` (seis objetivos específicos).
- **Tipo:** add.
- **Severidad:** Medium.
- **Problema:** la introducción enumera seis objetivos específicos (arquitectura
  modular, integración de extractores, barrido combinatorio, caracterización del
  coste, métricas intrínsecas/extrínsecas, análisis comparativo coste–calidad).
  La conclusión discute los resultados de forma temática pero **no cierra
  explícitamente el bucle** confirmando que cada objetivo se ha cumplido, algo
  habitualmente esperado en la conclusión de un TFM.
- **Corrección sugerida:** añadir un párrafo (o lista breve) que repase los seis
  objetivos específicos y constate, en una frase cada uno, su grado de
  consecución, enlazando al experimento o sección que lo evidencia.

---

## Low

### L1 — Prefijos de `\label` incoherentes con el nivel de seccionado

- **Ubicación:**
  - `doc/doc/tools.tex`: `\label{subsec:fundamentos}` sobre una
    `\subsubsection`; `\label{subsubsec:tools-similarity-search}` sobre una
    `\subsection`.
  - `doc/doc/conclusion.tex`: `\label{sec:main-results}` y
    `\label{sec:future-work}` sobre `\subsection`es (deberían ser `subsec:`).
- **Tipo:** change.
- **Severidad:** Low.
- **Problema:** los prefijos `sec:`/`subsec:`/`subsubsec:` que recomienda
  `LATEX_GUIDE.md` no se corresponden con el nivel real, lo que dificulta el
  mantenimiento y la lectura de las referencias cruzadas.
- **Corrección sugerida:** renombrar las etiquetas para que el prefijo refleje
  el nivel; actualizar los `\ref` correspondientes.

### L2 — «Estructura del documento» nombra mal la sección de materiales

- **Ubicación:** `doc/doc/intro.tex`, §`Estructura del documento` (l. 130–134).
- **Tipo:** change.
- **Severidad:** Low.
- **Problema:** el párrafo describe `\ref{sec:materiales}` como «las
  herramientas y tecnologías utilizadas», pero la sección se titula «Materiales
  y métodos». Pequeña incoherencia de nomenclatura.
- **Corrección sugerida:** usar el título real («Materiales y métodos») o una
  descripción consistente con él.

### L3 — El título «Materiales y métodos» apenas contiene metodología

- **Ubicación:** `doc/doc/tools.tex`, §título «Materiales y métodos».
- **Tipo:** change (opcional).
- **Severidad:** Low.
- **Problema:** la sección recoge dataset, marco conceptual, componentes y
  entorno; el «método» (flujo de trabajo, protocolo experimental) vive
  íntegramente en la sección siguiente, «Metodología». El «y métodos» del título
  resulta ligeramente engañoso. (Nota: la propuesta nombra explícitamente esta
  sección «Materiales y métodos», por lo que conservar el título es defendible.)
- **Corrección sugerida:** mantener el título por fidelidad a la propuesta, pero
  abrir la sección con una frase que aclare que el procedimiento metodológico se
  detalla en §Metodología; alternativamente, reservar el título «Materiales»
  para esta y dejar «Metodología» para la siguiente.

### L4 — Métricas de recuperación definidas fuera de «Métricas de evaluación»

- **Ubicación:** `doc/doc/tools.tex`, §`Búsqueda por similitud` (l. 186–193,
  donde se definen P@k, R@k, MRR y MAP@k) frente a §`Métricas de evaluación`
  (que solo cubre intrínsecas y extrínsecas).
- **Tipo:** move (opcional).
- **Severidad:** Low.
- **Problema:** la subsección «Métricas de evaluación» no es exhaustiva: las
  métricas de recuperación se definen aparte, en «Búsqueda por similitud». El
  lector que busque «todas las métricas» en un único sitio no las encuentra.
- **Corrección sugerida:** consolidar todas las métricas en «Métricas de
  evaluación» (intrínsecas, extrínsecas y de recuperación), o dejar una
  referencia cruzada explícita desde una a otra.

### L5 — `tab:subconjuntos` no usa `longtable`

- **Ubicación:** `doc/doc/tools.tex`, `tab:subconjuntos` (entorno `table` +
  `tabular`).
- **Tipo:** change (opcional).
- **Severidad:** Low.
- **Problema:** `LATEX_GUIDE.md` recomienda `longtable` para las tablas, y el
  resto del documento lo usa de forma sistemática; esta tabla es la excepción.
- **Corrección sugerida:** migrar a `longtable` por consistencia (o documentar
  por qué esta concreta usa un *float* `table`).

### L6 — Los recortes de autoría de Salamanca se describen como «entrenamiento»

- **Ubicación:** `doc/doc/tools.tex`, §`Conjunto de datos`, ~l. 78 («265 se
  reservan para entrenamiento … así como los recortes de autoría»).
- **Tipo:** change.
- **Severidad:** Low.
- **Problema:** los recortes de autoría de Salamanca son el conjunto de
  **evaluación** (185 recortes; el entrenamiento de autoría es exclusivamente
  Cuenca/StopGrafiti, según `methods.tex` §Evaluación extrínseca y la propia
  `tab:subconjuntos`). Englobarlos bajo «entrenamiento» confunde, aunque el
  sentido (apartarlos del conjunto no supervisado) sea correcto.
- **Corrección sugerida:** precisar que de las 265 imágenes reservadas, unas se
  usan para ajuste fino (detector y cabezas) y otras aportan los recortes de
  *evaluación* de autoría, evitando etiquetar estos últimos como entrenamiento.

---

## Notas de alineación con la propuesta (sin acción requerida)

- Las seis secciones recomendadas por `enunciado.md` están presentes, ordenadas
  y nombradas como sugiere la propuesta.
- Se respeta el alcance estrictamente técnico que exige la propuesta
  (intro §Alcance y limitaciones lo declara explícitamente).
- Se cubren los dos ejes que pide la propuesta: evaluación no supervisada
  (métricas intrínsecas sobre el banco completo) y supervisada (subconjunto
  etiquetado, E8a/E8b), más la caracterización del coste computacional en
  función de N (E6, E7, E9).
- Resultados y discusión van unidos en una sola sección, opción que la propuesta
  declara preferir.
- El resultado negativo de autoría (E8b) se reporta con honestidad y se explica
  por insuficiencia de datos anotados, en línea con la maintainability pedida en
  `CLAUDE.md` («si se encuentra un problema, decirlo directamente»).
