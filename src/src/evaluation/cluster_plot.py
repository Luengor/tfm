"""Interactive 2D cluster scatter as Vega-Lite HTML.

Projects embeddings to 2D with UMAP (when not already 2D) and writes a
self-contained Vega-Lite HTML page with hover tooltips, pan/zoom, brush
selection, a metrics table and a thumbnail grid of the selected points.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

NOISE_LABEL = -1


def project_to_2d(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
) -> np.ndarray:
    import umap

    n_samples = embeddings.shape[0]
    effective_neighbors = max(2, min(n_neighbors, n_samples - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=effective_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    return reducer.fit_transform(embeddings)


def plot_clusters_2d(
    points_2d: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    title: str,
    show_noise: bool = True,
    filenames: Sequence[str] | None = None,
    metrics: Mapping[str, Any] | None = None,
    images_root: str = "",
) -> None:
    if points_2d.shape[0] != labels.shape[0]:
        raise ValueError(
            f"points_2d and labels must align: {points_2d.shape[0]} vs {labels.shape[0]}"
        )

    if filenames is not None and len(filenames) != points_2d.shape[0]:
        raise ValueError(
            f"filenames must align with points: {len(filenames)} vs {points_2d.shape[0]}"
        )

    noise_mask = labels == NOISE_LABEL
    keep_mask = np.ones_like(labels, dtype=bool) if show_noise else ~noise_mask

    values = []
    for i in np.where(keep_mask)[0]:
        lbl = int(labels[i])
        values.append({
            "x": float(points_2d[i, 0]),
            "y": float(points_2d[i, 1]),
            "cluster": "noise" if lbl == NOISE_LABEL else str(lbl),
            "is_noise": bool(lbl == NOISE_LABEL),
            "filename": filenames[i] if filenames is not None else "",
        })

    unique_clusters = sorted(
        {v["cluster"] for v in values if not v["is_noise"]},
        key=lambda c: int(c),
    )
    domain = unique_clusters + (["noise"] if any(v["is_noise"] for v in values) else [])

    n_clusters = len(unique_clusters)
    cluster_colors = [
        f"hsl({int(360 * i / max(1, n_clusters))}, 70%, 50%)"
        for i in range(n_clusters)
    ]
    range_colors = cluster_colors + (["#cccccc"] if "noise" in domain else [])

    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "width": "container",
        "height": 600,
        "data": {"values": values, "name": "source"},
        "params": [
            {
                "name": "grid",
                "select": {
                    "type": "interval",
                    "translate": "[mousedown[!event.shiftKey], mouseup] > mousemove",
                    "zoom": "wheel!",
                },
                "bind": "scales",
            },
            {
                "name": "brush",
                "select": {
                    "type": "interval",
                    "encodings": ["x", "y"],
                    "on": "[mousedown[event.shiftKey], mouseup] > mousemove",
                    "translate": False,
                    "zoom": False,
                },
            },
        ],
        "mark": {"type": "circle"},
        "encoding": {
            "x": {"field": "x", "type": "quantitative", "title": "UMAP-1",
                  "scale": {"zero": False}},
            "y": {"field": "y", "type": "quantitative", "title": "UMAP-2",
                  "scale": {"zero": False}},
            "color": {
                "field": "cluster",
                "type": "nominal",
                "scale": {"domain": domain, "range": range_colors},
                "legend": {"title": "cluster"} if n_clusters <= 30 else None,
            },
            "size": {
                "condition": {"test": "datum.is_noise", "value": 20},
                "value": 45,
            },
            "opacity": {
                "condition": {"param": "brush", "value": 0.95},
                "value": 0.18,
            },
            "tooltip": [
                {"field": "filename", "type": "nominal"},
                {"field": "cluster", "type": "nominal"},
                {"field": "x", "type": "quantitative", "format": ".3f"},
                {"field": "y", "type": "quantitative", "format": ".3f"},
            ],
        },
        "config": {"view": {"stroke": "transparent"}},
    }

    metrics_html = _render_metrics_table(metrics or {})
    html_doc = _render_html(title, spec, metrics_html, images_root, values)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_doc, encoding="utf-8")


def _format_metric_value(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if v != v:
            return "NaN"
        return f"{v:.4f}"
    return str(v)


def _render_metrics_table(metrics: Mapping[str, Any]) -> str:
    if not metrics:
        return ""
    rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td>"
        f"<td>{html.escape(_format_metric_value(v))}</td></tr>"
        for k, v in metrics.items()
    )
    return f"""<table class="metrics">
<thead><tr><th>metric</th><th>value</th></tr></thead>
<tbody>{rows}</tbody>
</table>"""


def _render_html(
    title: str,
    spec: dict,
    metrics_html: str,
    images_root: str,
    values: list[dict],
) -> str:
    spec_json = json.dumps(spec)
    points_json = json.dumps(values)
    safe_title = html.escape(title)
    safe_root = html.escape(images_root)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>{safe_title}</title>
<script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          margin: 16px; color: #222; }}
  h1 {{ font-size: 18px; margin: 0 0 12px 0; }}
  h2 {{ font-size: 15px; margin: 24px 0 8px 0; }}
  #vis {{ width: 100%; max-width: 1200px; }}
  .hint {{ color: #666; font-size: 12px; margin: 6px 0 0 0; }}
  .controls {{ margin: 16px 0 8px 0; font-size: 13px;
              display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
  .controls input[type="text"] {{ width: 360px; padding: 4px 6px;
                                 font-family: ui-monospace, Menlo, monospace; }}
  .controls button {{ padding: 4px 10px; cursor: pointer; }}
  table.metrics {{ border-collapse: collapse; margin-top: 16px;
                   font-size: 13px; min-width: 320px; }}
  table.metrics th, table.metrics td {{
      border: 1px solid #ddd; padding: 6px 10px; text-align: left; }}
  table.metrics th {{ background: #f4f4f4; }}
  table.metrics td:nth-child(2) {{ font-family: ui-monospace, Menlo, monospace;
                                   text-align: right; }}
  table.metrics tbody tr:nth-child(odd) {{ background: #fafafa; }}
  #grid {{ display: grid;
          grid-template-columns: repeat(auto-fill, minmax(var(--thumb, 200px), 1fr));
          gap: 10px; margin-top: 8px; }}
  #grid figure {{ margin: 0; border: 1px solid #ddd; border-radius: 4px;
                 padding: 4px; background: #fff; overflow: hidden;
                 cursor: zoom-in; transition: transform 0.08s ease; }}
  #grid figure:hover {{ transform: scale(1.02); border-color: #888; }}
  #grid figure.selected {{ outline: 2px solid #2a6df4; outline-offset: -2px; }}
  #grid img {{ width: 100%; height: calc(var(--thumb, 200px) * 0.9);
              object-fit: cover; display: block; background: #f0f0f0; }}
  #grid figcaption {{ font-size: 11px; word-break: break-all;
                     color: #444; margin-top: 4px;
                     font-family: ui-monospace, Menlo, monospace; }}
  #count {{ color: #666; font-size: 12px; }}
  #thumbSize {{ width: 160px; }}

  /* Lightbox modal */
  #lightbox {{ position: fixed; inset: 0; background: rgba(0,0,0,0.88);
              display: none; z-index: 9999;
              align-items: center; justify-content: center; }}
  #lightbox.open {{ display: flex; }}
  #lightbox img {{ max-width: 92vw; max-height: 88vh; object-fit: contain;
                  background: #111; box-shadow: 0 8px 32px rgba(0,0,0,0.5); }}
  #lbCaption {{ position: absolute; bottom: 16px; left: 50%;
               transform: translateX(-50%); color: #eee;
               font-family: ui-monospace, Menlo, monospace; font-size: 13px;
               background: rgba(0,0,0,0.5); padding: 4px 10px; border-radius: 4px; }}
  .lbBtn {{ position: absolute; top: 50%; transform: translateY(-50%);
           color: #fff; background: rgba(255,255,255,0.12); border: none;
           font-size: 28px; padding: 12px 18px; cursor: pointer;
           user-select: none; border-radius: 4px; }}
  .lbBtn:hover {{ background: rgba(255,255,255,0.25); }}
  #lbPrev {{ left: 16px; }}
  #lbNext {{ right: 16px; }}
  #lbClose {{ position: absolute; top: 16px; right: 16px;
             color: #fff; background: rgba(255,255,255,0.12); border: none;
             font-size: 22px; padding: 6px 14px; cursor: pointer;
             border-radius: 4px; }}
  #lbClose:hover {{ background: rgba(255,255,255,0.25); }}
  #lbIndex {{ position: absolute; top: 16px; left: 16px;
             color: #ccc; font-size: 13px;
             font-family: ui-monospace, Menlo, monospace; }}
</style>
</head>
<body>
<h1>{safe_title}</h1>
<div id="vis"></div>
<p class="hint">Drag = pan · wheel = zoom · <b>shift-drag</b> = select points.</p>
{metrics_html}

<h2>Selected images <span id="count"></span></h2>
<div class="controls">
  <label for="imageRoot">Image path prefix:</label>
  <input id="imageRoot" type="text" value="{safe_root}"
         placeholder="e.g. ../dataset/images/ or http://host/images/" />
  <label for="thumbSize">Thumb size:</label>
  <input id="thumbSize" type="range" min="100" max="500" step="20" value="200" />
  <span id="thumbSizeVal" class="hint">200px</span>
  <button id="clearBtn" type="button">Clear selection</button>
  <span id="hint" class="hint">Shift-drag chart to select · click thumb to enlarge.</span>
</div>
<div id="grid"></div>

<div id="lightbox" role="dialog" aria-hidden="true">
  <span id="lbIndex"></span>
  <button id="lbClose" type="button" aria-label="Close">✕</button>
  <button id="lbPrev" class="lbBtn" type="button" aria-label="Previous">‹</button>
  <img id="lbImg" alt="" />
  <button id="lbNext" class="lbBtn" type="button" aria-label="Next">›</button>
  <div id="lbCaption"></div>
</div>

<script>
  const spec = {spec_json};
  const allPoints = {points_json};
  const gridEl = document.getElementById('grid');
  const countEl = document.getElementById('count');
  const rootInput = document.getElementById('imageRoot');
  const clearBtn = document.getElementById('clearBtn');
  const thumbSize = document.getElementById('thumbSize');
  const thumbSizeVal = document.getElementById('thumbSizeVal');
  const lightbox = document.getElementById('lightbox');
  const lbImg = document.getElementById('lbImg');
  const lbCaption = document.getElementById('lbCaption');
  const lbIndex = document.getElementById('lbIndex');
  const lbClose = document.getElementById('lbClose');
  const lbPrev = document.getElementById('lbPrev');
  const lbNext = document.getElementById('lbNext');
  let lastSelected = [];
  let lbCursor = -1;

  function joinPath(root, name) {{
    if (!root) return name;
    if (/^[a-z]+:\\/\\//i.test(name)) return name;
    if (root.endsWith('/') || root.endsWith('\\\\')) return root + name;
    return root + '/' + name;
  }}

  function renderGrid(points) {{
    const root = rootInput.value || '';
    countEl.textContent = points.length ? `(${{points.length}})` : '';
    if (!points.length) {{
      gridEl.innerHTML = '';
      return;
    }}
    const frag = document.createDocumentFragment();
    points.forEach((p, idx) => {{
      const fig = document.createElement('figure');
      fig.dataset.idx = idx;
      const img = document.createElement('img');
      img.loading = 'lazy';
      img.alt = p.filename;
      img.src = joinPath(root, p.filename);
      img.onerror = () => {{ img.style.opacity = '0.3'; img.title = 'failed to load'; }};
      const cap = document.createElement('figcaption');
      cap.textContent = `[${{p.cluster}}] ${{p.filename}}`;
      fig.appendChild(img);
      fig.appendChild(cap);
      fig.addEventListener('click', () => openLightbox(idx));
      frag.appendChild(fig);
    }});
    gridEl.innerHTML = '';
    gridEl.appendChild(frag);
  }}

  function openLightbox(idx) {{
    if (!lastSelected.length) return;
    lbCursor = (idx + lastSelected.length) % lastSelected.length;
    const p = lastSelected[lbCursor];
    lbImg.src = joinPath(rootInput.value || '', p.filename);
    lbImg.alt = p.filename;
    lbCaption.textContent = `[${{p.cluster}}] ${{p.filename}}  (x=${{p.x.toFixed(3)}}, y=${{p.y.toFixed(3)}})`;
    lbIndex.textContent = `${{lbCursor + 1}} / ${{lastSelected.length}}`;
    lightbox.classList.add('open');
    lightbox.setAttribute('aria-hidden', 'false');
    highlightSelected();
  }}

  function closeLightbox() {{
    lightbox.classList.remove('open');
    lightbox.setAttribute('aria-hidden', 'true');
    lbImg.src = '';
    lbCursor = -1;
    highlightSelected();
  }}

  function step(delta) {{
    if (lbCursor < 0 || !lastSelected.length) return;
    openLightbox(lbCursor + delta);
  }}

  function highlightSelected() {{
    gridEl.querySelectorAll('figure').forEach(f => f.classList.remove('selected'));
    if (lbCursor >= 0) {{
      const node = gridEl.querySelector(`figure[data-idx="${{lbCursor}}"]`);
      if (node) node.classList.add('selected');
    }}
  }}

  function filterByBrush(state) {{
    if (!state || !state.x || !state.y) return [];
    const [x0, x1] = state.x;
    const [y0, y1] = state.y;
    const xMin = Math.min(x0, x1), xMax = Math.max(x0, x1);
    const yMin = Math.min(y0, y1), yMax = Math.max(y0, y1);
    return allPoints.filter(p =>
      p.x >= xMin && p.x <= xMax && p.y >= yMin && p.y <= yMax
    );
  }}

  vegaEmbed('#vis', spec, {{actions: true, renderer: 'canvas'}}).then(({{view}}) => {{
    view.addSignalListener('brush', (_name, state) => {{
      lastSelected = filterByBrush(state);
      renderGrid(lastSelected);
    }});
  }}).catch(err => {{
    console.error('vega-embed failed', err);
  }});

  rootInput.addEventListener('input', () => renderGrid(lastSelected));
  clearBtn.addEventListener('click', () => {{
    lastSelected = [];
    renderGrid([]);
    closeLightbox();
  }});

  thumbSize.addEventListener('input', () => {{
    const v = thumbSize.value;
    document.documentElement.style.setProperty('--thumb', v + 'px');
    thumbSizeVal.textContent = v + 'px';
  }});
  // apply initial thumb size
  document.documentElement.style.setProperty('--thumb', thumbSize.value + 'px');
  thumbSizeVal.textContent = thumbSize.value + 'px';

  lbClose.addEventListener('click', closeLightbox);
  lbPrev.addEventListener('click', () => step(-1));
  lbNext.addEventListener('click', () => step(1));
  lightbox.addEventListener('click', (e) => {{
    if (e.target === lightbox) closeLightbox();
  }});
  document.addEventListener('keydown', (e) => {{
    if (!lightbox.classList.contains('open')) return;
    if (e.key === 'Escape') closeLightbox();
    else if (e.key === 'ArrowLeft') step(-1);
    else if (e.key === 'ArrowRight') step(1);
  }});
</script>
</body>
</html>
"""
