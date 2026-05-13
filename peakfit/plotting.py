"""Plotting helpers using Plotly to create an HTML page with a 2x3 layout.

Layout (2 rows x 3 columns):
- Row 1, Cols 1-2: main data + fit
- Row 2, Col 1: residuals
- Row 2, Col 2: parameter table (HTML)
- Col 3 (rows 1-2): mathematical function definition (KaTeX)

This module supports producing HTML that references local `assets/` files
(`plotly.min.js`, KaTeX files) by passing `assets_dir` to `plot_fit`.
"""

from typing import Optional
import os
import numpy as np
import plotly.graph_objects as go


def plot_fit(iw, y, y_model, params: dict = None, param_errs: dict = None, r2: float = None,
             output_html: str = "fit_plot.html", show: bool = True,
             converged: Optional[bool] = None, convergence_message: Optional[str] = None,
             assets_dir: Optional[str] = None):
    """Create an HTML output with plots on the left and the math definition on the right.

    If `assets_dir` is provided and exists, the generated HTML will reference
    `plotly.min.js`, `katex.min.css`, `katex.min.js` and `auto-render.min.js` from
    that folder using relative paths. Otherwise CDN links will be used.
    """
    iw = np.asarray(iw)
    y = np.asarray(y)
    y_model = np.asarray(y_model)
    residuals = y - y_model

    # Build parameter table rows (strip leading 'i' from names for display)
    param_rows = []
    if params is not None:
        for k, v in params.items():
            display_name = k[1:] if isinstance(k, str) and k.startswith('i') else str(k)
            val = f"{v:.6g}" if isinstance(v, (int, float)) else str(v)
            std = ""
            if param_errs is not None and k in param_errs and param_errs[k] is not None:
                std = f"{param_errs[k]:.3g}"
            param_rows.append((display_name, val, std))
    if r2 is not None:
        # Use HTML superscript for R^2 in the table
        param_rows.append(("R<sup>2</sup>", f"{r2:.6g}", ""))

    # Main figure: data + fit (fixed height)
    fig_main = go.Figure()
    fig_main.add_trace(go.Scatter(x=iw, y=y, mode='markers', name='data', marker=dict(size=6)))
    fig_main.add_trace(go.Scatter(x=iw, y=y_model, mode='lines', name='fit', line=dict(width=2)))
    fig_main.update_layout(title='Data and Fit', margin=dict(l=40, r=10, t=40, b=40), height=520)

    # Residuals figure (fixed height)
    fig_resid = go.Figure()
    fig_resid.add_trace(go.Scatter(x=iw, y=residuals, mode='markers', name='residuals', marker=dict(size=4)))
    fig_resid.update_layout(title='Residuals', margin=dict(l=40, r=10, t=30, b=30), height=240)

    # Convert figures to HTML fragments. If local assets are available we will
    # reference them; otherwise embed Plotly into the main fragment so the
    # exported HTML is self-contained and shows plots offline.
    use_local_plotly = assets_dir is not None and os.path.isdir(assets_dir) and os.path.isfile(os.path.join(assets_dir, 'plotly.min.js'))
    if use_local_plotly:
      main_div = fig_main.to_html(full_html=False, include_plotlyjs=False)
      resid_div = fig_resid.to_html(full_html=False, include_plotlyjs=False)
    else:
      # embed plotly JS into the main fragment to avoid external CDN dependency
      main_div = fig_main.to_html(full_html=False, include_plotlyjs=True)
      resid_div = fig_resid.to_html(full_html=False, include_plotlyjs=False)

    # Build HTML parameter table (regular HTML so markup like <sup> works and text is selectable)
    table_html_lines = [
        '<div class="param-table-wrapper"><table id="param-table">',
        '<thead><tr><th>Parameter</th><th style="text-align:right">Value</th><th style="text-align:right">Std</th></tr></thead>',
        '<tbody>'
    ]
    for name, val, std in param_rows:
        table_html_lines.append(f'<tr><td class="param-name" style="text-align:left">{name}</td><td class="param-val">{val}</td><td class="param-std">{std}</td></tr>')
    table_html_lines.append('</tbody></table></div>')
    table_html = '\n'.join(table_html_lines)

    # Math expression block (LaTeX) explaining the model; ensure the explanatory sentence
    # is inside the math delimiters so KaTeX renders it.
    math_latex = r"""$$
  \begin{aligned}
  w_0(\kappa) &= \sqrt{C + D\cos\left(\frac{\pi\kappa}{2}\right)} - b,\\
  \varepsilon(\omega,\kappa) &= \frac{2(\omega - w_0(\kappa))}{g_0},\\
  \mathrm{num}(\omega,\kappa) &= \frac{(\varepsilon + q)^2}{1+\varepsilon^2},\\
  x(\kappa) &= \frac{\pi L}{a}\,\kappa,\\
  \mathrm{den}(\kappa) &= \frac{(\sin x - x\cos x)^2}{\kappa^4},\\
  \mathrm{integrand}(\omega,\kappa) &= \mathrm{den}(\kappa)\,\mathrm{num}(\omega,\kappa),\\
    y(\omega) &= N\int_{0}^{1} \mathrm{integrand}(\omega,\kappa)\,d\kappa + y_0.
  \end{aligned}
  $$"""
    math_block = f'<div id="math-block">{math_latex}</div>'

    # Small helper to escape text for the status banner
    def _html_escape(s):
        return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    status_html = ''
    if converged is True:
        msg = _html_escape(convergence_message or 'ok')
        status_html = f"<div id=\"fit-status\" style=\"padding:8px;border-radius:6px;background:#e6ffea;color:#0b6623;margin-bottom:8px;font-weight:600;\">Fit converged: {msg}</div>"
    elif converged is False:
        msg = _html_escape(convergence_message or '')
        status_html = f"<div id=\"fit-status\" style=\"padding:8px;border-radius:6px;background:#ffecec;color:#8b0000;margin-bottom:8px;font-weight:600;\">Fit failed: {msg}</div>"

    # Determine asset URLs (either local relative paths or CDN links)
    plotly_src = 'https://cdn.plot.ly/plotly-latest.min.js'
    katex_css = 'https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css'
    katex_js = 'https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js'
    katex_autorender = 'https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js'
    # Only use local assets when the expected JS file exists in the assets dir.
    if use_local_plotly:
        out_dir = os.path.abspath(os.path.dirname(output_html) or '.')
        try:
            rel = os.path.relpath(os.path.abspath(assets_dir), out_dir)
        except Exception:
            rel = os.path.abspath(assets_dir)
        rel = rel.replace('\\', '/')
        plotly_src = f"{rel}/plotly.min.js"
        katex_css = f"{rel}/katex.min.css"
        katex_js = f"{rel}/katex.min.js"
        katex_autorender = f"{rel}/auto-render.min.js"

    # Compose final HTML using a CSS grid (2 rows x 3 columns) with fixed plot sizes
    html_template = """<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Peak fit</title>
    <link rel="stylesheet" href="%%KATEX_CSS%%">
    %%PLOTLY_HEAD%%
    <script defer src="%%KATEX_JS%%"></script>
    <script defer src="%%KATEX_AUTORENDER%%"></script>
    <style>
      body { font-family: Arial, sans-serif; margin: 10px; }
      .status { grid-column: 1 / span 3; }
      /* fixed-size layout */
      .container { display: grid; grid-template-columns: 1.7fr 1fr 360px; grid-template-rows: auto auto; gap: 12px; align-items: start; }
      .main { grid-column: 1 / span 2; grid-row: 1; }
      .resid { grid-column: 1; grid-row: 2; }
      .table { grid-column: 2; grid-row: 2; }
      .math { grid-column: 3; grid-row: 1 / span 2; overflow:auto; max-height: 90vh; padding-left: 10px; }
      .param-table-wrapper { max-height: 420px; overflow:auto; }
      table { width: 100%; border-collapse: collapse; font-family: monospace; }
      th, td { border: 1px solid #ddd; padding: 6px; }
      th { background: #f0f0f0; font-weight: 600; text-align: left; }
      .param-val, .param-std { text-align: right; }
      /* Allow selecting/copying inside Plotly output */
      .js-plotly-plot, .plotly, table, td, th, div { user-select: text !important; -webkit-user-select: text !important; }
    </style>
  </head>
  <body>
    %%STATUS_HTML%%
    <div class="container">
      <div class="main">%%MAIN_DIV%%</div>
      <div class="math">%%MATH_BLOCK%%</div>
      <div class="resid">%%RESID_DIV%%</div>
      <div class="table">%%TABLE_HTML%%</div>
    </div>
    <script>
      // Render KaTeX in the math block after load
      window.addEventListener('DOMContentLoaded', function() {
        if (window.renderMathInElement) {
          renderMathInElement(document.getElementById('math-block'), {delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}]});
        }
      });
    </script>
  </body>
</html>"""

    # Decide what to place in the header for Plotly (either a script tag
    # referencing a local file / CDN, or empty when Plotly is already
    # embedded into `main_div`).
    if use_local_plotly:
      plotly_head = f"<script src=\"{plotly_src}\"></script>"
    else:
      # main_div already embeds Plotly when local assets are not available
      plotly_head = ''

    html = (html_template
        .replace('%%MAIN_DIV%%', main_div)
        .replace('%%RESID_DIV%%', resid_div)
        .replace('%%TABLE_HTML%%', table_html)
        .replace('%%MATH_BLOCK%%', math_block)
        .replace('%%STATUS_HTML%%', status_html)
        .replace('%%PLOTLY_HEAD%%', plotly_head)
        .replace('%%KATEX_CSS%%', katex_css)
        .replace('%%KATEX_JS%%', katex_js)
        .replace('%%KATEX_AUTORENDER%%', katex_autorender))

    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html)

    if show:
        try:
            import webbrowser
            webbrowser.open('file://' + os.path.abspath(output_html))
        except Exception:
            pass

    # Return the main figure for further programmatic use
    return fig_main
