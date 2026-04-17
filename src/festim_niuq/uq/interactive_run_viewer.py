#!/usr/bin/env python3
"""
Interactive Run Viewer for EasyVVUQ Campaign Results.

Generates a self-contained HTML file with interactive Plotly.js plots
and UI controls (dropdown selectors per parameter) to explore individual
quadrature sample results from an EasyVVUQ UQ campaign.

Usage:
    python interactive_run_viewer.py --runs-dir path/to/campaign/runs
    python interactive_run_viewer.py --runs-dir path/to/campaign/runs --output viewer.html

The generated HTML file can be opened in any modern browser. It shows:
  - All individual runs as light background traces
  - Statistical bands (mean +/- std, 10%-90% quantile band)
  - Dropdown selectors for each uncertain parameter with their quadrature values
  - The selected individual run highlighted in bold colour

Created by: Copilot (based on the FESTIM-NIUQ UQ framework)
"""

import argparse
import csv
import html
import json
import os
import re
import sys

import numpy as np
import yaml


def load_yaml(path):
    """Load a YAML file and return its contents as a dictionary."""
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def discover_run_dirs(base_dir):
    """
    Recursively discover run directories under *base_dir*.

    EasyVVUQ nests runs inside ``runs/runs_*/runs_*/…/run_<id>`` directories.
    We look for leaf directories whose name matches ``run_<integer>``.
    """
    run_dir_pattern = re.compile(r"run_(\d+)")
    found = []
    for root, dirs, _files in os.walk(base_dir):
        for d in dirs:
            m = run_dir_pattern.fullmatch(d)
            if m:
                found.append((int(m.group(1)), os.path.join(root, d)))
    # Sort by run id for deterministic order
    found.sort(key=lambda x: x[0])
    return found


def read_run_data(run_dir):
    """
    Read the per-run config and result profile from a single run directory.

    Returns
    -------
    params : dict
        Parameter name -> value (float) extracted from the generated config.yaml.
    x : np.ndarray or None
        Spatial coordinates.
    profiles : dict
        QoI column name -> np.ndarray of values.
    """
    # --- Read parameters from config.yaml ---
    config_path = os.path.join(run_dir, "config.yaml")
    params = {}
    if os.path.isfile(config_path):
        cfg = load_yaml(config_path)
        params = _extract_uncertain_params(cfg)

    # --- Read result profile ---
    result_file = _find_result_file(run_dir)
    x = None
    profiles = {}
    if result_file is not None:
        x, profiles = _read_csv_result(result_file)

    return params, x, profiles


# ---- Parameter extraction helpers ------------------------------------------

# Known parameter paths in the YAML config tree.  The mapping mirrors the
# AdvancedYAMLEncoder parameter_map used by easyvvuq_festim.py.
_PARAM_PATHS = {
    "D_0": ("materials", "D_0", "mean"),
    "kappa": ("materials", "thermal_conductivity", "mean"),
    "G": ("source_terms", "concentration", "value", "mean"),
    "Q": ("source_terms", "heat", "value", "mean"),
    "E_kr": ("boundary_conditions", "concentration", "right", "E_kr", "mean"),
    "h_conv": ("boundary_conditions", "temperature", "right", "h_conv", "mean"),
}


def _resolve_path(cfg, path_parts):
    """Walk *cfg* along *path_parts*, transparently entering the first element of lists."""
    node = cfg
    for part in path_parts:
        if node is None:
            return None
        if isinstance(node, list):
            node = node[0] if node else None
        if isinstance(node, dict):
            node = node.get(part)
        else:
            return None
    return node


def _extract_uncertain_params(cfg):
    """Return a dict of uncertain parameter values found in *cfg*."""
    params = {}
    for name, path in _PARAM_PATHS.items():
        val = _resolve_path(cfg, path)
        if val is not None:
            try:
                params[name] = float(val)
            except (TypeError, ValueError):
                pass
    return params


# ---- Result file I/O helpers ----------------------------------------------


def _find_result_file(run_dir):
    """Locate the primary result CSV/TXT in *run_dir*."""
    candidates = [
        os.path.join(run_dir, "results", "results_tritium_concentration.txt"),
        os.path.join(run_dir, "results", "results_tritium_concentration.csv"),
        os.path.join(run_dir, "output.csv"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    # Fallback: first .txt or .csv in results/ or run dir
    for search_dir in [os.path.join(run_dir, "results"), run_dir]:
        if os.path.isdir(search_dir):
            for f in sorted(os.listdir(search_dir)):
                if f.endswith((".txt", ".csv")):
                    return os.path.join(search_dir, f)
    return None


def _read_csv_result(filepath):
    """
    Read a CSV/TXT result file produced by FESTIM.

    Returns (x_array, {col_name: np.array}).  The first column is treated as
    the spatial coordinate ``x``.
    """
    # Try reading with header
    with open(filepath, "r") as fh:
        first_line = fh.readline().strip()

    # Determine delimiter
    delimiter = "," if "," in first_line else None

    try:
        with open(filepath, "r") as fh:
            reader = csv.reader(fh, delimiter="," if delimiter == "," else " ")
            header = None
            rows = []
            for i, row in enumerate(reader):
                # Filter out empty tokens that arise from whitespace splitting
                row = [t for t in row if t.strip()]
                if not row:
                    continue
                # Detect header: first row that cannot be fully parsed as floats
                if i == 0:
                    try:
                        [float(v) for v in row]
                        # All numeric – no header
                        rows.append([float(v) for v in row])
                    except ValueError:
                        header = row
                        continue
                else:
                    try:
                        rows.append([float(v) for v in row])
                    except ValueError:
                        continue
        data = np.array(rows)
    except Exception:
        data = np.loadtxt(filepath, delimiter=delimiter, comments="#")
        header = None

    if data.ndim == 1:
        data = data.reshape(1, -1)

    x = data[:, 0] if data.shape[1] > 1 else np.arange(data.shape[0])
    profiles = {}
    n_cols = data.shape[1]
    for col_idx in range(1, n_cols):
        col_name = header[col_idx] if header and col_idx < len(header) else f"col_{col_idx}"
        profiles[col_name] = data[:, col_idx]

    return x, profiles


# ---- Aggregate statistics --------------------------------------------------


def compute_statistics(all_profiles, qoi_name):
    """
    Compute mean, std, and percentile bands over all runs for a given QoI.

    Parameters
    ----------
    all_profiles : list of dict
        Each entry is the ``profiles`` dict for one run.
    qoi_name : str
        Column name to aggregate.

    Returns
    -------
    dict with keys: mean, std, p10, p90, p01, p99
    """
    arrays = [p[qoi_name] for p in all_profiles if qoi_name in p]
    if not arrays:
        return None
    stacked = np.array(arrays)
    return {
        "mean": np.mean(stacked, axis=0),
        "std": np.std(stacked, axis=0),
        "p10": np.percentile(stacked, 10, axis=0),
        "p90": np.percentile(stacked, 90, axis=0),
        "p01": np.percentile(stacked, 1, axis=0),
        "p99": np.percentile(stacked, 99, axis=0),
    }


# ---- HTML generation -------------------------------------------------------


def generate_html(run_data_list, output_path):
    """
    Generate a self-contained interactive HTML viewer.

    Parameters
    ----------
    run_data_list : list of tuple
        Each entry is ``(run_id, params_dict, x_array, profiles_dict)``.
    output_path : str
        Destination HTML file path.
    """
    if not run_data_list:
        print("No run data found. Cannot generate viewer.", file=sys.stderr)
        return

    # --- Determine parameter names and their unique values ---
    param_names = sorted({pname for _, params, _, _ in run_data_list for pname in params})
    param_values = {}
    for pname in param_names:
        vals = sorted({params.get(pname) for _, params, _, _ in run_data_list if pname in params})
        param_values[pname] = vals

    # --- Determine available QoIs (column names) ---
    qoi_names = []
    seen = set()
    for _, _, _, profiles in run_data_list:
        for q in profiles:
            if q not in seen:
                qoi_names.append(q)
                seen.add(q)

    # --- Prepare JSON payload for JavaScript ---
    runs_json = []
    for run_id, params, x, profiles in run_data_list:
        entry = {
            "id": run_id,
            "params": {k: v for k, v in params.items()},
            "x": x.tolist() if x is not None else [],
        }
        prof = {}
        for q in qoi_names:
            if q in profiles:
                prof[q] = profiles[q].tolist()
        entry["profiles"] = prof
        runs_json.append(entry)

    # --- Compute statistics for each QoI ---
    all_profiles = [profiles for _, _, _, profiles in run_data_list]
    stats_json = {}
    x_ref = None
    for _, _, x, _ in run_data_list:
        if x is not None:
            x_ref = x
            break
    for q in qoi_names:
        s = compute_statistics(all_profiles, q)
        if s is not None:
            stats_json[q] = {k: v.tolist() for k, v in s.items()}
    x_ref_json = x_ref.tolist() if x_ref is not None else []

    # --- Build descriptors ---
    param_descriptors = {
        "D_0": "Diffusion Coefficient D₀ [m²/s]",
        "kappa": "Thermal Conductivity κ [W/(m·K)]",
        "G": "Generation Rate G [m⁻³ s⁻¹]",
        "Q": "Heat Source Q [W/m³]",
        "E_kr": "Surface Recombination Energy Eₖᵣ [J/mol]",
        "h_conv": "Convective HTC h_conv [W/(m²·K)]",
    }

    html_content = _build_html(
        runs_json=runs_json,
        stats_json=stats_json,
        x_ref_json=x_ref_json,
        param_names=param_names,
        param_values=param_values,
        param_descriptors=param_descriptors,
        qoi_names=qoi_names,
    )

    with open(output_path, "w") as fh:
        fh.write(html_content)

    print(f"Interactive viewer written to: {output_path}")


def _format_value(v):
    """Format a float for display in the UI (short scientific notation)."""
    if abs(v) == 0:
        return "0"
    if abs(v) >= 1e4 or abs(v) < 1e-2:
        return f"{v:.4e}"
    return f"{v:.6g}"


def _build_html(runs_json, stats_json, x_ref_json, param_names, param_values, param_descriptors, qoi_names):
    """Return the full HTML string for the interactive viewer."""

    # Build dropdown option HTML for each parameter
    dropdowns_html = []
    for pname in param_names:
        label = html.escape(param_descriptors.get(pname, pname))
        vals = param_values[pname]
        options_html = '<option value="__any__">Any</option>\n'
        for v in vals:
            options_html += f'            <option value="{v}">{html.escape(_format_value(v))}</option>\n'
        dropdowns_html.append(f"""
        <div class="control-group">
          <label for="sel-{html.escape(pname)}">{label}</label>
          <select id="sel-{html.escape(pname)}" data-param="{html.escape(pname)}" class="param-select">
            {options_html}
          </select>
        </div>""")
    dropdowns_block = "\n".join(dropdowns_html)

    # QoI selector
    qoi_options = "\n".join(
        f'            <option value="{html.escape(q)}">{html.escape(q)}</option>' for q in qoi_names
    )

    # Serialise data – use json.dumps with limited precision
    runs_data_str = json.dumps(runs_json)
    stats_data_str = json.dumps(stats_json)
    x_ref_str = json.dumps(x_ref_json)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Interactive UQ Run Viewer</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f7fa; color: #333; }}
  header {{ background: #2c3e50; color: white; padding: 1rem 2rem; }}
  header h1 {{ font-size: 1.4rem; font-weight: 600; }}
  header p {{ font-size: 0.85rem; opacity: 0.8; margin-top: 0.3rem; }}
  .container {{ display: flex; gap: 1rem; padding: 1rem 2rem; max-width: 1800px; margin: 0 auto; }}
  .sidebar {{ min-width: 260px; max-width: 320px; flex-shrink: 0; }}
  .main-plot {{ flex: 1; min-width: 0; }}
  .panel {{ background: white; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.1); padding: 1rem; margin-bottom: 1rem; }}
  .panel h2 {{ font-size: 1rem; margin-bottom: 0.75rem; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 0.4rem; }}
  .control-group {{ margin-bottom: 0.6rem; }}
  .control-group label {{ display: block; font-size: 0.78rem; font-weight: 600; margin-bottom: 0.2rem; color: #555; }}
  .control-group select {{ width: 100%; padding: 0.35rem 0.4rem; border: 1px solid #ccc; border-radius: 4px; font-size: 0.82rem; background: white; }}
  .control-group select:focus {{ border-color: #3498db; outline: none; box-shadow: 0 0 0 2px rgba(52,152,219,0.2); }}
  #info-box {{ font-size: 0.8rem; line-height: 1.5; }}
  #info-box .info-label {{ font-weight: 600; color: #2c3e50; }}
  #plot-area {{ width: 100%; height: 650px; }}
  .toggle-row {{ display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 0.5rem; }}
  .toggle-row label {{ font-size: 0.78rem; cursor: pointer; display: flex; align-items: center; gap: 0.25rem; }}
  .match-info {{ font-size: 0.82rem; color: #666; margin-top: 0.5rem; }}
  .match-info strong {{ color: #e74c3c; }}
  .btn {{ padding: 0.4rem 0.8rem; background: #3498db; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.82rem; }}
  .btn:hover {{ background: #2980b9; }}
</style>
</head>
<body>

<header>
  <h1>Interactive UQ Run Viewer</h1>
  <p>Explore individual quadrature sample results from an EasyVVUQ uncertainty propagation campaign</p>
</header>

<div class="container">
  <div class="sidebar">
    <div class="panel">
      <h2>Quantity of Interest</h2>
      <div class="control-group">
        <label for="sel-qoi">Select QoI column</label>
        <select id="sel-qoi">
          {qoi_options}
        </select>
      </div>
    </div>

    <div class="panel">
      <h2>Parameter Selection</h2>
      <p style="font-size:0.75rem; color:#888; margin-bottom:0.6rem;">
        Select parameter values to highlight matching runs.
        Use &ldquo;Any&rdquo; to leave a parameter unconstrained.
      </p>
      {dropdowns_block}
      <div class="match-info" id="match-info"></div>
    </div>

    <div class="panel">
      <h2>Display Options</h2>
      <div class="toggle-row">
        <label><input type="checkbox" id="chk-all-runs" checked> All runs</label>
        <label><input type="checkbox" id="chk-mean" checked> Mean</label>
        <label><input type="checkbox" id="chk-std" checked> &pm;1&sigma;</label>
        <label><input type="checkbox" id="chk-p10p90" checked> 10%-90%</label>
        <label><input type="checkbox" id="chk-p01p99"> 1%-99%</label>
      </div>
      <div class="toggle-row">
        <label><input type="checkbox" id="chk-log-y"> Log Y-axis</label>
      </div>
    </div>

    <div class="panel">
      <h2>Selected Run Info</h2>
      <div id="info-box"><em>No run selected yet. Use the dropdowns above to select parameter values.</em></div>
    </div>
  </div>

  <div class="main-plot">
    <div class="panel">
      <div id="plot-area"></div>
    </div>
  </div>
</div>

<script>
// ===== Embedded Data =====
const RUNS = {runs_data_str};
const STATS = {stats_data_str};
const X_REF = {x_ref_str};
const PARAM_NAMES = {json.dumps(param_names)};
const QOI_NAMES = {json.dumps(qoi_names)};

// ===== Constants =====
const MATCH_REL_TOL = 1e-9;   // Relative tolerance for parameter matching
const MATCH_ABS_TOL = 1e-30;  // Absolute tolerance for parameter matching
const HIGHLIGHT_COLORS = [
  '#e74c3c', '#e67e22', '#27ae60', '#8e44ad',
  '#f39c12', '#1abc9c', '#c0392b', '#2980b9',
];

// ===== State =====
let selectedQoi = QOI_NAMES.length > 0 ? QOI_NAMES[0] : null;

// ===== Helpers =====
function getSelectedParams() {{
  const sel = {{}};
  PARAM_NAMES.forEach(pname => {{
    const el = document.getElementById('sel-' + pname);
    if (el && el.value !== '__any__') {{
      sel[pname] = parseFloat(el.value);
    }}
  }});
  return sel;
}}

function matchesSelection(run, selection) {{
  for (const [pname, pval] of Object.entries(selection)) {{
    const rv = run.params[pname];
    if (rv === undefined || rv === null) return false;
    if (Math.abs(rv - pval) > Math.abs(pval) * MATCH_REL_TOL + MATCH_ABS_TOL) return false;
  }}
  return true;
}}

// ===== Plot Update =====
function updatePlot() {{
  const qoi = document.getElementById('sel-qoi').value;
  selectedQoi = qoi;
  const sel = getSelectedParams();
  const showAllRuns = document.getElementById('chk-all-runs').checked;
  const showMean = document.getElementById('chk-mean').checked;
  const showStd = document.getElementById('chk-std').checked;
  const showP10P90 = document.getElementById('chk-p10p90').checked;
  const showP01P99 = document.getElementById('chk-p01p99').checked;
  const logY = document.getElementById('chk-log-y').checked;

  const traces = [];
  const matchingRuns = [];

  // ---- All individual runs (light grey background) ----
  if (showAllRuns) {{
    let firstAll = true;
    RUNS.forEach(run => {{
      if (run.profiles[qoi]) {{
        const xData = run.x.length > 0 ? run.x : X_REF;
        traces.push({{
          x: xData,
          y: run.profiles[qoi],
          type: 'scatter',
          mode: 'lines',
          line: {{ color: 'rgba(180,180,180,0.35)', width: 1 }},
          name: firstAll ? 'Individual runs' : undefined,
          legendgroup: 'all_runs',
          showlegend: firstAll,
          hoverinfo: 'skip',
        }});
        firstAll = false;
      }}
    }});
  }}

  // ---- Statistical bands ----
  const st = STATS[qoi];
  if (st) {{
    const xSt = X_REF;

    // 1%-99% band (widest, lightest)
    if (showP01P99 && st.p01 && st.p99) {{
      traces.push({{
        x: xSt.concat([...xSt].reverse()),
        y: st.p99.concat([...st.p01].reverse()),
        fill: 'toself',
        fillcolor: 'rgba(52,152,219,0.08)',
        line: {{ color: 'transparent' }},
        type: 'scatter',
        name: '1%-99% band',
        showlegend: true,
        hoverinfo: 'skip',
      }});
    }}

    // 10%-90% band
    if (showP10P90 && st.p10 && st.p90) {{
      traces.push({{
        x: xSt.concat([...xSt].reverse()),
        y: st.p90.concat([...st.p10].reverse()),
        fill: 'toself',
        fillcolor: 'rgba(52,152,219,0.15)',
        line: {{ color: 'transparent' }},
        type: 'scatter',
        name: '10%-90% band',
        showlegend: true,
        hoverinfo: 'skip',
      }});
    }}

    // +/- 1 std band
    if (showStd && st.mean && st.std) {{
      const upper = st.mean.map((m, i) => m + st.std[i]);
      const lower = st.mean.map((m, i) => m - st.std[i]);
      traces.push({{
        x: xSt.concat([...xSt].reverse()),
        y: upper.concat([...lower].reverse()),
        fill: 'toself',
        fillcolor: 'rgba(52,152,219,0.22)',
        line: {{ color: 'transparent' }},
        type: 'scatter',
        name: '\u00b11\u03c3 band',
        showlegend: true,
        hoverinfo: 'skip',
      }});
    }}

    // Mean line
    if (showMean && st.mean) {{
      traces.push({{
        x: xSt,
        y: st.mean,
        type: 'scatter',
        mode: 'lines',
        line: {{ color: '#2c3e50', width: 2.5, dash: 'dash' }},
        name: 'Mean',
      }});
    }}
  }}

  // ---- Highlighted matching runs ----
  const selKeys = Object.keys(sel);
  if (selKeys.length > 0) {{
    let colorIdx = 0;
    RUNS.forEach(run => {{
      if (matchesSelection(run, sel) && run.profiles[qoi]) {{
        matchingRuns.push(run);
        const xData = run.x.length > 0 ? run.x : X_REF;
        const color = HIGHLIGHT_COLORS[colorIdx % HIGHLIGHT_COLORS.length];
        colorIdx++;
        // Build hover text with parameter values
        const hoverParts = [`Run ${{run.id}}`];
        PARAM_NAMES.forEach(pn => {{
          if (run.params[pn] !== undefined) {{
            hoverParts.push(`${{pn}}: ${{run.params[pn].toExponential(4)}}`);
          }}
        }});
        traces.push({{
          x: xData,
          y: run.profiles[qoi],
          type: 'scatter',
          mode: 'lines',
          line: {{ color: color, width: 3 }},
          name: `Run ${{run.id}}`,
          text: hoverParts.join('<br>'),
          hoverinfo: 'text+y',
        }});
      }}
    }});
  }}

  // ---- Layout ----
  const layout = {{
    xaxis: {{
      title: 'Coordinate x [m]',
      gridcolor: '#e0e0e0',
    }},
    yaxis: {{
      title: qoi,
      type: logY ? 'log' : 'linear',
      gridcolor: '#e0e0e0',
      exponentformat: 'e',
    }},
    plot_bgcolor: '#fafafa',
    paper_bgcolor: 'white',
    margin: {{ l: 80, r: 30, t: 50, b: 60 }},
    title: {{
      text: `UQ Campaign Results: ${{qoi}}`,
      font: {{ size: 16 }},
    }},
    legend: {{
      x: 1, y: 1,
      xanchor: 'right',
      bgcolor: 'rgba(255,255,255,0.85)',
      bordercolor: '#ccc',
      borderwidth: 1,
    }},
    hovermode: 'closest',
  }};

  Plotly.react('plot-area', traces, layout, {{ responsive: true }});

  // ---- Update info box ----
  const infoBox = document.getElementById('info-box');
  const matchInfo = document.getElementById('match-info');
  if (selKeys.length === 0) {{
    infoBox.innerHTML = '<em>No parameter selected. Use the dropdowns above.</em>';
    matchInfo.innerHTML = `Total runs: ${{RUNS.length}}`;
  }} else if (matchingRuns.length === 0) {{
    infoBox.innerHTML = '<em>No runs match the current selection.</em>';
    matchInfo.innerHTML = `<strong>0</strong> of ${{RUNS.length}} runs match`;
  }} else {{
    let infoHtml = '';
    matchingRuns.forEach(run => {{
      infoHtml += `<div style="margin-bottom:0.5rem;border-left:3px solid #3498db;padding-left:0.5rem;">`;
      infoHtml += `<span class="info-label">Run ${{run.id}}</span><br>`;
      PARAM_NAMES.forEach(pn => {{
        if (run.params[pn] !== undefined) {{
          const isSelected = sel[pn] !== undefined;
          const style = isSelected ? 'font-weight:600;color:#2c3e50;' : 'color:#888;';
          infoHtml += `<span style="${{style}}">${{pn}}: ${{run.params[pn].toExponential(4)}}</span><br>`;
        }}
      }});
      infoHtml += `</div>`;
    }});
    infoBox.innerHTML = infoHtml;
    matchInfo.innerHTML = `<strong>${{matchingRuns.length}}</strong> of ${{RUNS.length}} runs match`;
  }}
}}

// ===== Event Listeners =====
document.getElementById('sel-qoi').addEventListener('change', updatePlot);
document.querySelectorAll('.param-select').forEach(el => el.addEventListener('change', updatePlot));
document.getElementById('chk-all-runs').addEventListener('change', updatePlot);
document.getElementById('chk-mean').addEventListener('change', updatePlot);
document.getElementById('chk-std').addEventListener('change', updatePlot);
document.getElementById('chk-p10p90').addEventListener('change', updatePlot);
document.getElementById('chk-p01p99').addEventListener('change', updatePlot);
document.getElementById('chk-log-y').addEventListener('change', updatePlot);

// ===== Initial render =====
updatePlot();
</script>
</body>
</html>"""


# ---- CLI entry point -------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate an interactive HTML viewer for EasyVVUQ campaign results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --runs-dir path/to/campaign/runs
  %(prog)s --runs-dir path/to/campaign/runs --output my_viewer.html
        """,
    )
    parser.add_argument(
        "--runs-dir",
        required=True,
        help="Path to the EasyVVUQ campaign directory (or its runs/ subdirectory).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="uq_interactive_viewer.html",
        help="Output HTML file path (default: uq_interactive_viewer.html).",
    )
    args = parser.parse_args()

    runs_dir = args.runs_dir
    if not os.path.isdir(runs_dir):
        print(f"Error: directory not found: {runs_dir}", file=sys.stderr)
        return 1

    print(f"Scanning for run directories under: {runs_dir}")
    run_dirs = discover_run_dirs(runs_dir)
    print(f"Found {len(run_dirs)} run directories.")

    if not run_dirs:
        print("No run_<id> directories found. Please check the path.", file=sys.stderr)
        return 1

    run_data_list = []
    skipped = 0
    for run_id, rdir in run_dirs:
        params, x, profiles = read_run_data(rdir)
        if profiles:
            run_data_list.append((run_id, params, x, profiles))
        else:
            skipped += 1

    print(f"Loaded data from {len(run_data_list)} runs ({skipped} skipped – no result files).")

    if not run_data_list:
        print("No usable run data found.", file=sys.stderr)
        return 1

    generate_html(run_data_list, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
