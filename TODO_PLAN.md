# TODO Implementation Plan

A comprehensive catalog of all `#TODO` comments in the codebase, categorized by file, complexity, and with suggested corrections where applicable.

**Legend:**
- 🟢 **Simple** — 1–5 line fix, low risk
- 🟡 **Medium** — 10–50 lines, some refactoring needed
- 🔴 **Complex** — Major feature, architecture change, or research required

---

## Table of Contents

1. [festim_model/Model.py](#1-festim_modelmodelpy) (35 TODOs)
2. [festim_model/diagnostics/Diagnostics.py](#2-festim_modeldiagnosticsdiagnosticspy) (15 TODOs)
3. [uq/easyvvuq_festim.py](#3-uqeasyvvuq_festimpy) (20 TODOs)
4. [uq/easyvvuq_festim_correlated.py](#4-uqeasyvvuq_festim_correlatedpy) (5 TODOs)
5. [uq/festim_model_run.py](#5-uqfestim_model_runpy) (2 TODOs)
6. [uq/festim_model_scan.py](#6-uqfestim_model_scanpy) (8 TODOs)
7. [uq/util/plotting.py](#7-uqutilplottingpy) (10 TODOs)

---

## 1. festim_model/Model.py

### 🟢 Simple

| Line | TODO | Suggested Fix |
|------|------|---------------|
| 116 | `find a way to fill this in from FESTIM Model object` | After simulation completes (`run()` method), assign `self.results = <extracted data>` — this is already partially done at line ~720 where `self.result_flag = True`. Add `self.results = self.model.exports[0].data` or equivalent. |
| 165 | `test thoroughly!` | Not a code change — remove the TODO and add heat conduction to the test suite (`tests/test_scientific.py`). |
| 240 | `mind round-off errors in the mesh size` | Add rounding: `vertices = np.round(vertices, decimals=15)` or use `np.unique(vertices)` after concatenation to remove near-duplicates. |
| 251 | `add a fallback for unsupported coordinate systems` | Add after the mesh creation: `if self.coordinate_system_type not in ('cartesian', 'cylindrical', 'spherical'): raise ValueError(f"Unsupported coordinate system: {self.coordinate_system_type}")` |
| 818 | `this could be a list of QoI names` | Change `problem_instance["qoi_name"] = "tritium_concentration"` to `problem_instance["qoi_names"] = config_transport_problem.get("qoi_names", ["tritium_concentration"])` |
| 834 | `can be merged with the following code block` | Merge the `self.species` dict-comprehension with the loop on lines 836–837 into a single comprehension: `self.species = {k: F.Species(self.species_descriptor[k]["festim_name"]) for k in species_names_config}` |
| 838 | `make automatic parsing of names; create species naming dictionary` | The `species_descriptor` dict on line 829 already provides this mapping. Consider reading it from config YAML instead of hardcoding. |
| 870 | `could be done by in-line comprehension, e.g. ternary operator` | Replace the if/else block (lines 865–880) with: `settings_kwargs = {"transient": self.transient, "atol": absolute_tolerance, "rtol": relative_tolerance}; if self.transient: settings_kwargs["final_time"] = self.total_time; problem_instance["festim_problem"].settings = F.Settings(**settings_kwargs)` |
| 981 | `make Python refer it by reference` | Python already passes objects by reference — `self.model = self.problems[model_to_solve]["festim_problem"]` is already a reference, not a copy. **Remove this TODO.** |
| 1249 | `double check if this is according to FESTIM2.0` | Verify against FESTIM 2.0 docs; `F.Mesh1D(vertices)` is the correct API. **Remove TODO after verification.** |
| 1342 | `should it have a ValueError as a fallback?` | Add: `if field is None: raise ValueError(f"Unknown boundary condition quantity: {bc_quantity}")` |
| 1351 | `should it have ValueError as a fallback?` | Add: `if surface_loc_id is None: raise ValueError(f"Unknown boundary location '{bc_location}' for {bc_quantity}")` |
| 1469 | `double check the direction convention` | Add a comment confirming the convention based on FESTIM docs: `# Convention: positive flux = outward (confirmed per FESTIM docs)`. **Remove TODO after verification.** |
| 1666 | `assure that (1) all objects in list are FESTIM2.0 problems, (2) problems do not repeat` | Add validation: `assert all(hasattr(p, 'settings') for p in problem_list), "All problems must be FESTIM2.0 problem objects"; assert len(problem_list) == len(set(id(p) for p in problem_list)), "Duplicate problem references found"` |

### 🟡 Medium

| Line | TODO | Description |
|------|------|-------------|
| 132 | `estimate good simulation time via diffusion and other transport coefficients` | Implement a helper: `estimate_simulation_time(D_0, E_D, T, length)` using `t_diff = length² / D_eff` where `D_eff = D_0 * exp(-E_D/(k_B*T))`. Use as default when `total_time` not in config. |
| 188 | `make time step adaptive, based on model parameters and mesh size` | Compute initial dt from mesh Fourier number: `dt = alpha * h_min² / D_eff` where `alpha < 0.5` for stability. Already partially possible via FESTIM's adaptive stepping. |
| 760 | `add derived quantities, outputs, postprocessing` | Add a method `_add_derived_quantities_v2()` that registers `F.TotalVolume`, `F.SurfaceFlux`, etc. from config `quantities_of_interest`. |
| 793 | `should be split into common and problem specific` | Refactor `_specify_geometry` to have a base geometry setup (shared mesh, subdomains) and per-problem overrides. Extract problem-specific setup into `_configure_problem(problem_name, config)`. |
| 904 | `the two problems can be specified without if-statement with a map` | Create a mapping: `PROBLEM_CLASS_MAP = {"tritium_transport": F.HydrogenTransportProblem, "heat_transport": F.HeatTransferProblem}` and use `problem_instance["festim_problem"] = PROBLEM_CLASS_MAP[problem_name]()`. |
| 948 | `look up FESTIM2.0 class for coupled steady problem` | Research FESTIM 2.0 API for steady-state coupled problems. Currently raises `NotImplementedError` which is appropriate until resolved. |
| 1250–1251 | `add spherical coordinates` / `add refined meshes` | Extend `_specify_geometry` to handle `coordinate_system: spherical` and non-uniform meshes via config. |

### 🔴 Complex

| Line | TODO | Description |
|------|------|-------------|
| 366 | `fetch data from HTM DataBase` | Requires integration with external HTM Database API. Create a `MaterialDatabase` class that can fetch properties by material name. |
| 615–618 | `XDMF/VTK/HDF5 export and import` | Requires learning DOLFINX I/O APIs. Implement `_export_xdmf()`, `_export_vtk()`, `_export_hdf5()` methods using `dolfinx.io`. Data tensor format: `[quantity × time × elements]` in HDF5. |
| 728–733 | `Think of better BCs` / `Read Lithium data` / `Explore geometries` / `Add physical effects` / `Couple with heat conductivity` | These are research/roadmap items, not code TODOs. Move to a project roadmap or GitHub Issues. |
| 861–862 | `pass other parameters` / `test on case with spurious oscillations` | Add `linear_solver`, `preconditioner` config options to Settings. Create a test case with known oscillatory behavior. |

---

## 2. festim_model/diagnostics/Diagnostics.py

### 🟢 Simple

| Line | TODO | Suggested Fix |
|------|------|---------------|
| 45 | `by default, try to read results from the model attribute` | Change to: `self.result_folder = result_folder if result_folder else (getattr(model, 'result_folder', None) or "./results")` |
| 152 | `think if the flag is needed` | The `result_flag` code is already commented out. **Remove the commented block and the TODO entirely.** |
| 229 | `put all the descriptors like naming mapping here` | Already done — the `quantities_of_interest_descriptor` dict on lines 208–227 serves this purpose. **Remove the TODO.** |
| 275 | `get rid of both if statements for 0th and 1st iteration` | Use pre-allocation: move `data_total = np.zeros((n_points, n_timesteps))` before the loop and remove the special `current_step() == 1` branch. Initialize `n_points` from the first `f.read()` call. |
| 423–424 | `read file as a CSV file` / `consider no data for milestone` | Replace raw indexing with `pd.read_csv()` and add: `if qoi_values.shape[1] <= i: logger.warning(f"No data for milestone time {time}"); continue` |
| 535 | `possibility to specify subset of quantities to visualise` | Add parameter: `def visualise(self, qoi_filter=None)` and filter: `qois_to_plot = [q for q in self.results if qoi_filter is None or q in qoi_filter]` |

### 🟡 Medium

| Line | TODO | Description |
|------|------|-------------|
| 99 | `(1) read using pandas (2) keep only last column` | Replace `np.genfromtxt` with `pd.read_csv()` throughout the file. For time series, use `df.iloc[:, -1]`. |
| 177 | `read milestones from input files` | Parse from config: `self.milestone_times = config.get("simulation", {}).get("milestones", self.times[::n_plot_frequency])` |
| 187 | `make work for both reading from file and from object` | Create a unified `_load_results()` method that tries object first, then file, using a consistent return format. |
| 197 | `Read the mesh from the result object` | Extract mesh from results: `self.mesh = results.get("mesh", None) or self._read_mesh_from_file()` |
| 204 | `make a better mechanism to define type of simulation from result files` | Check if results have multiple timesteps: `self.transient_flag = len(self.times) > 1` instead of hardcoding `True`. |
| 234 | `Make output in format: {qoi_name: dataframe(times x coordinates)}` | Refactor `read_vtx` to return `pd.DataFrame` with multi-index (time, coordinate). |
| 287 | `read all QoIs in file within a single function call` | In `read_vtx`, iterate over all available variable names in the VTX file using the ADIOS2 API. |
| 291 | `change functionality to plot from a pandas dataframe` | Refactor `visualise_*` methods to accept `pd.DataFrame` input with time/coordinate columns. |
| 461 | `make a descriptor file in a separate package for YAML parsing` | Create `festim_model/descriptors.py` with a YAML-loadable schema for quantity names, units, and formatting. |
| 506 | `make more flexible w.r.t. changes in config` | Read the title subtitle parameters dynamically: `params_str = ", ".join(f"{k}={v}" for k, v in model.config.get("model_parameters", {}).items())` |

---

## 3. uq/easyvvuq_festim.py

### 🟢 Simple

| Line | TODO | Suggested Fix |
|------|------|---------------|
| 706 | `Probably get last analysis results from the campaign` | This is already done on line 710: `results = campaign.get_last_analysis()`. **Remove the TODO.** |
| 849 | `TODO:` (empty stub at end of file) | **Delete lines 848–853** — these are stale notes, not actionable TODOs. |

### 🟡 Medium

| Line | TODO | Description |
|------|------|-------------|
| 171 | `mesh can be individual for each QoI` | Store mesh per-QoI: `meshes = {qoi: results.raw_data[qoi].get("x", default_mesh) for qoi in qois}`. Use the appropriate mesh when plotting each QoI. |
| 251 | `read means and default from the configuration file` | Parse means from config: `means = {name: config["materials"].get(name, {}).get("mean", default) for name in param_names}`. The correlated version already does this. |
| 372–373 | `tackle not implemented distributions` / `tackle different distribution specifications` | Add a distribution factory: `DIST_MAP = {"normal": cp.Normal, "uniform": cp.Uniform, "lognormal": cp.LogNormal, "beta": cp.Beta}` and dispatch based on config `distribution_type` field. |
| 417 | `read an (example) output file to get the QoI names` | Auto-detect: `with open(sample_output_file) as f: qois = f.readline().strip().split(",")`. Run once at campaign setup. |
| 473 | `rearrange FESTIM data output` | Design a structured output schema with columns `[x, qoi_name, time_step]` instead of the current flat format. |
| 477 | `provide parameters with fixed values` | Use the `AdvancedYAMLEncoder.fixed_parameters` feature (already implemented in `Encoder.py`). Pass fixed params to encoder constructor. |
| 497 | `store the YAML schema as a separate file` | Extract `parameter_map` dict to `uq/config/parameter_schema.yaml` and load with `yaml.safe_load()`. |
| 537 | `modify the YAML more arbitrarily` | Already supported by `AdvancedYAMLEncoder`'s `fixed_parameters` dict and `parameter_map`. Document usage. |
| 726 | `extract more data, on individual trajectories` | Use `campaign.get_collation_result()` to get per-sample data. Store in results pickle alongside statistics. |
| 728–730 | `specify results filename` / `save results` / `add config and distributions` | Create a `save_campaign_results(campaign, results, config, distributions, output_dir)` utility that bundles all artifacts into a timestamped folder. |

### 🔴 Complex

| Line | TODO | Description |
|------|------|-------------|
| 545 | `change output and decoder to YAML` | Design a YAML output schema for UQ-specific quantities. Implement a custom `YAMLDecoder` class extending EasyVVUQ's `BaseDecoder`. |
| 784–787 | `add more parameters for Arrhenius law` / `try higher BC` / `higher polynomial degree` / `check negative concentrations` | These are parameter study tasks, not code changes. Move to a research notebook or GitHub Issues. |

---

## 4. uq/easyvvuq_festim_correlated.py

### 🟢 Simple

| Line | TODO | Suggested Fix |
|------|------|---------------|
| 247 | `define for the Normal distribution: should be 1.0` | Add: `expansion_factor_normal = 1.0  # For Normal distribution, STD = sigma, so expansion factor is 1.0` |
| 764 | `TODO:` (empty stub) | **Delete** the empty TODO comment. |

### 🟡 Medium

| Line | TODO | Description |
|------|------|-------------|
| 298 | `assure symmetry of correlation matrix: use frozenset as key pair` | Refactor `parameter_correlations` dict to use `frozenset` keys: `parameter_correlations[frozenset(("D_0", "thermal_conductivity"))] = corr`. Add validation: `def validate_symmetric(corr_matrix): assert np.allclose(corr_matrix, corr_matrix.T), "Correlation matrix must be symmetric"`. |
| 303 | `more Pythonic way with double list comprehension` | Replace the loop with: `covariance_matrix = np.array([[parameter_correlations.get((p1, p2), 1.0 if p1 == p2 else 0.0) * std_vector[i] * std_vector[j] for j, p2 in enumerate(param_names)] for i, p1 in enumerate(param_names)])` |
| 460 | `change output and decoder to YAML` | Same as easyvvuq_festim.py line 545 — shared effort. |
| 474 | `figure out how not to use CreateRunDirectory` | Investigate EasyVVUQ's `Actions` API — `CreateRunDirectory` may be required by the framework. If removable, use direct file paths in Encode/Decode instead. |

---

## 5. uq/festim_model_run.py

### 🟡 Medium

| Line | TODO | Description |
|------|------|-------------|
| 114 | `make into a separate script such that if this fails, the model results are saved` | Wrap the post-processing block (lines 117–140) in `try/except` with `finally: save_results_for_uq(results, model)`. Move post-processing to a separate function that can fail gracefully. |
| 115 | `single out run and post-process scripts and run a single BASH script` | Create `festim_model_postprocess.py` as a standalone script. Create a wrapper `run_and_postprocess.sh` that calls both sequentially. |

---

## 6. uq/festim_model_scan.py

### 🟢 Simple

| Line | TODO | Suggested Fix |
|------|------|---------------|
| 339 | `make sure type conversion during iteration over numpy array is correct` | Add explicit cast: `param_value = float(param_value)` before passing to the UQ campaign. |
| 353 | `save and display the modified parameter in the scan` | Add: `logger.info(f"Scan iteration {i}: {param_name} = {param_value:.3E}")` and append to a results list for later saving. |

### 🟡 Medium

| Line | TODO | Description |
|------|------|-------------|
| 65 | `think of a better way to specify the range` | Accept explicit bounds from config: `param_lo = config.get("scan", {}).get("lower_bound", param_def_val * 10**(-level_variation))`. Fall back to current logic if not provided. |
| 124 | `write and store paths to parameters` | Create a `param_paths` dict mapping param names to their YAML paths (e.g., `{"length": "geometry.domains.0.length"}`). Save to a JSON alongside results. |
| 183 | `make fallback if config does not have this parameter` | Wrap in try/except: `try: config[param_name] = value; except KeyError: logger.warning(f"Parameter {param_name} not found in config root, skipping direct assignment")` |
| 271 | `map results on [0,1] normalised coordinate system` | After collecting results, normalize: `for result in results: result["r_norm"] = result["r"] / result["length"]` |
| 286 | `read *_yaml.template file to get default configuration` | Use `load_config(args.config)` which is already available. Remove TODO if config loading is already handled upstream. |
| 358 | `make a plot of Sensitivity indices at r=0 as function of parameter value` | Collect `{param_value: sobols_first_at_r0}` across scan iterations. Plot with matplotlib: `ax.plot(param_values, sobol_values_at_r0)`. For 3D: use `ax.plot_surface()`. |

---

## 7. uq/util/plotting.py

### 🟢 Simple

| Line | TODO | Suggested Fix |
|------|------|---------------|
| 192 | `pass and display proper units for the length` | Use the existing descriptor: `axs[i].set_xlabel(f"Radius, [{self.quantities_descriptor.get('x', {}).get('unit', 'm')}]")` |
| 195 | `read full name of the QoI from results` | Already using `self.quantities_descriptor[self.quantity]['name']`. The TODO is resolved. **Remove it.** |
| 309 | `should be param_name_s` | Fix the commented line: change `qoi_name_s` to `param_name_s` in the comment. |
| 463 | `compare those in absolute values - fix the y axis limits?` | Add after the plot: `ax.set_ylim(bottom=0)` or `ax.set_ylim([0, max(abs(y_max), abs(y_min))])` depending on data. |

### 🟡 Medium

| Line | TODO | Description |
|------|------|-------------|
| 314 | `get rid of the repetition of pairs` | Use `itertools.combinations` instead of `itertools.product` to avoid (i,j) and (j,i) duplicates: `for p1, p2 in itertools.combinations(param_name_s, 2)`. |
| 412 | `might be needed to read from campaign.db` | Add fallback: `if results.raw_data is None: results = campaign.get_collation_result()` — requires passing `campaign` to the plotter. |
| 425 | `figure out how to plot treemaps at arbitrary locations` | EasyVVUQ's `plot_sobols_treemap` uses matplotlib's `squarify`. To position at custom locations, use `ax.inset_axes()` and render the treemap into the inset. |
| 477 | `add total Sobol indices as well` | Mirror the first-order Sobol plotting logic: `sobols_total = results.sobols_total(); self.plot_sobols_total_vs_r(rs, sobols_total, qois, foldername=plot_folder_name)`. Create the new method by duplicating and adapting `plot_sobols_first_vs_r`. |
| 589 | `check if r_at_r changes with time, or is constant` | Add assertion: `assert all(r == r_at_r[0] for r in r_at_r), "Mesh coordinates change with time — not supported"`. If they can change, use per-timestep mesh. |
| 928 | `serialize_yaml - implement, or copy` | Implement: `import yaml; def serialize_yaml(data, filename): with open(filename, "w") as f: yaml.dump(data, f, default_flow_style=False)` |

---

## Summary

| Complexity | Count | Description |
|-----------|-------|-------------|
| 🟢 Simple | ~30 | 1–5 line fixes: add error handling, remove stale TODOs, use existing APIs |
| 🟡 Medium | ~45 | 10–50 line changes: refactoring, new utility functions, pandas migration |
| 🔴 Complex | ~20 | Major features: HTM DB integration, XDMF/VTK export, new distribution types, architecture redesign |

### Recommended Priority Order

1. **Remove stale/resolved TODOs** (lines 981, 706, 849, 764, 229, 152, 195) — immediate cleanup
2. **Add error handling** (lines 251, 1342, 1351, 1666) — improves robustness
3. **Simple code improvements** (lines 834, 870, 275, 339, 353) — reduces technical debt
4. **Pandas migration** (Diagnostics.py lines 99, 291, 423) — modernizes I/O
5. **Distribution factory** (easyvvuq_festim.py line 372) — extends UQ capabilities
6. **Problem class mapping** (Model.py line 904) — improves architecture
7. **Complex features** — plan as separate GitHub Issues with clear scope
