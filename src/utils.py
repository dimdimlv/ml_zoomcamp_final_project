from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import json
import pandas as pd


def load_obesity_dataset(
	cache: bool = False,
	cache_dir: Optional[str] = None,
	include_info: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame] | Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
	"""
	Load the UCI "Estimation of obesity levels based on eating habits and physical condition"
	dataset (UCI ID: 544) as pandas DataFrames.

	Parameters
	----------
	cache : bool, default False
		If True, caches the fetched dataset locally and loads from cache if available.
	cache_dir : str | None, default None
		Directory to use for caching. If None and `cache=True`, a default directory
		will be created under project root: .cache/obesity_uci_544.
	include_info : bool, default False
		If True, also returns dataset metadata and variable information.

	Returns
	-------
	(X, y) or (X, y, metadata, variables)
		- `X`: features as a pandas DataFrame
		- `y`: targets as a pandas DataFrame/Series
		- `metadata`: dict with dataset metadata (if include_info=True)
		- `variables`: dict with variable info (if include_info=True)

	Notes
	-----
	- Requires the `ucimlrepo` package. Install via `pip install ucimlrepo`
	  or add to requirements.txt.
	- Example (Python script):
		from src.utils import load_obesity_dataset
		X, y, meta, vars = load_obesity_dataset(cache=True, include_info=True)

	- Example (Notebook):
		import sys, os
		sys.path.append(os.path.abspath(".."))  # if notebook is in /notebooks
		from src.utils import load_obesity_dataset
		X, y = load_obesity_dataset(cache=True)
	"""

	# Initialize cache-related paths to satisfy type checkers
	cache_path: Optional[Path] = None
	features_fp: Optional[Path] = None
	targets_fp: Optional[Path] = None
	metadata_fp: Optional[Path] = None
	variables_fp: Optional[Path] = None

	# Resolve cache paths
	if cache:
		if cache_dir is None:
			# project root assumed to be parent of src/
			project_root = Path(__file__).resolve().parents[1]
			cache_path = project_root / ".cache" / "obesity_uci_544"
		else:
			cache_path = Path(cache_dir)

		features_fp = cache_path / "features.pkl"
		targets_fp = cache_path / "targets.pkl"
		metadata_fp = cache_path / "metadata.json"
		variables_fp = cache_path / "variables.pkl"

		if features_fp.exists() and targets_fp.exists():
			X = pd.read_pickle(features_fp)
			y = pd.read_pickle(targets_fp)
			if include_info:
				has_meta = metadata_fp.exists()
				has_vars = variables_fp.exists()
				if has_meta and has_vars:
					with metadata_fp.open("r", encoding="utf-8") as f:
						metadata = json.load(f)
					variables = pd.read_pickle(variables_fp)
					return X, y, metadata, variables
				# fall through to fresh fetch to supply missing info
			else:
				return X, y

	# Fetch from UCI repository via ucimlrepo
	try:
		from ucimlrepo import fetch_ucirepo  # type: ignore
	except ImportError as exc:
		raise ImportError(
			"ucimlrepo is required to load the dataset. Install with 'pip install ucimlrepo'"
		) from exc

	dataset = fetch_ucirepo(id=544)

	# Guard against optional attributes flagged by type checkers
	data = dataset.data
	if data is None:
		raise RuntimeError("Dataset 'data' is not available from ucimlrepo")
	X = data.features
	y = data.targets

	metadata = dataset.metadata
	variables = dataset.variables

	# Write cache if requested
	if cache:
		# Guard: ensure cache paths are initialized in the earlier block
		assert cache_path is not None
		assert features_fp is not None
		assert targets_fp is not None
		assert metadata_fp is not None
		assert variables_fp is not None
		# Guard: ensure data objects are present for pickling
		assert X is not None
		assert y is not None
		assert metadata is not None
		assert variables is not None

		cache_path.mkdir(parents=True, exist_ok=True)
		X.to_pickle(features_fp)
		y.to_pickle(targets_fp)
		with metadata_fp.open("w", encoding="utf-8") as f:
			json.dump(metadata, f, ensure_ascii=False, indent=2)
		variables.to_pickle(variables_fp)

	if include_info:
		# Guard: ensure metadata and variables are present for return
		assert metadata is not None
		assert variables is not None
		return X, y, metadata, variables
	return X, y
