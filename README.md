# Union Glassdoor Analysis Pipeline

This repository combines union-election data with Glassdoor/Revelio review data and Compustat controls. It builds analysis-ready datasets at both firm-year and comment-event levels for studying employee reviews around union elections.

## Current Status

- Git branch: `main`, tracking `origin/main`.
- Working tree was clean before this README update.
- The repository already has generated Compustat controls, title-classification artifacts, firm-year regression panels, comment-level event-window panels, Stata exports, winsorized variants, and Stata variable-name maps.
- The main firm-year regression output is `outputs/union_glassdoor_firm_year_regression.parquet`.
- The main comment-level output is `outputs/union_glassdoor_comment_level_window365.parquet`.

## Directory Layout

- `src/`: Python scripts for combining union, Glassdoor, and Compustat data.
- `data/`: project-local data files, if any.
- `outputs/`: generated parquet/csv/dta/json/md artifacts.
- `logs/`: run logs.
- `notebooks/`: exploratory checks, currently including `unionglassdoor.ipynb`.

## Main Pipeline

Typical order:

```bash
python src/build_compustat_controls.py
python src/build_union_title_translation_map.py
python src/build_union_title_classification.py
python src/build_union_glassdoor_firm_year.py
python src/build_union_glassdoor_comment_level.py
```

The title-translation/classification steps use the Glassdoor standardized title universe. The firm-year and comment-level steps use outputs from both `union` and `glassdoor`.

## Project Dependencies

This repository is the **integration layer** that combines outputs from two upstream projects:

- **`union`** project provides: `union_election_rc_votes_gvkey_only.parquet` — union-election data with matched Compustat firm identifiers (gvkey).
- **`glassdoor`** project provides: 
  - `firm_year_glassdoor_union.parquet` — firm-year aggregated Glassdoor ratings and review counts
  - `job_title_standardized_universe.csv` — standardized job titles from Glassdoor reviews
- **Compustat controls** are built locally via `build_compustat_controls.py`.
- **Title classification** (unionizable vs. excluded) is performed in this layer (not in `glassdoor`), producing `union_classified_title_universe.csv` and title-translation maps.

## Key Scripts

- `build_compustat_controls.py`: builds firm-year Compustat controls and industry variables, including lagged controls and Fama-French 48 industry classification.
- `build_union_title_translation_map.py`: normalizes and translates titles from the Glassdoor standardized title universe for union-related classification work.
- `build_union_title_classification.py`: classifies job titles by union relevance and writes diagnostics, examples, and protocol files.
- `build_union_glassdoor_firm_year.py`: merges union-election outcomes, Glassdoor firm-year ratings, and Compustat controls into firm-year regression panels with Stata-compatible exports.
- `build_union_glassdoor_comment_level.py`: merges Glassdoor review/comment records to union-election events within a +/-365 day window, attaches controls, and exports raw and winsorized analysis files.

## Important Inputs

- Union election file: `/data/disk4/workspace/projects/union/outputs/union_election_rc_votes_gvkey_only.parquet`
- Glassdoor firm-year file: `/data/disk4/workspace/projects/glassdoor/outputs/firm_year_glassdoor_union.parquet`
- Glassdoor title universe: `/data/disk4/workspace/projects/glassdoor/outputs/job_title_standardized_universe.csv`
- Compustat controls output: `/data/disk4/workspace/projects/union_glassdoor/outputs/compustat_firm_controls.parquet`

## Important Outputs

- `outputs/compustat_firm_controls.parquet`
- `outputs/union_title_translation_map.csv`
- `outputs/union_classified_title_universe.csv`
- `outputs/union_glassdoor_firm_year_regression.parquet`
- `outputs/union_glassdoor_firm_year_regression.dta`
- `outputs/union_glassdoor_firm_year_regression_winsor_1_99.parquet`
- `outputs/union_glassdoor_comment_level_window365.parquet`
- `outputs/union_glassdoor_comment_level_window365.dta`
- `outputs/union_glassdoor_comment_level_window365_winsor_1_99.parquet`

## Notes for AI Handoff

- This repository is the integration layer. **Before debugging merge or validation failures, verify outputs from upstream `union` and `glassdoor` projects first** — e.g., check that `union_election_rc_votes_gvkey_only.parquet` and `firm_year_glassdoor_union.parquet` have expected row counts and no unexpected missing values.
- Treat `outputs/` as generated artifacts unless the task explicitly asks to inspect or regenerate them.
- Several scripts write both parquet and Stata `.dta` files; check Stata variable-name maps when adding long or nonstandard variable names.
- The firm-year script includes post-export validation checks for expected industry and Glassdoor variables.
- Keep code changes small and verify with `git status` before and after edits.
