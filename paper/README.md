# GMD paper — CRAN-PM v1.0

Manuscript material for the GMD model description paper companion to the ECCV submission.

## Layout

| Path                       | Purpose                                                          |
|----------------------------|------------------------------------------------------------------|
| `OUTLINE.md`               | Full section-by-section outline with REUSE/REWRITE/NEW markers   |
| `main.tex` (TBD)           | Top-level Copernicus LaTeX manuscript                            |
| `sections/*.tex`           | One file per major section (modular)                             |
| `bib/eccv_refs.bib`        | Bibliography ported from ECCV submission                         |
| `bib/gmd_extra.bib`        | New citations (CAMS, EEA, GHAP, operational AQ, GMD methods)     |
| `figures/`                 | Final figures committed to the manuscript                        |
| `figures_eccv_source/`     | ECCV figures, kept as material to potentially reuse              |
| `tables/`                  | Standalone LaTeX table files                                     |
| `scripts/`                 | Plot scripts that regenerate figures from cran-pm outputs        |

## Workflow

1. Outline sign-off (`OUTLINE.md`) — user reviews then approves.
2. Plan compute experiments (`OUTLINE.md` "Required new experiments" table) and
   schedule the SLURM jobs (P0 first, ~14 GPU-days).
3. Drop in the Copernicus template (`copernicus.cls` + sample `template.tex`).
4. Write section by section, committing each as it's drafted.
5. Generate new figures from cran-pm outputs once experiments complete.
6. Internal review pass (co-authors), submit to GMD.

## Status

Created 2026-05-10 by Phase 6 of the cran-pm packaging plan.
Outline complete. Awaiting user sign-off.
