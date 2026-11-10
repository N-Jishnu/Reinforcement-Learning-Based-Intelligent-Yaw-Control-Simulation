# Extractable components

The dashboard uses Streamlit primitives and has no custom web components to extract. Its section renderers could become reusable Python helpers if more pages are added, but they are intentionally local to the single dashboard page:

## ControllerTrace
- Source: `dashboard/app.py`
- Category: basic
- Description: controls and plots for replaying one wind trace through a selected controller.
- Extractable props: controller name, trace length, seed.
- Hardcoded: plot titles, action-space mapping, metrics labels.

## ExperimentAnalysis
- Source: `dashboard/app.py`
- Category: basic
- Description: tabular and graphical feedforward and advanced neural-network experiment results.
- Extractable props: result-directory paths.
- Hardcoded: CSV names, headings, metric semantics.
