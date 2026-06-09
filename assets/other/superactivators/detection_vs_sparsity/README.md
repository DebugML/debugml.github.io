# Compare-Methods Blog Assets

This folder contains a blog-ready Chart.js bundle for the existing Llama patch detection figures generated from `Compare-Methods.ipynb`.

Scope:

- Datasets: `Coco`, `Broden-OpenSurfaces`, `Broden-Pascal`, `iSarcasm`, `GoEmotions`
- Model: `Llama`
- Sample type: `patch`
- Concept series: `avg`, `linsep`, `kmeans`, `linsep kmeans`
- Baselines: `prompt`, `random`
- Metric: weighted-average `f1`
- Percent-through-model: `100`

Files:

- `export_llama_patch_detection_vs_sparsity_bundle.py`: Rebuilds the JSON bundle from raw files in `Experiments/`.
- `llama_patch_detection_vs_sparsity.json`: Exported chart data for the five blog charts.
- `llama_patch_detection_vs_sparsity_chart.js`: Native Chart.js renderer for the full bundle.
- `llama_patch_detection_vs_sparsity_preview.html`: Standalone preview page.

Suggested usage:

1. Copy `llama_patch_detection_vs_sparsity.json` and `llama_patch_detection_vs_sparsity_chart.js` into your blog assets.
2. Load Chart.js on the page.
3. Add a container like `<div id="compare-methods-charts"></div>`.
4. Fetch the JSON and call `renderCompareMethodsCharts("#compare-methods-charts", bundle)`.
5. Serve the assets over HTTP for preview. The HTML file uses `fetch`, so `file://` loading may be blocked by the browser.

Note:

- There is no existing `Blog_Posts/` directory in this repo, so this bundle lives under `Figs/Blog_Figs/detection_vs_sparsity/`.
- The blog charts correspond to the paper figures saved as `Figs/Paper_Figs/patch_*_Llama_dataset_detection.png`.
