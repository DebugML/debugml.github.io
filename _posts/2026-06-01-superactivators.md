---
title: "The SuperActivator Mechanism: Transformers Concentrate Reliable Concept Signals in the Tail"
layout: single
excerpt: "Amid noisy concept activations, transformer attention dynamics amplify reliable concept signals into a sparse high-activation tail."
header:
  overlay_filter: "0.70"
  overlay_image: /assets/images/superactivators/GoEmotions_Llama_simple.png
  teaser: /assets/images/superactivators/GoEmotions_Llama_stretch.png
  actions:
    - label: "Paper"
      url: https://arxiv.org/abs/2512.05038
    - label: "Code"
      url: https://github.com/BrachioLab/SuperActivators
authors:
  - "Cassandra Goldberg"
  - "Chaehyeon Kim"
  - "Adam Stein"
  - "Eric Wong"
---

<script>
MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']]
  }
};
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

<style>
  .superact-table-wrap {
    margin: 1.2rem 0 0.4rem;
    overflow-x: visible;
    text-align: center;
  }

  .superact-detection-title {
    margin: 0.6rem 0 0.4rem;
    padding-bottom: 0.24rem;
    border-bottom: 2px solid #15284a;
    font-size: 0.88rem;
    font-weight: 700;
    color: #15284a;
  }

  .superact-detection-subtitle {
    margin: -0.18rem 0 0.32rem;
    font-size: 0.7rem;
    color: #5b6575;
  }

  .superact-emphasis {
    font-weight: 600;
  }

  .superact-theory-box {
    margin: 1rem 0;
    padding: 0.85rem 1rem;
    border-left: 4px solid #2f6fb5;
    border-radius: 6px;
    background: #f3f7fc;
    color: #15284a;
  }

  .superact-theory-box.corollary {
    border-left-color: #4f8a64;
    background: #f3f8f4;
  }

  .superact-theory-title {
    margin: 0 0 0.25rem;
    font-size: 0.88rem;
    font-weight: 700;
    color: #15284a;
  }

  .superact-theory-box p {
    margin: 0;
    font-size: 0.92rem;
    line-height: 1.45;
  }

  .superact-detection-table {
    width: auto;
    min-width: 800px;
    margin: 0 auto;
    border-collapse: collapse;
    font-size: 0.8rem;
    line-height: 1.14;
  }

  .superact-detection-table th,
  .superact-detection-table td {
    padding: 0.31rem 0.36rem;
    border-bottom: 1px solid #d7dfe8;
    text-align: center;
    vertical-align: middle;
  }

  .superact-detection-table thead th {
    background: #15284a;
    color: #ffffff;
    font-weight: 600;
    vertical-align: top;
    padding-top: 0.22rem;
    padding-bottom: 0.18rem;
  }

  .superact-detection-table thead tr:first-child th {
    font-size: 0.74rem;
    letter-spacing: 0.02em;
  }

  .superact-detection-table thead tr:last-child th {
    vertical-align: top;
  }

  .superact-detection-table th:first-child,
  .superact-detection-table td:first-child {
    text-align: left;
    white-space: nowrap;
  }

  .superact-detection-table td:not(:first-child) {
    white-space: nowrap;
  }

  .dataset-label {
    display: inline-block;
    padding: 0.07rem 0.22rem;
    border-radius: 999px;
    background: #edf2f7;
    color: #31435f;
    font-size: 0.64rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    text-transform: uppercase;
  }

  .superact-detection-table tbody tr:nth-child(even) {
    background: #f7f9fc;
  }

  .superact-detection-table tbody tr:hover {
    background: #edf3ff;
  }

  .score {
    white-space: nowrap;
    font-variant-numeric: tabular-nums;
  }

  .score-main.score-best {
    text-decoration: underline;
    text-decoration-thickness: 1px;
    text-underline-offset: 0.12em;
  }

  .score-main.score-ours {
    color: #15284a;
    font-weight: 700;
  }

  .score-error {
    margin-left: 0.04rem;
    font-size: 0.61em;
    letter-spacing: -0.01em;
    color: #667085;
  }

  .ours-tag {
    display: block;
    font-size: 0.62rem;
    font-weight: 600;
    line-height: 1.05;
    margin-top: 0.03rem;
  }
</style>

> Concept vectors are meant to be helpful interpretability tools, associating directions in a model's latent space with human-understandable concepts. However, in practice their activations are noisy and inconsistent. Within this noise, we find a clear pattern: as activations pass through transformer layers, concept-aligned heads amplify the most extreme signals into a sparse high-activation tail. These high-tail tokens, which we call SuperActivators, provide a clear signal of concept presence.

# Where Is the Concept, Actually?

<p>Concept vectors give us a lightweight way to connect human-meaningful ideas (like objects, attributes, or emotions) to a model's internal representations, helping us understand and sometimes influence opaque deep learning models.</p>

<p>For a given image or text sample, we score each token by how strongly it aligns with that concept; ideally, true concept tokens score higher than the rest. <span class="superact-emphasis">In practice, these activation scores are noisy and unreliable, misrepresenting true concept presence.</span></p>

{% include toggle-multidataset-js.html
   id="multi-datasets"
   caption="Raw activations are shown as heatmaps, with red indicating high activation and blue indicating low activation; SuperActivators are marked with green squares. Click between datasets, and toggle between raw activations and +SuperActivators views."
   auto_ms=2000
   resume_ms=5000
   default_label="COCO"

   dataset1_label="COCO"
   dataset1_raw="/assets/images/superactivators/Coco_example_nosuper.png"
   dataset1_super="/assets/images/superactivators/Coco_example.png"

   dataset2_label="OpenSurfaces"
   dataset2_raw="/assets/images/superactivators/OpenSurfaces_example_nosuper.png"
   dataset2_super="/assets/images/superactivators/OpenSurfaces_example.png"

   dataset3_label="Pascal"
   dataset3_raw="/assets/images/superactivators/Pascal_example_nosuper.png"
   dataset3_super="/assets/images/superactivators/Pascal_example.png"

   dataset4_label="iSarcasm"
   dataset4_raw="/assets/images/superactivators/iSarcasm_example_nosuper.png"
   dataset4_super="/assets/images/superactivators/iSarcasm_example.png"

   dataset5_label="GoEmotions"
   dataset5_raw="/assets/images/superactivators/GoEmotions_example_nosuper.png"
   dataset5_super="/assets/images/superactivators/GoEmotions_example.png"
%}
<figcaption style="text-align:center;">Raw activations are shown as heatmaps, with red indicating high activation and blue indicating low activation; SuperActivators are marked with green squares. Click between datasets, and toggle between raw activations and +SuperActivators views.</figcaption>


In the COCO example, the activation heatmaps for *Animal* and *Person* appear to highlight the same tokens, even though only *Animal* is present. As a result, if you only saw the *Person* heatmap, you might incorrectly assume a person is in the image. The reverse also happens: even when *Car* is present, many true *Car* tokens barely activate for the *Car* concept.

Such noisy activation signals make it difficult to reliably detect or localize concepts. This raises the question:

<div style="text-align: center; font-size: 1.2em; font-style: italic; margin: 30px 0; color: #15284a;"> Do reliable concept signals exist within noisy activations, and if so, where do they appear? </div>

To answer this question, we zoom out beyond a single image or text sample and look at activation distributions across a dataset.

# The SuperActivator Mechanism Cuts Through the Noise
While most activations remain noisy, we discover that a small set of reliable concept signals concentrates in the upper tail of the in-concept activation distribution. This tail forms through a transformer dynamic, which we call the **SuperActivator Mechanism**, where already concept-aligned tokens are amplified across layers until they separate from the surrounding noise.

The resulting high-tail tokens, or **SuperActivators**, are reliable concept signals because they exhibit two key properties:

1. **Precision**: when the signal fires, it is distinguishable from out-of-concept noise.
2. **Recall**: the signal appears in most samples where the concept is present.

![SuperActivator example](/assets/images/superactivators/GoEmotions_Llama_sample.png)

Operationally, SuperActivators are defined by a sparsity parameter, δ, which isolates the top percentile of the in-concept distribution, so δ = 0.05 keeps the top 5% of in-concept activations.

We observe the same pattern across many settings:

<table style="width:auto;display:table;margin:0.9rem auto 1rem !important;border-collapse:collapse;text-align:left;">
  <thead>
    <tr>
      <th style="padding:0.35rem 0.9rem;">Modalities</th>
      <th style="padding:0.35rem 0.9rem;">Concept Types</th>
      <th style="padding:0.35rem 0.9rem;">Models</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding:0.35rem 0.9rem;vertical-align:top !important;">
        Image<br>
        <span style="display:block;margin-left:0.25rem;line-height:1.25;"><span style="display:inline-block;width:0.55rem;height:0.55rem;border-left:1px solid #9aa4b2;border-bottom:1px solid #9aa4b2;border-radius:0 0 0 5px;margin-right:0.25rem;vertical-align:0.18rem;"></span>4 datasets</span>
        Text<br>
        <span style="display:block;margin-left:0.25rem;line-height:1.25;"><span style="display:inline-block;width:0.55rem;height:0.55rem;border-left:1px solid #9aa4b2;border-bottom:1px solid #9aa4b2;border-radius:0 0 0 5px;margin-right:0.25rem;vertical-align:0.18rem;"></span>3 datasets</span>
      </td>
      <td style="padding:0.35rem 0.9rem;vertical-align:top !important;">Mean prototypes<br>Linear separators<br>K-Means clusters<br>K-Means separators</td>
      <td style="padding:0.35rem 0.9rem;vertical-align:top !important;">CLIP<br>LLaMA-3.2-Vision-Instruct<br>Gemma-2-9B<br>Qwen3-Embedding-4B</td>
    </tr>
  </tbody>
</table>

This breadth suggests that the SuperActivator Mechanism reflects a **general principle of how transformers encode semantics**.

<h1 id="where-do-superactivators-come-from" style="margin:1.2rem 0 0 !important;padding-bottom:0 !important;line-height:1.05;">Where Do SuperActivators Come From?</h1>
<p style="margin:0 0 0.8rem !important;padding-top:0 !important;">To understand where SuperActivators come from, we first examine how activation distributions evolve through the model, then provide a theoretical analysis of why concept-aligned attention creates this tail.</p>

## Separation Emerges in the Tail Across Layers
Below, we track activation distributions across model layers for tokens labeled as *in-concept* versus *out-of-concept*.

<style>
  #hist-datasets-figbox { padding: .5rem !important; }
  #hist-datasets-figbox > div[style*="position:absolute"] {
    position: static !important;
    left: auto !important;
    bottom: auto !important;
    transform: none !important;
    margin: .65rem auto 0;
    display: flex !important;
    justify-content: center;
  }
</style>

{% include toggle-multidataset-js.html
   id="hist-datasets"
   caption="Activation distributions separate primarily in the extreme tail as model depth increases."
   auto_ms=2200
   resume_ms=5000
   default_label="OpenSurfaces"
   dataset_prompt="Datasets — click between histogram views"
   view_a_label="Raw Distributions"
   view_b_label="+ SuperActivators"
   view_a_alt="Raw activation distributions"
   view_b_alt="Activation distributions with SuperActivators"

   dataset1_label="OpenSurfaces"
   dataset1_raw="/assets/images/superactivators/hists/Llama_Broden-OpenSurfaces_supers_False_activation_distributions_grid.png"
   dataset1_super="/assets/images/superactivators/hists/Llama_Broden-OpenSurfaces_supers_True_activation_distributions_grid.png"

   dataset2_label="COCO"
   dataset2_raw="/assets/images/superactivators/hists/Llama_Coco_supers_False_activation_distributions_grid.png"
   dataset2_super="/assets/images/superactivators/hists/Llama_Coco_supers_True_activation_distributions_grid.png"

   dataset3_label="Pascal"
   dataset3_raw="/assets/images/superactivators/hists/Llama_Broden-Pascal_supers_False_activation_distributions_grid.png"
   dataset3_super="/assets/images/superactivators/hists/Llama_Broden-Pascal_supers_True_activation_distributions_grid.png"

   dataset4_label="GoEmotions"
   dataset4_raw="/assets/images/superactivators/hists/Llama_GoEmotions_supers_False_activation_distributions_grid.png"
   dataset4_super="/assets/images/superactivators/hists/Llama_GoEmotions_supers_True_activation_distributions_grid.png"

  dataset5_label="iSarcasm"
  dataset5_raw="/assets/images/superactivators/hists/Llama_iSarcasm_supers_False_activation_distributions_grid.png"
  dataset5_super="/assets/images/superactivators/hists/Llama_iSarcasm_supers_True_activation_distributions_grid.png"
%}

In early layers, the out-of-concept distribution is roughly normal and centered around 0, while the in-concept distribution looks similar but with a slight positive shift or skew.

As we move deeper, the concept signal does not get stronger everywhere: most in-concept activations still overlap with the out-of-concept distribution, which explains the observed noise. However, a small high-activation tail pulls away cleanly enough to give us **precision**.

Crucially, we also observe that most in-concept samples have at least one activation in this well-separated tail, giving us **recall**.

{% include superactivators-detection-sparsity-chart.html %}

## Theory: Why This Tail Emerges
For a transformer model to propagate a concept signal forward, we assume at least one attention head in each layer has a concept-aligned read-write path.

Here, we present the idealized case where these attention heads are perfectly concept-aligned, with no interference from other heads, MLPs, or output projection mixing. Nearly the same results hold with noise, as long as the concept signal is large enough.

Under these assumptions, the residual update has a simple structure: each token keeps its current concept activation and receives an attention-weighted update from the other tokens.

We first prove that this residual attention update amplifies concept activation differences in general:

<div class="superact-theory-box">
  <div class="superact-theory-title">Theorem 1: Activation Gap Amplification</div>
  <p>If any two tokens already differ in concept activation, a concept-aligned attention head makes that gap larger in the next layer.</p>
</div>

<img src="/assets/images/superactivators/theorems/thm_1.png" alt="" style="display:block;max-width:100%;height:auto;margin:1rem auto;">

This has two direct consequences:

<div class="superact-theory-box corollary">
  <div class="superact-theory-title">Corollary 1: Attention Concentration</div>
  <p>As activation gaps grow, attention increasingly concentrates on the most extreme tokens.</p>
</div>

Once attention has concentrated on the extremes, same-tail tokens attend to the same extreme token and receive nearly the same update, which drives the second consequence:

<div class="superact-theory-box corollary">
  <div class="superact-theory-title">Corollary 2: Within-Tail Equalization</div>
  <p>Relative activations within the same tail eventually equalize.</p>
</div>

SuperActivators arise in the finite-depth regime of real transformers, after the tail has separated but before it collapses into this uniform behavior.

We next prove where activation gap growth is strongest:

<div class="superact-theory-box">
  <div class="superact-theory-title">Theorem 2: Tail-Asymmetric Amplification</div>
  <p>Any existing skew in the activation distribution is amplified across layers.</p>
</div>

<img src="/assets/images/superactivators/theorems/thm_2.png?v=20260606-1856" alt="" style="display:block;max-width:100%;height:auto;margin:1rem auto;">

The slight positive tail we observe early on is amplified by concept-aligned heads into the increasingly extreme high-activation tails we see empirically.

# SuperActivators Provide Reliable and Localized Concept Signals

We evaluate the extreme tail implied by the theory on two tasks:

- **concept detection:** *whether* a concept is present anywhere in a sample, and *how sparse* the reliable evidence can be
- **concept localization:** *where* a concept appears within a sample

## SuperActivators Improve Detection with Sparse Evidence

We predict that a concept is present if the sample contains a SuperActivator:

<figure id="concept-detection-bars" style="max-width:1100px;margin:1.5rem auto;text-align:center;">
  <div style="display:flex;align-items:flex-start;justify-content:center;gap:1.2rem;width:100%;">
    <div style="height:360px;position:relative;flex:1 1 auto;min-width:0;">
      <canvas id="concept-detection-bars-canvas" aria-label="Grouped bar chart of average concept detection F1 by dataset and method with error bars" role="img"></canvas>
    </div>
    <div id="concept-detection-bars-legend" style="margin-left:auto;border:1px solid #c8d0dc;border-radius:6px;padding:.6rem .55rem;text-align:left;background:#ffffff;min-width:125px;font:12px Verdana, Geneva, sans-serif;color:#111111;">
      <div style="font-weight:700;margin-bottom:.5rem;color:#15284a;line-height:1.2;">Detection<br>Methods</div>
    </div>
  </div>
  <figcaption style="margin-top:.6rem;font-size:.95rem;color:#555;line-height:1.4;">
    Average concept detection F1 across datasets for LLaMA-3.2-11B-Vision-Instruct linear separator concepts
  </figcaption>
</figure>

<script>
(function () {
  function renderConceptDetectionBars() {
    const canvas = document.getElementById("concept-detection-bars-canvas");
    const legend = document.getElementById("concept-detection-bars-legend");
    if (!canvas || !window.Chart) return;

    const labels = ["CLEVR", "COCO", "OpenSurfaces", "Pascal", "Sarcasm", "iSarcasm", "GoEmotions"];
    const detectionDatasets = [
      { label: "RandTok", values: [0.97, 0.61, 0.44, 0.66, 0.66, 0.89, 0.37], errors: [0.09, 0.01, 0.01, 0.01, 0.06, 0.04, 0.03], color: "#8dd3c7" },
      { label: "LastTok", values: [0.88, 0.68, 0.41, 0.60, 0.68, 0.72, 0.31], errors: [0.00, 0.01, 0.01, 0.01, 0.05, 0.03, 0.03], color: "#fdb462" },
      { label: "MeanTok", values: [0.92, 0.55, 0.39, 0.59, 0.66, 0.79, 0.19], errors: [0.00, 0.01, 0.01, 0.01, 0.06, 0.03, 0.03], color: "#bebada" },
      { label: "CLS", values: [0.96, 0.57, 0.46, 0.65, 0.74, 0.91, 0.32], errors: [0.02, 0.01, 0.01, 0.01, 0.06, 0.03, 0.03], color: "#fb8072" },
      { label: "Prompt", values: [0.99, 0.69, 0.49, 0.68, 0.68, 0.79, 0.25], errors: [0.01, 0.05, 0.06, 0.05, 0.07, 0.05, 0.10], color: "#80b1d3" },
      { label: "SuperAct", values: [1.00, 0.83, 0.56, 0.82, 0.87, 0.92, 0.46], errors: [0.00, 0.01, 0.02, 0.01, 0.04, 0.03, 0.03], color: "#15284a" }
    ];

    function formatValue(value) {
      return Number(value).toFixed(2);
    }

    function buildLegend() {
      if (!legend || legend.dataset.ready === "true") return;
      detectionDatasets.forEach((series) => {
        const item = document.createElement("div");
        item.style.display = "flex";
        item.style.alignItems = "center";
        item.style.gap = ".45rem";
        item.style.margin = ".28rem 0";
        item.style.whiteSpace = "nowrap";

        const swatch = document.createElement("span");
        swatch.style.display = "inline-block";
        swatch.style.width = "24px";
        swatch.style.height = "10px";
        swatch.style.borderRadius = "2px";
        swatch.style.background = series.color;

        const label = document.createElement("span");
        label.textContent = series.label;
        if (series.label === "SuperAct") {
          label.style.textDecoration = "underline";
          label.style.textUnderlineOffset = "0.12em";
        }

        item.appendChild(swatch);
        item.appendChild(label);
        legend.appendChild(item);
      });
      legend.dataset.ready = "true";
    }

    const errorBarPlugin = {
      id: "superactivatorsDetectionErrorBars",
      afterDatasetsDraw(chart) {
        const yScale = chart.scales.y;

        chart.data.datasets.forEach((dataset, datasetIndex) => {
          const meta = chart.getDatasetMeta(datasetIndex);
          if (!meta || meta.hidden) return;

          meta.data.forEach((bar, index) => {
            const value = dataset.data[index];
            const err = dataset.errors && typeof dataset.errors[index] === "number" ? dataset.errors[index] : 0;
            const topY = yScale.getPixelForValue(Math.min(1, value + err));
            const bottomY = yScale.getPixelForValue(Math.max(0, value - err));
            chart.ctx.save();
            chart.ctx.strokeStyle = dataset.borderColor || "#222222";
            chart.ctx.lineWidth = datasetIndex === chart.data.datasets.length - 1 ? 1.5 : 1.25;
            chart.ctx.beginPath();
            chart.ctx.moveTo(bar.x, topY);
            chart.ctx.lineTo(bar.x, bottomY);
            chart.ctx.stroke();
            chart.ctx.restore();
          });
        });
      }
    };

    buildLegend();

    new window.Chart(canvas, {
      type: "bar",
      data: {
        labels,
        datasets: detectionDatasets.map((series) => ({
          label: series.label,
          data: series.values,
          errors: series.errors,
          backgroundColor: series.color,
          borderColor: series.label === "SuperAct" ? "#111111" : "#333333",
          borderWidth: 0,
          categoryPercentage: 0.72,
          barPercentage: 0.9
        }))
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
          legend: { display: false },
          title: {
            display: true,
            text: "Concept Detection Performance (F1)",
            color: "#111111",
            font: { family: "Verdana, Geneva, sans-serif", size: 18, weight: "700" },
            padding: { bottom: 10 }
          },
          tooltip: {
            callbacks: {
              label(context) {
                const err = context.dataset.errors && typeof context.dataset.errors[context.dataIndex] === "number" ? context.dataset.errors[context.dataIndex] : 0;
                return context.dataset.label + ": " + formatValue(context.parsed.y) + " +/- " + formatValue(err);
              }
            }
          }
        },
        scales: {
          x: {
            title: {
              display: true,
              text: "Dataset",
              color: "#222222",
              font: { family: "Verdana, Geneva, sans-serif", size: 13, weight: "600" }
            },
            ticks: {
              color: "#222222",
              padding: 0,
              maxRotation: 35,
              minRotation: 25,
              font: { family: "Verdana, Geneva, sans-serif", size: 11 }
            },
            grid: { display: false, tickLength: 2 },
            border: { color: "#222222" }
          },
          y: {
            min: 0,
            max: 1,
            title: {
              display: true,
              text: "Average F1",
              color: "#222222",
              font: { family: "Verdana, Geneva, sans-serif", size: 13, weight: "600" }
            },
            ticks: {
              stepSize: 0.2,
              color: "#222222",
              font: { family: "Verdana, Geneva, sans-serif", size: 11 },
              callback(value) { return Number(value).toFixed(1); }
            },
            grid: { color: "rgba(0, 0, 0, .12)" },
            border: { color: "#222222" }
          }
        }
      },
      plugins: [errorBarPlugin]
    });
  }

  if (window.Chart) {
    renderConceptDetectionBars();
  } else {
    window.addEventListener("load", renderConceptDetectionBars);
  }
})();
</script>

Notably, **our SuperActivator-based method consistently outperforms all other concept detection baselines**, improving F₁ scores by up to 0.14.

By sweeping the sparsity threshold, we find that **performance consistently peaks when using only a small fraction of the most highly activated tokens**—typically between δ=5-10%. Adding more tokens from the labeled concept region intuitively seems like it should help, but actually hurts performance.


{% include toggle-multidataset-static-js.html
     id="my-datasets"
     default_label="COCO"
     alt="For each in-concept sample, how much of the concept region is made up of SuperActivators?"
     frame_style="display:block;width:100%;box-sizing:border-box;"
     caption="CDF of the SuperActivator fraction within in-concept tokens. Most samples fall below 0.2, meaning fewer than one in five in-concept tokens is a SuperActivator."

     dataset1_label="COCO"
     dataset1_img="/assets/images/superactivators/cdfs/Coco.png"

     dataset2_label="OpenSurfaces"
     dataset2_img="/assets/images/superactivators/cdfs/Broden-OpenSurfaces.png"

     dataset3_label="Pascal"
     dataset3_img="/assets/images/superactivators/cdfs/Broden-Pascal.png"

     dataset4_label="iSarcasm"
     dataset4_img="/assets/images/superactivators/cdfs/iSarcasm.png"

     dataset5_label="GoEmotions"
     dataset5_img="/assets/images/superactivators/cdfs/GoEmotions.png"
%}

## SuperActivators Improve Attributions
Instead of explaining the global concept vector, we explain alignment with the local SuperActivators.

{% include inversion-gallery.html %}

As shown in the examples above, global concept vector attributions are very noisy, while SuperActivator attributions concentrate much more cleanly on the actual concept.

{% include superactivators-inversion-chart.html %}

Besides improving accuracy, SuperActivator-based attributions are also more *faithful*: the tokens they highlight increase the model's concept alignment when inserted and reduce it when removed.


Crucially, these improvements aren't tied to any single explainer. We tested <u>nine</u> different attribution methods, and **every single method improved when we swapped the global vector for the SuperActivator objective**.


<div style="background-color: #15284a; color: #ffffff; padding: 25px; border-radius: 8px; margin: 40px 0; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
  <h2 style="color: #ffffff; margin-top: 0; border-bottom: 1px solid rgba(255,255,255,0.3); padding-bottom: 10px; display: inline-block;">Key Takeaway</h2>
  <p style="font-size: 1.2em; margin: 15px 0 0 0; font-weight: 500;">
    Ignore the bulk, only trust the tail.
  </p>
</div>

---


For more details, see our [paper](https://arxiv.org/abs/2512.05038) and [code](https://github.com/BrachioLab/SuperActivators).

# Citation

```bibtex
@article{goldberg2025superactivators,
  title={The SuperActivator Mechanism: Transformers Concentrate Reliable Concept Signals in the Tail},
  author={Goldberg, Cassandra and Kim, Chaehyeon and Stein, Adam and Wong, Eric},
  journal={arXiv preprint arXiv:2512.05038},
  year={2025},
  url={https://arxiv.org/abs/2512.05038}
}
```
