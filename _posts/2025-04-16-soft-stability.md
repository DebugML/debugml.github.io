---
title: "Probabilistic Stability Guarantees for Feature Attributions"
layout: single
excerpt: "We scale certified explanations to provide practical guarantees in high-dimensional settings."


header:
 overlay_filter: "0.75"
 overlay_image: /assets/images/soft_stability/stability_rocks.png
 teaser: /assets/images/soft_stability/stability_rocks.png
 actions:
   - label: "Paper"
     url:
   - label: "Code"
     url: https://github.com/helenjin/soft_stability/
   - label: "Tutorial"
     url: https://github.com/helenjin/soft_stability/blob/main/tutorial.ipynb


authors:
 - Helen Jin
 - Anton Xue
 - Weiqiu You
 - Surbhi Goel
 - Eric Wong


gallery_hard_vs_soft:
 - url: /assets/images/soft_stability/hard_vs_soft_pipeline.png
   image_path: /assets/images/soft_stability/hard_vs_soft_pipeline.png
   title: A visual example of the pipeline to find certified radii by hard stability vs. soft stability.


gallery_unstable:
 - url: /assets/images/soft_stability/unstable.png
   image_path: /assets/images/soft_stability/unstable.png
   title: An unstable selection of features from SHAP.


gallery_lime_vs_shap_soft:
 - url: /assets/images/soft_stability/lime_vs_shap_soft.png
   image_path: /assets/images/soft_stability/lime_vs_shap_soft.png
   title: 


gallery_algo:
 - url: /assets/images/soft_stability/estimation_algo.png
   image_path: /assets/images/soft_stability/estimation_algo.png
   title: Algorithm for Estimating Stability Rate.


gallery_soft_certifies_more:
 - url: /assets/images/soft_stability/vit_imagenet_soft.png
   image_path: /assets/images/soft_stability/vit_imagenet_soft.png
   title: Soft stability certifies more than hard stability (Soft).
 - url: /assets/images/soft_stability/vit_imagenet_hard.png
   image_path: /assets/images/soft_stability/vit_imagenet_hard.png
   title: Soft stability certifies more than hard stability (Hard).


gallery_soft_certifies_more_tweeteval:
 - url: /assets/images/soft_stability/roberta_tweeteval_soft.png
   image_path: /assets/images/soft_stability/roberta_tweeteval_soft.png
   title: Soft stability certifies more than hard stability (Soft).
 - url: /assets/images/soft_stability/roberta_tweeteval_hard.png
   image_path: /assets/images/soft_stability/roberta_tweeteval_hard.png
   title: Soft stability certifies more than hard stability (Hard).


gallery_smoothing_wrapper:
 - url: /assets/images/soft_stability/smoothing_wrapper.png
   image_path: /assets/images/soft_stability/smoothing_wrapper.png
   title: We should not be using this particular image because it is a screenshot.


gallery_smoothing_improves_stability:
 - url: /assets/images/soft_stability/smoothing_improves_stability.png
   image_path: /assets/images/soft_stability/smoothing_improves_stability.png
   title: We should not be using this particular image because it is a screenshot.

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

<script src="https://cdn.plot.ly/plotly-2.29.1.min.js"></script>


<style>
    /* Basic styles for tabs */
    .tab-container { display: flex; cursor: pointer; }
    .tab { padding: 10px 20px; margin-right: 5px; background: #ddd; border-radius: 5px; }
    .tab.active { background: #aaa; font-weight: bold; }
    #plot-container { margin-top: 20px; }

    .plot-row {
        display: flex;
        gap: 20px; /* Optional spacing between plots */
    }

    .plot-box {
        flex: 1;                      /* Each plot takes 50% of row */
        position: relative;
        padding-bottom: 43%;         /* Aspect ratio: height = 43% of width */
    }

    .plot-inner {
        position: absolute;
        top: 0; left: 0;
        width: 100%;
        height: 100%;
    }

</style>


> Stability guarantees are an emerging tool for understanding how reliable explanations are.
> However, current methods rely on specialized architectures and give guarantees that are too conservative to be useful.
> To address these limitations, we introduce **soft stability**, a more general and flexible approach for certifying explanations that works with any model and gives more useful guarantees. 
> Our guarantees are orders of magnitude greater than existitng methods and can scale to be usable in practice in high-dimensional settings.


Powerful machine learning models are increasingly deployed in real-world applications. 
However, their opacity poses a significant challenge to safety, especially in high-stakes settings where interpretability is critical for decision-making. 
A common approach to explaining these models is through feature attributions methods, which highlight the input features that contribute most to a prediction.
However, these explanations that are the selected features are often brittle, as shown in the following figure.


<!--
{% include gallery id="gallery_unstable" layout="" caption="**(Anton: THIS IMAGE WILL BE REPLACED) An unstable selection of features from LIME.**\\
Given an input image (top), [LIME](https://github.com/marcotcr/lime){:target='_blank'} selects features it deems important for the classifier $f$'s prediction (middle). However, adding just four additional features causes the prediction to change from 'Walker hound' to 'Beagle' (bottom). This suggests that the explanation is highly sensitive to the inclusion of additional features, making it unreliable."%}
-->

<figure style="display:flex; margin:auto; gap:20px;">
  <div style="flex:1; text-align:center;">
    Original Image
    <img src="/assets/images/soft_stability/turtle_original.png"/>
    <br> Sea Turtle
  </div>

  <div style="flex:1; text-align:center;">
    Explanation
    <img src="/assets/images/soft_stability/turtle_lime.png"/>
    <br> <span style="color: #2ca02c">Sea Turtle ✓</span>
  </div>

  <div style="flex:1; text-align:center;">
    + 3 Features
    <img src="/assets/images/soft_stability/turtle_lime_pertb.png"/>
    <br> <span style="color: #d62728">Corral Reef ✗</span>
  </div>
</figure>

<figcaption>
  <strong> An unstable explanation. </strong>
  Given an input image (left), the features selected by LIME (middle) are enough to preserve Vision Transformer's prediction.
  However, adding just three more features (right, in yellow) flips the prediction, suggesting that the explanation is not robust.
</figcaption>




An ideal explanation should be *robust*: if a subset of features is genuinely explanatory for the prediction, then revealing any small set of additional features should not change the prediction, up to some tolerance threshold.
In fact, this is the notion of **hard stability**, which we explored in a [previous blog post](https://debugml.github.io/multiplicative-smoothing/){:target="_blank"}.
However, exactly computing this maximum hard stability tolerance is computationally challenging, while the lower-bound estimation algorithms that we developed suffer from two major drawbacks:

- They rely on *specialized architectures*, in particular smoothed classifiers, which constrain their applicability.
- Their guarantees are *overly conservative*, meaning they certify only small perturbations, limiting practical use.

In this work, we introduce **soft stability**, a novel approach to robustness that overcomes these limitations.
The core idea behind stability is to measure how an explanation's prediction changes as more features are revealed.
We illustrate this concept as follows.

{% include gallery id="gallery_hard_vs_soft" layout="" caption="**Soft stability provides a fine-grained measure of robustness.** LIME's explanation is only hard stable at radius $r \leq 2$.
In contast, the stability rate --- the key metric of soft stability --- offers a more nuanced view of sensitivity to added features." %}



## Soft stability: a more flexible and scalable guarantee


Although both hard stability and soft stability describe how predictions change as features are revealed, the fundamental difference lies in how they measure robustness.
We compare and contrast their definitions below.


<div class="notice--danger">
<strong> Definition. [Hard Stability] </strong>
An explanation is <strong> hard stable </strong> at radius $r$ if including up to any $r$ additional features does not change the prediction.
</div>

We use "radius" to refer to the perturbation size, i.e., the number of features added, following robustness conventions.
This radius is also used as part of soft stability's definition.
But rather than measuring whether the prediction is *always* preserved, soft stability instead measures *how often* it is preserved.


<div class="notice--info">
<strong> Definition. [Soft Stability] </strong>
At radius $r$, an explanation's <strong> stability rate </strong> $\tau_r$ is the probability that adding up to $r$ additional features does not change the prediction. 
</div>


The stability rate provides a fine-grained measure of an explanation's robustness.
For example, two explanations may appear similar, but could in fact have very different robustness values.



<figure style="display:flex; margin:auto; gap:20px;">
  <div style="flex:1; text-align:center;">
    Original
    <img src="/assets/images/soft_stability/cat_original.png"/>
    <!-- <br> -->
  </div>

  <div style="flex:1; text-align:center;">
    LIME
    <img src="/assets/images/soft_stability/cat_lime.png"/>
    <!-- <br> -->
    <!-- <span style="color: #d62728">$\tau_2 = 0.37$ ✗</span> -->
    $\tau_2 = 0.37$
    <!--
    <br> <span style="color: #2ca02c">Sea Turtle ✓</span>
    -->
  </div>

  <div style="flex:1; text-align:center;">
    SHAP
    <img src="/assets/images/soft_stability/cat_shap.png"/>
    <!-- <br> -->
    <span style="color: #2ca02c">$\tau_2 = 0.76$ ✓</span>
    <!--
    <br> <span style="color: #d62728">Corral Reef ✗</span>
    -->
  </div>
</figure>

<figcaption>
  <strong> Similar explanations may have different stability rates. </strong>
  Despite visual similarities, the explanations generated by LIME (middle) and SHAP (right) have different stability rates at radius $r = 2$.
  In this example, SHAP's explanation is considerably more stable than LIME's.
</figcaption>




By shifting to a probabilistic perspective, soft stability offers a more refined view of explanation robustness.
Two key benefits follow:
1. **Model-agnostic certification**: The soft stability rate is efficiently computable for any classifier, whereas hard stability is only easy to certify for smoothed classifers.
2. **Practical guarantees**: The certificates for soft stability are much larger and more practically useful than those obtained from hard stability.


### Certifying soft stability: challenges and algorithms
At first, certifying soft stability (computing the stability rate) appears daunting.
If there are $m$ possible features that may be included at radius $r$, then there are $\mathcal{O}(m^r)$ many perturbations to check!
In fact, this combinatorial explosion is the same computational bottleneck encountered when one tries to naively certify hard stability.



Fortunately, we can efficiently **estimate** the stability rate to a high accuracy using standard sampling techniques from statistics.
This procedure is summarized in the following figure.


{% include gallery id="gallery_algo" layout="" caption="**Estimating the stability rate $\tau_r$.** An estimator $\hat{\tau}_r$ constructed using $N \geq \log(2/\delta) / (2 \varepsilon^2)$ perturbation samples will, with probability at least $1 - \delta$, attain an accuracy of $\lvert \hat{\tau}_r - \tau_r \rvert \leq \varepsilon$. We give a reference implementation in our [tutorial notebook](https://github.com/helenjin/soft_stability/blob/main/tutorial.ipynb){:target='_blank'}. In this example, $\hat{\tau}_r = 0.953$." %}


We outline this estimation process in more detail below.

<div class="notice--success">
<strong>Algorithm for estimating the stability rate $\tau_r$.</strong> <br>

We can compute an estimator $\hat{\tau}_r$ in the following manner: <br>

<div style="margin-left: 20px;">
1. Sample (uniformly, with replacement) $N$ perturbations of the explanation, where each perturbation includes at most $r$ additional features. <br>

2. Let $\hat{\tau}_r$ be the fraction of the samples whose predictions match the original explanation's prediction. <br>
</div>
 
If $\hat{\tau}_r$ is computed with $N \geq \frac{\log(2/\delta)}{2 \varepsilon^2}$ samples, then the following holds: 
with probability at least $1 - \delta$, its estimation accuracy is $\lvert \hat{\tau}_r - \tau_r \rvert \leq \varepsilon$.
</div>


We also give a reference implementation in our [tutorial notebook](https://github.com/helenjin/soft_stability/blob/main/tutorial.ipynb){:target="_blank"}. 


There are three main computational benefits of estimating soft stability in this manner.
First, the estimation algorithm is model-agnostic, which means that soft stability can be certified for any model, not just smoothed ones --- in contrast to hard stability.
Second, this algorithm is sample-efficient: the number of samples depends only on the hyperparameters $\varepsilon$ and $\delta$, meaning that the runtime cost scales linearly with the cost of running the classifier.
Thirdly, as we show next, soft stability certificates are much less conservative than hard stability, making them more practical for giving fine-grained and meaningful measures of explanation robustness.



## Experiments

We next consider how soft stability compares with hard stability in practice. 
We empirically evaluate on vision and language tasks.


We first show the stability rates attainble with a [Vision Transformer](https://huggingface.co/google/vit-base-patch16-224){:target="_blank"} model over $1000$ [samples from ImageNet](https://github.com/helenjin/soft_stability/tree/main/imagenet-sample-images) using different explanation methods.
For each method, we select the top-25% ranked features as the explanation.



<div class="plot-row">
  <div class="plot-box"><div id="vit_soft_stability" class="plot-inner"></div></div>
  <div class="plot-box"><div id="vit_hard_stability" class="plot-inner"></div></div>
</div>

<script>
  function plotFromJSON(jsonPath, divID, title) {
    fetch(jsonPath)
      .then(res => res.json())
      .then(data => {
        const radii = data.radii;
        const methods = Object.keys(data).filter(k => k !== 'radii');

        const traces = methods.map(method => ({
          x: radii,
          y: data[method],
          type: 'scatter',
          mode: 'lines',
          name: method,
        }));

        const layout = {
          title: title,
          margin: { t: 40, l: 40, r: 40, b: 40 },
          xaxis: { title: 'Perturbation Radius' },
          yaxis: { title: 'Stability Rate' }
        };

        Plotly.newPlot(divID, traces, layout, { responsive: true });
      })
      .catch(err => console.error(`Error loading ${jsonPath}:`, err));
  }

  // Plot both datasets
  plotFromJSON('/assets/images/soft_stability/blog_vit_soft_stability.json', 'vit_soft_stability', 'VIT Soft Stability');
  plotFromJSON('/assets/images/soft_stability/blog_vit_hard_stability.json', 'vit_hard_stability', 'VIT Hard Stability');
</script>


While the hard stability guarantees quickly become vacuous at even small radii, the soft stability guarantees continue to yield meaningful guarantees.

Although our work primmarily focuses on vision models, soft stability is also computable for language models.
Next, we show the stability rates we can attain on [RoBERTa](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment){:target="_blank"} and [TweetEval](https://huggingface.co/datasets/cardiffnlp/tweet_eval){:target="_blank"}.


<div class="plot-row">
  <div class="plot-box"><div id="roberta_soft_stability" class="plot-inner"></div></div>
  <div class="plot-box"><div id="roberta_hard_stability" class="plot-inner"></div></div>
</div>

<script>
  // Plot both datasets
  plotFromJSON('/assets/images/soft_stability/blog_roberta_soft_stability.json', 'roberta_soft_stability', 'RoBERTa Soft Stability');
  plotFromJSON('/assets/images/soft_stability/blog_roberta_hard_stability.json', 'roberta_hard_stability', 'RoBERTa Hard Stability');
</script>



<!--
A caveat of the soft stability estimation is that it is inherently probabilistic, which directly contrasts with the deterministic style of hard stability.
To boost the estimation confidence, one can take more samples to better approximate the true soft stability rate.
-->


## Mild smoothing improves soft stability
Should we completely abandon smoothing?
Not necessarily!
Although the algorithm for certifying does not require a smoothed classifier, we empirically found that mildly smoothed models often have empirically improved stability rates.
Moreover, we can explain these empirical observations using techniques from [Boolean function analysis](https://en.wikipedia.org/wiki/Analysis_of_Boolean_functions).


<details>
<summary>Click for details</summary>
<div markdown="1">


The particular smoothing implementation we consider involves randomly masking (i.e., dropping, zeroing) features, which we define as follows.

{% include gallery id="gallery_smoothing_wrapper" layout="" caption="**Random masking of a classifier.** Randomly masked copies of the original input are given to a model and the outputs are averaged. Each feature is kept with probability $\lambda$, i.e., dropped with probability $1 - \lambda$. In this example, the task is to classify whether or not lung disease is present." %}


**Definition. [Random Masking]**
For an input $x \in \mathbb{R}^d$ and classifier $f: \mathbb{R}^d \to \mathbb{R}^m$, define the smoothed classifier as $\tilde{f}(x) = \mathbb{E}_{\tilde{x}} f(\tilde{x})$, where independently for each feature $x_i$, the smoothed feature is $\tilde{x}_i = x_i$ with probability $\lambda$, and $\tilde{x}_i = 0$ with probability $1 - \lambda$.
That is, **a smaller $\lambda$ means stronger smoothing.**
{: .notice--info}


In the [original context](https://debugml.github.io/multiplicative-smoothing/){:target="_blank"} of certifying hard stability, this was also referred to as *multiplicative smoothing* because the noise scales the input.
One can think of the smoothing parameter $\lambda$ as the probability that any given feature is kept, i.e., each feature is randomly masked (zeroed, dropped) with probability $1 - \lambda$.
Smoothing becomes stronger as $\lambda$ shrinks: at $\lambda = 1$, no smoothing occurs because $\tilde{x} = x$ always; at $\lambda = 1/2$, half the features of $x$ are zeroed out on average; at $\lambda = 0$, the classifier predicts on an entirely zeroed input because $\tilde{x} = 0_d$.


Importantly, we observe that smoothed classifiers can have improved soft stability, particularly for weaker models!
Below, we show examples for ViT and ResNet50.

<div id="plot-container">
  <div class="plot-row">
    <div class="plot-box">
      <div class="plot-inner" id="vit-stability-vs-lambda"></div>
    </div>
    <div class="plot-box">
      <div class="plot-inner" id="resnet50-stability-vs-lambda"></div>
    </div>
  </div>
</div>

<script>
// Function to create plot
function createPlot(elementId, data, title) {
    const traces = [
        {
            x: data.radii,
            y: data["lambda_1.0"],
            name: 'λ = 1.0',
            mode: 'lines',
            line: {width: 2},
            hoverinfo: 'x+y+name'
        },
        {
            x: data.radii,
            y: data["lambda_0.9"],
            name: 'λ = 0.9',
            mode: 'lines',
            line: {width: 2},
            hoverinfo: 'x+y+name'
        },
        {
            x: data.radii,
            y: data["lambda_0.8"],
            name: 'λ = 0.8', 
            mode: 'lines',
            line: {width: 2},
            hoverinfo: 'x+y+name'
        },
        {
            x: data.radii,
            y: data["lambda_0.7"],
            name: 'λ = 0.7', 
            mode: 'lines',
            line: {width: 2},
            hoverinfo: 'x+y+name'
        },
        {
            x: data.radii,
            y: data["lambda_0.6"],
            name: 'λ = 0.6', 
            mode: 'lines',
            line: {width: 2},
            hoverinfo: 'x+y+name'
        },
        {
            x: data.radii,
            y: data["lambda_0.5"],
            name: 'λ = 0.5', 
            mode: 'lines',
            line: {width: 2},
            hoverinfo: 'x+y+name'
        }
    ];

    const layout = {
        title: title,
        xaxis: {
            title: 'Perturbation Radius',
            showgrid: true,
            zeroline: true
        },
        yaxis: {
            title: 'Stability Rate', 
            range: [0, 1],
            showgrid: true,
            zeroline: true
        },
        showlegend: true,
        legend: {
            x: 0.1,
            y: 0.1,
            xanchor: 'left',
            yanchor: 'bottom',
            bgcolor: 'rgba(255,255,255,0.8)',
            bordercolor: '#ccc',
            borderwidth: 1
        },
        margin: {
            l: 40,
            r: 40, 
            b: 40,
            t: 40,
            pad: 4
        },
        hovermode: 'x unified'
    };

    Plotly.newPlot(elementId, traces, layout, {responsive: true});
}

// Load and plot ViT data
fetch('/assets/images/soft_stability/blog_vit_stability_vs_lambda.json')
    .then(response => response.json())
    .then(data => {
        createPlot('vit-stability-vs-lambda', data, 'ViT Stability vs. Smoothing');
    })
    .catch(error => {
        console.error('Error loading ViT JSON file:', error);
    });

// Load and plot ResNet50 data
fetch('/assets/images/soft_stability/blog_resnet50_stability_vs_lambda.json')
    .then(response => response.json())
    .then(data => {
        createPlot('resnet50-stability-vs-lambda', data, 'ResNet50 Stability vs. Smoothing');
    })
    .catch(error => {
        console.error('Error loading ResNet50 JSON file:', error);
    });
</script>


To study the relation between smoothing and stability rate, we use tools from [Boolean function analysis](https://en.wikipedia.org/wiki/Analysis_of_Boolean_functions){:target="_blank"}.
Our main theoretical finding is as follows.

**Main Result.**  Smoothing improves the lower bound on the stability rate by shrinking its gap to 1 by a factor of $\lambda$.
{: .notice--success}



In more detail, for any fixed input-explanation pair, the stability rate of any classifier $f$ and the stability rate of its smoothed variant $\tilde{f}$ have the following relationship:



$$
1 - Q \leq \tau_r (f) \,\, \implies\,\, 1 - \lambda Q \leq \tau_r (\tilde{f}),
$$

where $Q$ is a quantity that depends on $f$ (specifically, its Boolean Fourier spectrum) and the distance to the decision boundary.


Although this result is on a lower bound, it aligns with our empirical observation that smoothed classifiers tend to be more stable.
Interestingly, we found it challenging to bound this improvement using [standard techniques](https://arxiv.org/abs/2105.10386){:target="_blank"}.
This motivated us to develop novel theoretical tooling, which we leave the details and experiments for in the [paper](){:target="_blank"}.

</div>
</details>


## Conclusion
In this blog post, we explore a practical variant of stability guarantees that improves upon existing methods in the literature.

For more details, please check out our [paper](){:target="_blank"}, [code](https://github.com/helenjin/soft_stability/){:target="_blank"}, and [tutorial](https://github.com/helenjin/soft_stability/blob/main/tutorial.ipynb){:target="_blank"}.


Thank you for reading!
If you find our work helpful, please consider citing it!

```bibtex
@article{jin2025softstability,
 title={Probabilistic Stability Guarantees for Feature Attributions},
 author={Jin, Helen and Xue, Anton and You, Weiqiu and Goel, Surbhi and Wong, Eric},
 journal={arXiv},
 year={2025}
}
```
