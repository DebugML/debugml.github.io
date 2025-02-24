---
title: "Probabilistic Stability Guarantees for Feature Attributions"
layout: single
excerpt: "We introduce a relaxed variant of stability, called soft stability, with probabilistic guarantees for feature attributions."

header:
  overlay_filter: "0.75"
  overlay_image: /assets/images/soft_stability/stability_rocks.png
  teaser: /assets/images/soft_stability/stability_rocks.png
  actions:
    - label: "Paper"
      url: 
    - label: "Code"
      url: https://github.com/helenjin/ssg/tree/main

authors:
  - Helen Jin
  - Anton Xue
  - Weiqiu You
  - Surbhi Goel
  - Eric Wong


gallery_hard_vs_soft:
  - url: /assets/images/soft_stability/hard_vs_soft_real.png
    image_path: /assets/images/soft_stability/hard_vs_soft_real.png
    title: A visual example of certified radii by hard stability vs. soft stability.

gallery_unstable:
  - url: /assets/images/soft_stability/unstable.png
    image_path: /assets/images/soft_stability/unstable.png
    title: An unstable selection of features from SHAP.

gallery_algo:
  - url: /assets/images/soft_stability/algo.png
    image_path: /assets/images/soft_stability/algo.png
    title: Algorithm for Estimating Stability Rate.

gallery_soft_certifies_more:
  - url: /assets/images/soft_stability/vit_imagenet_soft.png
    image_path: /assets/images/soft_stability/vit_imagenet_soft.png
    title: Soft stability certifies more than hard stability (Soft).
  - url: /assets/images/soft_stability/vit_imagenet_hard.png
    image_path: /assets/images/soft_stability/vit_imagenet_hard.png
    title: Soft stability certifies more than hard stability (Hard).

---

<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [ ['$','$'], ["\\(","\\)"] ],
      processEscapes: true
    }
  });
</script>

<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

> Stability guarantees are an important tool for understanding how reliable feature attributions are. 
> However, existing certification methods rely on smoothed classifiers and yield overly conservative bounds that are not as informative.
> To address this, we introduce the concept of soft stability and propose a sampling-based certification algorithm that is both model-agnostic and sample-efficient.
> We find that our proposed probabilistic stability guarantees provide a better way to encapsulate the reliability of feature attributions on both vision and language tasks.

## Background
Powerful machine learning models are increasingly deployed in practice. 
However, their opacity presents a major challenge in being adopted in high-stake domains, where transparent and reliable explanations are needed in decision making. 
In healthcare, for instance, doctors require insights into the diagnostic steps to trust the model and integrate them into clinical practice effectively. 
Similarly, in the legal domain, attorneys must ensure that decisions reached with the assistance of models meet stringent judicial standards.

Due to their simplicity and generality, feature attribution-based methods are a popular choice for explanations.
A feature attribution method assigns a score to each input feature to indicate its importance to the model's prediction. Although attribution scores are typically real-valued, it is common to simplify them to a boolean mask by selecting only the top-$k$ most relevant features.


<!-- 
We need more reliable explanations.
but hard stability has fundamental limitations -->



## Hard Stability vs. Soft Stability
To ensure that a explanation is actually reliable, we study stability as a desirable property. 
We find that the previously proposed hard stability has fundamental limitations, and thus introduce an alternative, more relaxed variant of stability, called soft stability.


{% include gallery id="gallery_unstable" layout="" caption="**An unstable selection of features from SHAP.**\\
Although the image masked by the original explanation makes the same prediction as the original image (the second row vs. the first row), adding one patch to the explanation changes the highest predicted class from 'candon' to 'artichoke'." %}

Before getting into what stability entails, let us ground our discussion by first establishing that intuitively, “similar” attributions should *induce* the same prediction with respect to a given classifier and input.
Although various measures of similarity exist, we focus on additive perturbations. Specifically, an additive perturbed attribution is one that contains more information (features) than the original attribution, where the intuition is that adding more features to a “good quality” should not easily alter the prediction. 

**Definition.** [Additive Perturbations]
For an attribution $\alpha$ and radius $r > 0$, define its $r$-additive perturbation set as:
$$\\
\begin{equation*}
    \Delta_r (\alpha)
    = \{\alpha' \in \{0,1\}^n: \alpha' \geq \alpha,\, ||\alpha' - \alpha||_0 \leq r\},
\end{equation*}\\$$
where $\alpha' \geq \alpha$ iff each $\alpha_i ' \geq \alpha_i$ and $||\cdot||_0$ denotes the $\ell^0$ norm, which measures the number of non-zero coordinates.
{: .notice--info}

Intuitively, $\Delta_r (\alpha)$ represents the set of attributions that are at least as informative as $\alpha$, adding at most $r$ features. 
We can then study the robustness of an explanation by analyzing its **stability**: an attribution should be considered *stable* if adding a small number of features does not alter the classifier’s decision. 
First introduced in the [MuS blog post](https://debugml.github.io/multiplicative-smoothing/) as "incrementally stable", hard stability is defined as follows:

**Definition.** [Hard Stability]
For a classifier $f$ and input $x$, the explanation $\alpha$ is *hard-stable* with radius $r$ if:
$f(x \odot \alpha') \cong f(x \odot \alpha)$ for all $\alpha' \in \Delta_r$.
{: .notice--info}

Hard stability says an explanation is stable up to radius $r$ if for all cases of adding $r$ features, the prediction does not change.
Although this gives a strong argument, a key limitation is that hard stability can only give guarantees on smoothed classifiers, instead of the original classifiers.
Increased smoothness leads to larger certified radii but at the cost of accuracy. 
This trade-off arises because excessive smoothing reduces a model’s sensitivity, making it harder to distinguish between classes. 
Moreover, even when smoothing-based certification is feasible, the resulting certified radii are often conservative because the radii depend on a global property (the Lipschitz constant κ) to make local guarantees about feature perturbations.

To address these limitations, we look into a probabilistic relaxation of hard stability that avoids the need to smooth the classifier. 
We introduce the notion of soft stability for an explanation as follows: 

**Definition.** [Soft Stability]
For a classifier $f$ and input $x$, define the *stability rate* of attribution $\alpha$ at radius $r$ as:
$$\\ 
\begin{equation*}
    \tau_r (f, x, \alpha) = \text{Pr}_{\alpha' \sim \Delta_r} [f(x \odot \alpha') \cong f(x \odot \alpha)],
\end{equation*}$$
where $\alpha' \sim \Delta_r$ is uniformly sampled.
{: .notice--info}


While hard stability certifies whether all small perturbations to an attribution yield the same prediction, soft stability quantifies how often the prediction remains consistent. 
In general, compared to their hard variants, probabilistic guarantees are more flexible to apply and efficient to compute, and can certify larger radii. In the following example, we can see that hard stability only certifies up to 1 patch, while soft stabitiliy certifies up to 5 patches. 

{% include gallery id="gallery_hard_vs_soft" layout="" caption="**A visual example of certified radii by hard stability vs. soft stability.** \\
For an image of a penguin masked to show only the top 44% explanation by LIME, hard stability certifies that adding one patch won't change the prediction. In contrast, soft stability can certify adding up to 5 patches with a probabilistic guarantee." %}


## Certifying Soft Stability
Unlike hard stability, which requires destructively smoothing the classifier and often yields conservative guarantees, soft stability can be estimated efficiently for any classifier.
The key measure, the stability rate $\tau_r$, can be estimated via the following algorithm:

**Theorem.** [Estimation Algorithm]
Let $N \geq \frac{\log(2/\delta)}{2 \varepsilon^2}$
for any $\varepsilon > 0$ and $\delta > 0$.
For a classifier $f$, input $x$, explanation $\alpha$, and radius $r$, define the stability rate estimator:
$$\begin{equation*} 
    \hat{\tau}_r
    = \frac{1}{N} \sum_{i = 1}^{N} \mathbf{1}\big[f(x \odot \alpha^{(i)}) \cong f(x \odot \alpha)\big],
\end{equation*} \\$$
where $\alpha^{(1)}, \ldots, \alpha^{(N)} \sim \Delta_r (\alpha)$ are i.i.d. samples. Then, with probability $\geq 1 - \delta$, it holds that $|{\tau_r - \hat{\tau}_r}| \leq \varepsilon$.
{: .notice--danger}




{% include gallery id="gallery_algo" layout="" caption="**Visualization of the Algorithm for Estimating Stability Rate.**" %}

Notably, the required sample size $N$ depends only on $\varepsilon$ and $\delta$, since $\tau_r$ is a one-dimensional statistic.
Because $N$ is independent of $f$, the estimation algorithm scales linearly in the cost of evaluating $f$.
Moreover, certifying soft stability does not require deriving a smoothed classifier through a destructive smoothing classifier.
Unlike hard stability, which applies to the smoothed classifier $\tilde{f}$, soft stability provides robustness guarantees directly on the original classifier $f$.
This eliminates the need for a destructive smoothing process that risks degrading accuracy.

### Question: How much do soft stability certificates improve over hard stability?

We evaluate the attainable soft stability rates and compare them to their counterpart hard stability rates on different classification models. We explore vision and language settings in our paper, but for conciseness, we show only the vision example below.

{% include gallery id="gallery_soft_certifies_more" layout="" caption="**Soft stability certifies more than hard stability.**" %}

We observe that the attainable radii are much larger by soft stability than hard stability.
In particular, soft stability attains radii up to two orders of magnitude larger than hard stability does. We can see that soft stability effectively differentiates attribution methods, with LIME and SHAP showing a sizable advantage over IntGrad, MFABA, and random baselines across all radii. 
In contrast, hard stability certifies overly low radii for all methods, making it ineffective for distinguishing stability differences. 
Note that a caveat of the soft stability bounds is that they are inherently probabilistic, which directly contrasts with the deterministic style of hard stability. 
To remedy this, one can always take more samples to get closer to the true soft stability rate.

## Mild Smoothing Improves Soft Stability
Should we completely abandon smoothing? It turns out no, not necessarily. Although the soft stability certification algorithm does not explicitly require smoothing, we observe that mildly smoothing the model empirically improves stability rates.

We now introduce the multiplicative smoothing operator originally used to certify hard stability in the previous [MuS blog post](https://debugml.github.io/multiplicative-smoothing/), wherein the main idea is for the resulting smoothed classifier to be more robust to the inclusion and exclusion of features. 
This is achieved by randomly masking features in the following process. 

**Definition.** [Random Masking]
For any classifier $f$ and smoothing parameter $\lambda \in [0,1]$, define the random masking operator $M_\lambda$ as:
$$\begin{equation*}
    M_\lambda f (x)
    = \mathbb{E}_{z \sim \text{Bern} (\lambda)^n} f(x \odot z), \\
    \text{where \(z_1, \ldots, z_n \sim \text{Bern}(\lambda)\) are i.i.d. samples.}
\end{equation*}$$
{: .notice--info}

One can think of the smoothing parameter $\lambda$ as the probability that any given feature is kept.
That is, each feature is randomly masked (zeroed, dropped) with probability $1 - \lambda$.
We say that smoothing becomes stronger as $\lambda$ shrinks: at $\lambda = 1$, no smoothing occurs because $M_1 f(x) = f(x)$; at $\lambda = 1/2$, half the features of $x \odot z$ are zeroed out on average;
at $\lambda = 0$, the classifier predicts on an entirely zeroed input because $M_0 f(x) = f(\mathbf{0}_n)$. Random masking is also called multiplicative smoothing because the noise scales the input, unlike standard additive noising.

In addition to random masking, we use Boolean function analysis tools, which studies real-valued functions of Boolean-valued inputs, to analyze the masked version of feature attributions.

Our main theoretical finding is that smoothing improves the worst-case stability rate by a factor of $\lambda$.
{: .notice--danger}

Although this result is on a lower bound, it aligns with our empirical observation that smoothed classifiers tend to be more stable.
Interestingly, we found it challenging to bound the stability rate of $M_\lambda$-smoothed classifiers using standard Boolean analytic techniques.
This motivated us to develop novel analytic tooling, the process of which we describe in the next section.

<!-- (one paragraph and one informal statement of the theorem)
smoothing helps because boolean basis~~~ if you want to learn more about it, you can check out our paper
this method worked on all models, including smoothed models. as it turns out, we can prove smoothing improves the certificate
give some informal intuition -->

For more details and experiments, including those that address the questions of whether smoothing degrade accuracy and how smoothing affect soft stability, please see our [paper]().

<!-- ## Experimental Takeaways
### Soft Stability Certifies More Than Hard Stability

### Mildly Smoothing Preserves Accuracy

### Smoothing Improves Stability

### Stability Improves with Larger Selections -->


## Conclusion
In this post, we relax the constraints needed for hard stability and present probablistic stability guarantees as a formal property for feature attribution methods: a selection of features is *soft stable* if the additional inclusion of other features uniformly sampled does not alter its induced class. 
<!-- In practice, however, hard stability can be non-trivial to certify, and its smoothing requirement has performance tradeoffs.  -->

For more details in thoeretical proofs and experiments, kindly refer to our [paper]() and [code]().

## Citation
Thank you for taking the time to read through this blog post. 

Please cite our work if you find it helpful.
```bibtex
@article{jin2025softstability,
  title={Probabilistic Stability Guarantees for Feature Attributions}, 
  author={Jin, Helen and Xue, Anton and You, Weiqiu and Goel, Surbhi and Wong, Eric},
  journal={arXiv},
  year={2025}
}
```
