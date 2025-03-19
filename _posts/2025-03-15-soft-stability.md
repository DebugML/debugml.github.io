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
 - url: /assets/images/soft_stability/algo_new.png
   image_path: /assets/images/soft_stability/algo_new.png
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

gallery_smoothing_improves_stability:
 - url: /assets/images/soft_stability/smoothing_improves_stability.png
   image_path: /assets/images/soft_stability/smoothing_improves_stability.png
   title: We should not be using this particular image because it is a screenshot.

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

> Stability guarantees are an emerging tool for understanding how reliable explanations are.
> However, current methods rely on specialized architectures and give guarantees that are too conservative to be useful.
> To address these limitations, we introduce **soft stability**, a more general and flexible approach for certifying explanations that works with any model and gives more useful guarantees. 
> Our guarantees are orders of magnitude greater than existitng methods and can scale to be usable in practice in high-dimensional settings.


Powerful machine learning models are increasingly deployed in real-world applications. 
However, their opacity poses a significant challenge to safety, especially in high-stakes settings where interpretability is critical for decision-making. 
A common approach to explaining these models is through feature attributions methods, which highlight the input features that contribute most to a prediction.
However, these explanations that are the selected features are often brittle, as shown in the following figure.


{% include gallery id="gallery_unstable" layout="" caption="**(Anton: THIS IMAGE WILL BE REPLACED) An unstable selection of features from LIME.**\\
Given an input image (top), [LIME](https://github.com/marcotcr/lime){:target='_blank'} selects features it deems important for the classifier $f$'s prediction (middle). However, adding just four additional features causes the prediction to change from 'Walker hound' to 'Beagle' (bottom). This suggests that the explanation is highly sensitive to the inclusion of additional features, making it unreliable."%}


An ideal explanation should be *robust*: if a subset of features is genuinely explanatory for the prediction, then revealing any small set of additional features should not change the prediction, up to some tolerance threshold.
In fact, this is the notion of **hard stability**, which we explored in a [previous blog post](https://debugml.github.io/multiplicative-smoothing/){:target="_blank"}.
However, exactly computing this maximum hard stability tolerance is computationally challenging, while the lower-bound estimation algorithms that we developed suffer from two major drawbacks:

- They rely on *specialized architectures*, in particular smoothed classifiers, which constrain their applicability.
- Their guarantees are *overly conservative*, meaning they certify only small perturbations, limiting practical use.

In this work, we introduce **soft stability**, a novel approach to robustness that overcomes these limitations.
The core idea behind stability is to measure how an explanation's prediction changes as more features are revealed.
We illustrate this concept as follows.

{% include gallery id="gallery_hard_vs_soft" layout="" caption="**The difference between hard stability and soft stability.**" %}



<!--
[Eric] This notion of stability ... (dont say our previous work)
move this part up, just jump into soft stability in this section

[Our previous work](https://debugml.github.io/multiplicative-smoothing/){:target="_blank"} introduced **hard stability**, a property that ensures a prediction remains unchanged for perturbations up to a certain tolerance. 
However, determining this tolerance is challenging: computing the maximum tolerance is computationally expensive, while a lower bound requires specialized architectures (smoothed classifiers). 
Because it is difficult to provably guarantee to what point explanations remain hard stable, we propose an alternative approach.
-->

<!-- even then, the certifiable tolerance is often too small to be practical and lower than empirically observed limits. -->

## Soft stability: a more flexible and scalable guarantee


Although both hard stability and soft stability describe how predictions change as features are revealed, the fundamental difference lies in how they measure robustness.
We compare and contrast their definitions below.


<div class="notice--danger">
<strong> Definition. [Hard Stability] </strong>
An explanation is <strong> hard stable </strong> at radius $r$ if including up to any $r$ additional features does not change the prediction.
</div>

We use "radius" to refer to the perturbation size, i.e., the number of features added, following robustness conventions.
<!-- To further clarify naming conventions, this is was also known as [incemental stability](https://debugml.github.io/multiplicative-smoothing/#lipschitz-smoothness-for-incremental-stability). -->
This radius is also used as part of soft stability's definition.
But rather than measuring whether the prediction is *always* preserved, soft stability instead measures *how often* it is preserved.


<div class="notice--info">
<strong> Definition. [Soft Stability] </strong>
At radius $r$, an explanation's <strong> stability rate </strong> $\tau_r$ is the probability that adding up to $r$ additional features does not change the prediction. 
</div>


The stability rate provides a fine-grained measure of an explanation's robustness.
For example, two explanations may appear similar, but could in fact have very different robustness values.


{% include gallery id="gallery_lime_vs_shap_soft" layout="" caption="**Soft stability is a fine-grained measure of explanation robustness.**" %}


By shifting to a probabilistic perspective, soft stability offers a more refined view of explanation robustness.
Two key benefits follow:
1. **Model-agnostic certification**: The soft stability rate is efficiently computable for any classifier, whereas hard stability is only easy to certify for smoothed classifers.
2. **Practical guarantees**: The certificates for soft stability are much larger and more practically useful than those obtained from hard stability.

<!--
In the above 'Walker hound' example, the stability rate at radius $r = 4$ is $\tau_4 = 95.3$%.
Note that soft stability is a generalization of hard stability, which can only give a yes/no statement about robustness---in this case, the explanation is *not* hard stable at radius $r = 4$ because $\tau_4 < 100$%.
By simply shifting to a probabilistic perspective, soft stability offers a more refined view of explanation robustness.
The two key benefits are:
1. **Model-agnostic certification**: The soft stability rate is efficiently computable for any classifier, whereas hard stability is only easy to certify for smoothed classifers.
2. **Practical guarantees**: The certificates for soft stability are much larger and more practically useful than those obtained from hard stability.
-->

### Certifying soft stability: challenges and algorithms
At first, certifying soft stability (computing the stability rate) appears daunting.
If there are $m$ possible features that may be included at radius $r$, then there are $\mathcal{O}(m^r)$ many perturbations to check!
In fact, this combinatorial explosion is the same computational bottleneck encountered when one tries to naively certify hard stability.



<!--
## Technical details
[Eric] merge this part into the previous section, is too redundant currently


The fundamental difference between hard and soft stability lies in how they define robustness:

|  | Formal Mathematical Guarantees | Computational Requirements |
|----------------|-----------|----------------------------|
| **Hard Stability** | *All* perturbations up to $r$ features leave the prediction unchanged. | Expensive to certify; requires smoothed classifiers. |
| **Soft Stability** | The prediction remains unchanged *with high probability*. | Efficient, sample-based estimation. |

Both hard stability and soft stability describe how predictions change as features are revealed. 
Hard stability ensures that revealing up to $r$ features *always* preserves the prediction, while soft stability measures how often this holds. 
Hard stability is a stricter condition, guaranteeing invariance for all perturbations within $r$, whereas soft stability allows for occasional changes.
When the stability rate reaches 100%, soft stability becomes equivalent to hard stability. This is further illustrated in the figure below.


For the remainder of this section, we will discuss the computational details for certifying hard stability and soft stability. 
-->

<!--
[Eric] two paragraphs only
title: computational problems, why its hard
next title paragraph: here is our algorithm to solve this
-->

<!--
### Computational difficulties in certifying hard stability
Certifying hard stability is challenging because determining the maximum tolerance---the largest perturbation radius at which the prediction remains unchanged---requires exhaustively checking all possible perturbations up to that radius to ensure none cause a prediction flip. 
When a classifier lacks mathematically convenient properties, this process becomes computationally intractable, especially in high-dimensional spaces where the number of possible perturbations grows rapidly. 
For instance, if there are $m$ possible features that can be included, the total number of cases to check up to radius $r \leq m$ is given by $\binom{m}{1} + \binom{m}{2} + \cdots + \binom{m}{r}$, leading to a combinatorial explosion of $\mathcal{O}(m^r)$ complexity that makes brute-force verification impractical for real-world models.

To address this challenge, prior work relies on [smoothing classifiers](#mild-smoothing-improves-soft-stability),
to attain mathematical properties more amenable for hard stability certification.
Specifically, computing a lower bound on the maximum tolerance becomes more tractable with a smoothed classifier.
However, smoothed classifiers come with significant drawbacks. 
First, the resulting stability guarantees only apply to the smoothed classifier rather than the original one.
Second, because smoothing often degrades accuracy, the guarantees are with respect to an (often) worse model.
Third, the bounds tend to be too conservative, certifying much smaller perturbation radii than what empirical sampling suggests is achievable ([Xue et al. 2024, Section 4.1](https://arxiv.org/abs/2307.05902){:target="_blank"}).
This discrepancy limits the practical utility of smoothed classifiers for stability certification.
-->

Fortunately, we can efficiently **estimate** the stability rate to a high accuracy using standard sampling techniques from statistics.
This procedure is summarized in the following figure.


{% include gallery id="gallery_algo" layout="" caption="**Estimating the stability rate $\tau_r$.** An estimator $\hat{\tau}_r$ constructed using $N \geq \log(2/\delta) / (2 \varepsilon^2)$ perturbation samples will, with probability at least $1 - \delta$, attain an accuracy of $\lvert \hat{\tau}_r - \tau_r \rvert \leq \varepsilon$. We give a reference implementation in our [tutorial notebook](https://github.com/helenjin/soft_stability/blob/main/tutorial.ipynb){:target='_blank'}. " %}


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


<!--
The resulting estimated soft stability rate $\hat{\tau}_r$ is accurate to the true stability rate $\tau_r$ with high confidence.
For example, for $\delta = 0.05$ and $\varepsilon = 0.1$, taking $N \geq 185$ samples allows the estimator $\hat{\tau}_r$ to fall within $0.1$ of the true $\tau_r$ at least $95$% of the time. 

with probability $\geq 1 - \delta$, we have that $\lvert \hat{\tau}_r - \tau_r \rvert \leq \varepsilon$.
The technical details follow by standard concentration theorems on sampling. 
We give a demonstration of the stability rate estimation algorithm in the following [tutorial notebook](https://github.com/helenjin/soft_stability/blob/main/tutorial.ipynb){:target="_blank"}. 

[Eric] add figure, breaks up text

fix this
-->

There are three main computational benefits of estimating soft stability in this manner.
First, the estimation algorithm is model-agnostic, which means that soft stability can be certified for any model, not just smoothed ones --- in contrast to hard stability.
Second, this algorithm is sample-efficient: the number of samples depends only on the hyperparameters $\varepsilon$ and $\delta$, meaning that the runtime cost scales linearly with the cost of running the classifier.
Thirdly, as we show next, soft stability certificates are much less conservative than hard stability, making them more practical for giving fine-grained and meaningful measures of explanation robustness.

<!--
[Eric] say this later

Thirdly, as our subsequent experiments will show, soft stability certificates are much less conservative than hard stability, making them more practical for measuring the robustness of explanations.
Next, we give some experimental evaluations that compare soft and hard stability in practice.
-->



## Experiments

We next consider how soft stability compares with hard stability in practice. 
We empirically evaluate on vision and language tasks.

We first show the stability rates attainble with a [Vision Transformer](https://huggingface.co/google/vit-base-patch16-224){:target="_blank"} mdoel over $1000$ [samples from ImageNet](https://github.com/helenjin/soft_stability/tree/main/imagenet-sample-images).

[Eric] make it more integrated instead of saying it all at the end, also condense
what we say
make these graphs html native 
put data into json format
can use claude


{% include gallery id="gallery_soft_certifies_more" layout="" caption="**Soft stability certifies more than hard stability.** LIME and SHAP showing a sizable advantage over IntGrad, MFABA, and random baselines across all radii." %}

Impressively, the attainably radii at which soft stability are much larger than that of hard stability, by up to two orders of magnitude.
Although this trend is most pertinent for vision, it is also present for language models.
Next, we show the stability rates we can attain on [RoBERTa](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment){:target="_blank"} and [TweetEval](https://huggingface.co/datasets/cardiffnlp/tweet_eval){:target="_blank"}.

{% include gallery id="gallery_soft_certifies_more_tweeteval" layout="" caption="**Soft stability certifies more than hard stability.** " %}


A caveat of the soft stability estimation is that it is inherently probabilistic, which directly contrasts with the deterministic style of hard stability.
To boost the estimation confidence, one can take more samples to better approximate the true soft stability rate.

<!--
We observe that the attainable radii from soft stability are much larger, by up to two orders of magnitude, than those obtained by hard stability.
This trend is most pertinent in the vision task but is also present in the language task.
Furtheremore, for the vision task, we can see that soft stability effectively differentiates various explanation methods, in contrast to hard stability.
Note that a caveat of the soft stability estimation is that it is inherently probabilistic, which directly contrasts with the deterministic style of hard stability.
To boost the estimation confidence, one can take more samples to better approximate the true soft stability rate.
-->


## Mild smoothing improves soft stability
Should we completely abandon smoothing?
Not necessarily!
Although the algorithm for certifying does not require a smoothed classifier, we empirically found that mildly smoothed models often have empirically improved stability rates.
Moreover, we can explain these empirical observations using techniques from [Boolean function analysis](https://en.wikipedia.org/wiki/Analysis_of_Boolean_functions).

<!--
[Eric] we can show empirically that theory stuff
-->
<!-- found that mildly smoothed models often have empirically improved stability rates. -->

<details>
<summary>Click for details</summary>
<div markdown="1">


The particular smoothing implementation we consider involves randomly masking (i.e., dropping, zeroing) features, which we define as follows.

**Definition. [Random Masking]**
For an input $x \in \mathbb{R}^d$ and classifier $f: \mathbb{R}^d \to \mathbb{R}^m$, define the smoothed classifier as $\tilde{f}(x) = \mathbb{E}_{\tilde{x}} f(\tilde{x})$, where independently for each feature $x_i$, the smoothed feature is $\tilde{x}_i = x_i$ with probability $\lambda$, and $\tilde{x}_i = 0$ with probability $1 - \lambda$.
{: .notice--info}


In the [original context](https://debugml.github.io/multiplicative-smoothing/){:target="_blank"} of certifying hard stability, this was also referred to as *multiplicative smoothing* because the noise scales the input.
One can think of the smoothing parameter $\lambda$ as the probability that any given feature is kept, i.e., each feature is randomly masked (zeroed, dropped) with probability $1 - \lambda$.
Smoothing becomes stronger as $\lambda$ shrinks: at $\lambda = 1$, no smoothing occurs because $\tilde{x} = x$ always; at $\lambda = 1/2$, half the features of $x$ are zeroed out on average; at $\lambda = 0$, the classifier predicts on an entirely zeroed input because $\tilde{x} = 0_d$.


Importantly, we observe that smoothed classifiers can have improved soft stability!

{% include gallery id="gallery_smoothing_improves_stability" layout="" caption="**[Anton] this image is a screenshot.**" %}

To study the relation between smoothing and stability rate, we use tools from [Boolean function analysis](https://en.wikipedia.org/wiki/Analysis_of_Boolean_functions){:target="_blank"}.
Our main theoretical finding is as follows.

**Main Result.**  Smoothing improves the worst-case stability rate by a factor of $\lambda$.
{: .notice--success}



In more detail, for any fixed input-explanation pair, the stability rate of any classifier $f$ and the stability rate of its smoothed variant $\tilde{f}$ have the following relationship:



$$
1 - \mathcal{Q} \leq \tau_r (f) \,\, \implies\,\, 1 - \lambda \mathcal{Q} \leq \tau_r (\tilde{f}),
$$

where $\mathcal{Q}$ is a quantity that depends on $f$ (specifically, its Boolean Fourier spectrum).

<!-- 
 and the stability rates are computed with respect to $x$ and $\alpha$. -->


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


