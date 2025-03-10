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
 - url: /assets/images/soft_stability/hard_vs_soft_real.png
   image_path: /assets/images/soft_stability/hard_vs_soft_real.png
   title: A visual example of certified radii by hard stability vs. soft stability.


gallery_unstable:
 - url: /assets/images/soft_stability/unstable.png
   image_path: /assets/images/soft_stability/unstable.png
   title: An unstable selection of features from SHAP.


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

<!-- condense all this to 2 paragraphs:
1) feature attributions are unreliable
2) stability was introduced in blog post and its limitations, talk about hard stability vs soft stability definitions in the soft stability section
 -->

<!-- - feature attributions are unreliable, here is what we mean
- figure
- why unstable explanation is undesirable
- we study this problem in a previous blog post
- stability
- current certificate requires smoothing and is overly conservative
- limitations -->

Powerful machine learning models are increasingly deployed in real-world applications. 
However, their opacity poses a significant challenge to safety, especially in high-stakes settings where interpretability is critical for decision-making. 
A common approach to explaining these models is through feature attributions methods, which highlight the input features that contribute most to a prediction.
However, these explanations that are the selected features are often brittle, as shown in the following figure.


{% include gallery id="gallery_unstable" layout="" caption="**An unstable selection of features from LIME.**\\
Given an input image (top), [LIME](https://github.com/marcotcr/lime){:target='_blank'} selects features it deems important for the classifier $f$'s prediction (middle). However, adding just four additional features causes the prediction to change from 'Walker hound' to 'Beagle' (bottom). This suggests that the explanation is highly sensitive to the inclusion of additional features, making it unreliable."%}

An ideal explanation should be *robust*: if a subset of features is genuinely explanatory for the prediction, then revealing additional features should not cause the prediction to change. 
In our [previous blog post](https://debugml.github.io/multiplicative-smoothing/){:target="_blank"}, we introduced the concept of **stability guarantees**, which aim to certify the robustness of explanations. However, existing methods suffer from two major limitations:
- They rely on *specialized architectures*, in particular smoothed classifiers, which constrain their applicability.
- Their guarantees are *overly conservative*, meaning they certify only small perturbations, limiting practical use.

In this work, we introduce soft stability, a novel approach to robustness that overcomes these limitations.


<!-- ## Soft stability: an improved, probabilistic way to measure explanation robustness -->
## Improving explanation robustness with soft stability

The core idea behind stability is to measure how an explanation's prediction changes as more features are revealed.
We illustrate this concept as follows.

{% include gallery id="gallery_algo" layout="" caption="When revealing up to $r=4$ features of a given explanation uniformly at random, the prediction remains unchanged 95.3% of the time." %}

[Our previous work](https://debugml.github.io/multiplicative-smoothing/){:target="_blank"} introduced **hard stability**, a property that ensures a prediction remain unchanged for perturbations up to a certain tolerance. 
However, determining this tolerance is challenging: computing the maximum tolerance is computationally expensive, while a lower bound requires specialized architectures (smoothed classifiers). 
Because it is difficult to provably guarantee to what point explanations remain hard stable, we propose an alternative approach.

<!-- even then, the certifiable tolerance is often too small to be practical and lower than empirically observed limits. -->

### Soft stability: a more flexible alternative
Instead of requiring that *all* perturbations do not flip the prediction, we propose a probabilistic approach that measures how often the prediction changes:

<div class="notice--info">
<strong> Definition. [Soft Stability] </strong> <br>
At radius $r$, an explanation's <strong> stability rate </strong> $\tau_r$ is the probability that adding up to $r$ additional features does not change the prediction. 
<br> <br>
<em> Note: </em> Following robustness conventions, we refer to perturbuation size as "radius".
</div>

In the above 'Walker hound' example, the stability rate at radius $r = 4$ is $\tau_4 = 95.3$%.
Note that soft stability is a generalization of hard stability, which can only give a yes/no statement about robustness---in this case, the explanation is *not* hard stable at radius $r = 4$ because $\tau_4 < 100$%.
By simply shifting to a probabilistic perspective, soft stability offers a more refined view of explanation robustness.
The two key benefits are:
1. **Model-agnostic certification**: The soft stability rate is efficiently computable for any classifier, whereas hard stability is only easy to certify for smoothed classifers.
2. **Practical guarantees**: The certificates for soft stability are much larger and more practically useful than those obtained from hard stability.


In the next sections, we provide detailed comparisons between hard and soft stability and demonstrate the practical benefits of our new approach.


## Technical details

The fundamental difference between hard and soft stability lies in how they define robustness:

|  | Formal Mathematical Guarantees | Computational Requirements |
|----------------|-----------|----------------------------|
| **Hard Stability** | *All* perturbations up to $r$ features leave the prediction unchanged. | Expensive to certify; requires smoothed classifiers. |
| **Soft Stability** | The prediction remains unchanged *with high probability*. | Efficient, sample-based estimation. |

Both hard stability and soft stability describe how predictions change as features are revealed. 
Hard stability ensures that revealing up to $r$ features *always* preserves the prediction, while soft stability measures how often this holds. 
Hard stability is a stricter condition, guaranteeing invariance for all perturbations within $r$, whereas soft stability allows for occasional changes. When the stability rate reaches 100%, soft stability becomes equivalent to hard stability. This is further illustrated in the figure below.

{% include gallery id="gallery_hard_vs_soft" layout="" caption="**A visual example of certified radii by hard stability vs. soft stability.** \\
For an image of a penguin masked to show only the top 44% explanation by LIME, hard stability certifies that adding one patch won't change the prediction. In contrast, soft stability can certify adding up to 5 patches with a probabilistic guarantee." %}

For the remainder of this section, we will discuss the computational details for certifying hard stability and soft stability. 

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


### Sample-efficient algorithm for certifying soft stability
In contrast to the computationally expensive approach required for certifying hard stability, soft stability---specifically the stability rate---can be efficiently estimated using a standard sampling process. We describe this algorithm below.

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

The resulting estimated soft stability rate $\hat{\tau}_r$ is accurate to the true stability rate $\tau_r$ with high confidence.
For example, for $\delta = 0.05$ and $\varepsilon = 0.1$, taking $N \geq 185$ samples allows the estimator $\hat{\tau}_r$ to fall within $0.1$ of the true $\tau_r$ at least $95$% of the time. 

In other words, with probability $\geq 1 - \delta$, we have that $\lvert \hat{\tau}_r - \tau_r \rvert \leq \varepsilon$.
The technical details follow by standard concentration theorems on sampling. 
We give a demonstration of the stability rate estimation algorithm in the following [tutorial notebook](https://github.com/helenjin/soft_stability/blob/main/tutorial.ipynb){:target="_blank"}. 


There are three main computational benefits of estimating soft stability in this manner.
First, the estimation algorithm is model-agnostic, which means that soft stability can be certified for any model, not just smoothed ones --- in contrast to hard stability.
Second, this algorithm is sample-efficient: the number of samples depends only on the hyperparameters $\varepsilon$ and $\delta$, meaning that the runtime cost scales linearly with the cost of running the classifier.
Thirdly, as our subsequent experiments will show, soft stability certificates are much less conservative than hard stability, making them more practical for measuring the robustness of explanations.

Next, we give some experimental evaluations that compare soft and hard stability in practice.



## Experiments

We next consider how soft stability compares with hard stability in practice. 
We empirically evaluate on vision and language tasks.

Below, we show the stability rates we can attain on [Vision Transformer](https://huggingface.co/google/vit-base-patch16-224){:target="_blank"}, when taking $1000$ examples from ImageNet.


{% include gallery id="gallery_soft_certifies_more" layout="" caption="**Soft stability certifies more than hard stability.** LIME and SHAP showing a sizable advantage over IntGrad, MFABA, and random baselines across all radii." %}

Next, we show the stability rates we can attain on [RoBERTa](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment){:target="_blank"} and [TweetEval](https://huggingface.co/datasets/cardiffnlp/tweet_eval){:target="_blank"}.

{% include gallery id="gallery_soft_certifies_more_tweeteval" layout="" caption="**Soft stability certifies more than hard stability.** " %}

We observe that the attainable radii from soft stability are much larger, by up to two orders of magnitude, than those obtained by hard stability.
This trend is most pertinent in the vision task but is also present in the language task.
Furtheremore, for the vision task, we can see that soft stability effectively differentiates various explanation methods, in contrast to hard stability.
Note that a caveat of the soft stability estimation is that it is inherently probabilistic, which directly contrasts with the deterministic style of hard stability.
To boost the estimation confidence, one can take more samples to better approximate the true soft stability rate.


## Mild smoothing improves soft stability
Should we completely abandon smoothing?
It turns out no, not necessarily.
Although the algorithm for certifying does not require a smoothed classifier, we found that mildly smoothed models often have empirically improved stability rates.

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


