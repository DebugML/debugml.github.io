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

Powerful machine learning models are increasingly deployed in practice.
However, their opacity presents a major challenge when adopted in high-stake domains, where transparent and reliable explanations are needed in decision-making.
In healthcare, for instance, doctors require insights into the diagnostic steps to trust the model and integrate them into clinical practice effectively.
Similarly, in the legal domain, attorneys must ensure that decisions reached with the assistance of models meet stringent judicial standards.

## What makes an explanation unreliable?
Due to their simplicity and generality, feature attribution methods are a popular choice for explanations.
Feature attributions work by selecting the features most important for a prediction. 
Intuitively, revealing more features to an explanation that already induces the correct prediction should not change the outcome.
However, this selection is often unstable even to small changes, as we illustrate in the figure below.


{% include gallery id="gallery_unstable" layout="" caption="**An unstable selection of features from LIME.**\\
Given the input image (top), [LIME](https://github.com/marcotcr/lime) gives an explanation by selecting the features it thinks are most important to the classifier $f$'s prediction (middle). Although the explanation induces the same prediction as the full image, adding four additional features to it causes the prediction to change from 'Walker hound' to 'Beagle' (bottom).
This is not desirable because it suggests that the explanation given by LIME is sensitive to the addition of new information." %}

<!-- ## Hard Stability vs. Soft Stability -->
<!-- Intuitively, revealing more features to an explanation that already induces the correct prediction should not change the outcome. In the example above, feature selection created by keeping the original features and adding four additional features caused the prediction to change.  -->
To quantify and prevent this type of behavior, we describe revealing extra features by using additive perturbations.


**Definition.** 
Given an explanation made up of a selection of features, an **additive perturbation** adds more features to the original selection.
<!-- For an attribution $\alpha$ and radius $r > 0$, define its $r$-additive perturbation set as:
$$\\
\begin{equation*}
   \Delta_r (\alpha)
   = \{\alpha' \in \{0,1\}^n: \alpha' \geq \alpha,\, ||\alpha' - \alpha||_0 \leq r\},
\end{equation*}\\$$
where $\alpha' \geq \alpha$ iff each $\alpha_i ' \geq \alpha_i$ and $||\cdot||_0$ denotes the $\ell^0$ norm, which measures the number of non-zero coordinates. -->
{: .notice--info}

<!-- Intuitively, $\Delta_r (\alpha)$ represents the set of attributions that are at least as informative as $\alpha$, adding at most $r$ features. -->
Using additive perturbations, we can then study the robustness of an explanation by analyzing its **stability**: an explanation is *hard stable* if adding any number of features, up to a certain tolerance, does not alter the classifier’s decision.
First introduced in a previous [blog post](https://debugml.github.io/multiplicative-smoothing/) as "incrementally stable", hard stability is defined as follows:

## Hard stability (previous work) and its limitations


**Definition.**
An explanation is **hard stable with radius $r$** if adding up to $r$ more features does not alter its prediction.
{: .notice--info}

Hard stability measures how many additive perturbations an explanation can tolerate before changing predictions.
However, finding the maximum tolerable radius can be computationally intractable in practice: if the classifier lacks favorable mathematical properties, then one must exhaustively check an impractically large number of possible perturbations to certify stability.


In a previous [blog post](https://debugml.github.io/multiplicative-smoothing/), we address this challenge by introducing *multiplicative smoothing*, which transforms the original classifier $f$ into a smoothed classifier $\tilde{f}$ that is more robust to additive perturbations.
In particular, given an input and explanation, a smoothed classifier can give a *certified radius*, which is the guaranteed number of features that can be added to the explanation without altering the prediction. 


However, smoothing-based hard stability guarantees are often overly conservative: the certified radii are often far smaller than the maximum tolerance, as suggested by empirical sampling.
For instance, the certified radii might state that some explanation may tolerate adding up to $4$ features without changing the prediction.
Still, we may fail to find such a prediction-altering perturbation even when extensively sampling at radii of up to $20$ features.
Moreover, the computed guarantees apply to the smoothed classifier rather than the original one, which is undesirable because smoothed classifers often have worse accuracy.
In other words, the existing stability guarantees are conservative and with respect to a smoothed classifier that is possibly inaccurate.

## Soft stability: an improved, probabilistic relaxation of hard stability
To overcome these limitations, we seek alternative ways to evaluate explanation robustness that retain the key intuition behind hard stability but avoid its computational drawbacks.
Our key insight in this work is to probabilistically relax hard stability: rather than ensuring *every* perturbation up to a given radius preserves the prediction, we instead measure how often this property holds.

**Definition. [Soft Stability]**
At radius $r$, an explanation's **stability rate** $\tau_r$ is the probability that adding up to $r$ additional features does not change the prediction.
{: .notice--info}


The stability rate is the prinicipal metric in soft stability and is computable for any classifier, not just smoothed ones.
Importantly, this shift to a probabilistic perspective allows us to obtain less restrictive statements than hard stability.
In fact, soft stability is a generalization of hard stability: if at some radius $r$ an explanation has stability rate $\tau_r = 1$, then it is also hard stable with radius $r$.
We visualize the difference between hard and soft stability in the following figure.



{% include gallery id="gallery_hard_vs_soft" layout="" caption="**A visual example of certified radii by hard stability vs. soft stability.** \\
For an image of a penguin masked to show only the top 44% explanation by LIME, hard stability certifies that adding one patch won't change the prediction. In contrast, soft stability can certify adding up to 5 patches with a probabilistic guarantee." %}

A high stability rate indicates that the model's explanation is reliable, meaning small modifications to the selected features are unlikely to change the prediction. 

So, how does one compute the stability rate?
Although this is also computationally intractable, we can efficiently estimate it to a high degree of accuracy and confidence!
We illustrate this idea in the following figure.


{% include gallery id="gallery_algo" layout="" caption="**Algorithm for estimating the stability rate $\tau_r$.** To estimate the soft stability at radius $r$ to accuracy $\varepsilon$ and confidence $1 - \delta$, it suffices to take $N \geq \frac{\log(2/\delta)}{2 \varepsilon^2}$ samples uniformly from the additive perturbations of size $\leq r$. That is, we uniformly sample from the set of $\alpha'$ where $\alpha' \supseteq \alpha$ and $\lvert \alpha' \setminus \alpha \rvert \leq r$. The stability rate estimator $\hat{\tau}_r$ will satisfy $\lvert \hat{\tau}_r - \tau_r \rvert \leq \varepsilon$ with probability $\geq 1 - \delta$." %}


The resulting estimated soft stability rate $\hat{\tau}_r$ is accurate to the true stability rate $\tau_r$ with high confidence.
In other words, with probability $\geq 1 - \delta$, we have that $\lvert \hat{\tau}_r - \tau_r \rvert \leq \varepsilon$.
The technical details follow by standard theoretical concentrations on sampling. 
We give a demonstration of this estimation process in our [tutorial notebook](https://github.com/helenjin/soft_stability/blob/main/tutorial.ipynb). 


There are three main computational benefits of estimating soft stability in this manner.
First, the estimation algorithm is model-agnostic, which means that soft stability can be certified for any model, not just smoothed ones --- in contrast to hard stability.
Second, this algorithm is sample-efficient: the number of samples depends only on the hyperparameters $\varepsilon$ and $\delta$, meaning that the runtime cost scales linearly with the cost of running the classifier.
Thirdly, as our subsequent experiments will show, soft stability certificates are much less conservative than hard stability, making them more practical for measuring the robustness of explanations.

<!-- We give a demonstration of this estimation process in our [tutorial notebook](https://github.com/helenjin/soft_stability/blob/main/tutorial.ipynb). 
Estimating the stability rate is model-agnostic, which means we can apply this to any classifier, not just smoothed ones. 
To validate this, we next present empirical results comparing soft and hard stability across different models and feature attribution methods, demonstrating that soft stability yields significantly larger certified radii while maintaining model accuracy.
This implies that soft stability certificates are not only more applicable to models but also less conservative, providing stronger, more practical guarantees without the need for smoothing. -->

### Question: How much do soft stability certificates improve over hard stability?
We next consider how soft stability compares with hard stability in practice. 
We empirically evaluate on vision and language tasks.

Below, we show the stability rates we can attain on [Vision Transformer](https://huggingface.co/google/vit-base-patch16-224), when taking $1000$ examples from ImageNet.

<!-- 
We evaluate the attainable soft stability rates and compare them to their counterpart hard stability rates on different classification models. We explore vision and language settings in our paper, but for conciseness, we show only the vision example below. -->

{% include gallery id="gallery_soft_certifies_more" layout="" caption="**Soft stability certifies more than hard stability.** LIME and SHAP showing a sizable advantage over IntGrad, MFABA, and random baselines across all radii." %}

Next, we show the stability rates we can attain on [RoBERTa]() and [TweetEval](https://huggingface.co/datasets/cardiffnlp/tweet_eval).

{% include gallery id="gallery_soft_certifies_more_tweeteval" layout="" caption="**Soft stability certifies more than hard stability.** " %}

We observe that the attainable radii from soft stability are much larger, by up to two orders of magnitude, than those obtained by hard stability.
This trend is most pertinent in the vision task but is also present in the language task.
Furtheremore, for the vision task, we can see that soft stability effectively differentiates various explanation methods, in contrast to hard stability.
Note that a caveat of the soft stability estimation is that it is inherently probabilistic, which directly contrasts with the deterministic style of hard stability.
To boost the estimation confidence, one can take more samples to better approximate the true soft stability rate.

## Mild Smoothing Improves Soft Stability
Should we abandon smoothing?
It turns out no, not necessarily.
Although the algorithm for certifying does not require a smoothed classifier, we found that mildly smoothed models often have empirically improved stability rates.

The particular type of smoothing we consider was introduced in our previous [blog post](https://debugml.github.io/multiplicative-smoothing/), which we call *multiplicative smoothing*.
One might alternatively think of this as randomized masking (i.e., dropping, zeroing) of features, which we describe next.

**Definition. [Random Masking]**
For an input $x \in \mathbb{R}^d$ and classifier $f$, define the smoothed classifier as $\tilde{f}(x) = \mathbb{E}_{\tilde{x}} f(\tilde{x})$, where independently for each feature $x_i$, the smoothed feature is $\tilde{x}_i = x_i$ with probability $\lambda$, and $\tilde{x}_i = 0$ with probability $1 - \lambda$.
<!-- For any input $x \in \mathbb{R}^d$, define the smoothed classifier as $\tilde{f}(x) = \mathbb{E} f(\tilde{x})$, where for all $i = 1, \ldots, d$, we independently have: $\tilde{x}_i = x_i$ with probability $\lambda$, and $\tilde{x}_i = 0$ with probability $1 - \lambda$. -->
<!-- 
For any classifier $f$ and smoothing parameter $\lambda \in [0,1]$, define the random masking operator $M_\lambda$ as:
$$\begin{equation*}
   M_\lambda f (x)
   = \mathbb{E}_{z \sim \text{Bern} (\lambda)^n} f(x \odot z), \\
   \text{where \(z_1, \ldots, z_n \sim \text{Bern}(\lambda)\) are i.i.d. samples.}
\end{equation*}$$ -->
{: .notice--info}


One can think of the smoothing parameter $\lambda$ as the probability that any given feature is kept.
That is, each feature is randomly masked (zeroed, dropped) with probability $1 - \lambda$.
We say that smoothing becomes stronger as $\lambda$ shrinks: at $\lambda = 1$, no smoothing occurs because $\tilde{x} = x$ always; at $\lambda = 1/2$, half the features of $x$ are zeroed out on average; at $\lambda = 0$, the classifier predicts on an entirely zeroed input because $\tilde{x} = 0_d$. 
Random masking is also called multiplicative smoothing because the noise scales the input.


In addition to random masking, we use Boolean function analysis, which studies real-valued functions of Boolean-valued inputs, as tools to analyze the masked version of feature attributions. Our main theoretical finding is as follows:


**Main Result.**  Smoothing improves the worst-case stability rate by a factor of $\lambda$.
{: .notice--danger}

In more detail, given some classifier $f$, input $x$, and explanation $\alpha$.
Let $\mathcal{Q}$ be a quantity that depends on $f$ (specifically, its Boolean Fourier spectrum).
Then, the stability rate between the original classifier $f$ and a smoothed classifier $\tilde{f}$ satisfies the relation:

$$
1 - \mathcal{Q} \leq \tau_r (f, x, \alpha) \,\, \implies\,\, 1 - \lambda \mathcal{Q} \leq \tau_r (\tilde{f}, x, \alpha)
$$


Although this result is on a lower bound, it aligns with our empirical observation that smoothed classifiers tend to be more stable.
Interestingly, we found it challenging to bound this improvement using standard Boolean analytic techniques.
This motivated us to develop novel analytic tooling, which we leave the details and experiments for in the [paper]().


<!-- (one paragraph and one informal statement of the theorem)
smoothing helps because boolean basis~~~ if you want to learn more about it, you can check out our paper
this method worked on all models, including smoothed models. as it turns out, we can prove smoothing improves the certificate
give some informal intuition -->


<!-- For more details and experiments, including those that address the questions of whether smoothing degrade accuracy and how smoothing affect soft stability, please see our [paper](). -->


<!-- ## Experimental Takeaways
### Soft Stability Certifies More Than Hard Stability


### Mildly Smoothing Preserves Accuracy


### Smoothing Improves Stability


### Stability Improves with Larger Selections -->




## Conclusion
In this blog post, we explore a practical variant of stability guarantees that improves upon existing methods in the literature.
For more details, please check out our [paper]() and [code]().


Thank you for reading!
If you find our work helpful, please consider citing it.

```bibtex
@article{jin2025softstability,
 title={Probabilistic Stability Guarantees for Feature Attributions},
 author={Jin, Helen and Xue, Anton and You, Weiqiu and Goel, Surbhi and Wong, Eric},
 journal={arXiv},
 year={2025}
}
```



