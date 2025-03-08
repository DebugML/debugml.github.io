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
Given an input image (top), [LIME](https://github.com/marcotcr/lime) selects features it deems important for the classifier $f$'s prediction (middle). However, adding just four additional features causes the prediction to change from 'Walker hound' to 'Beagle' (bottom). This suggests that the explanation is highly sensitive to the inclusion of additional features, making it unreliable."%}

An ideal explanation should be *robust*: if a subset of features is genuinely explanatory for the prediction, then revealing additional features should not cause the prediction to change. 
In our [previous blog post](https://debugml.github.io/multiplicative-smoothing/), we introduced the concept of **stability guarantees**, which aim to certify the robustness of explanations. However, existing methods suffer from two major limitations:
- They rely on *specialized architectures*, in particular smoothed classifiers, which constrain their applicability.
- Their guarantees are *overly conservative*, meaning they certify only small perturbations, limiting practical use.

In this work, we introduce soft stability, a novel approach that overcomes these limitations.
Unlike prior methods, soft stability provides probabilistic guarantees that scale effectively to high-dimensional settings while remaining model-agnostic.


<!-- ## Soft stability: an improved, probabilistic way to measure explanation robustness -->
## Improving explanation robustness with soft stability

The core idea behind stability is to measure how an explanation's prediction changes as more features are revealed.
We illustrate this concept as follows:

{% include gallery id="gallery_algo" layout="" caption="When randomly revealing up to $r=4$ features to a given explanation, the prediction remains unchanged 95.3% of the time." %}

[Our previous work](https://debugml.github.io/multiplicative-smoothing/) introduced **hard stability**, a property that ensures a prediction remain unchanged for perturbations up to a certain tolerance. 
However, determining this tolerance is challenging: computing the maximum tolerance is computationally expensive, while the lower bound requires specialized architectures (smoothed classifiers).

<!-- even then, the certifiable tolerance is often too small to be practical and lower than empirically observed limits. -->

### Soft stability: a more flexible alternative
Instead of requiring that *all* perturbations do not flip the prediction, we propose a probabilistic approach that measures how often the prediction changes:

**Definition. [Soft Stability]**
At radius $r$, an explanation's **stability rate** $\tau_r$ is the probability that adding up to $r$ additional features does not change the prediction.
{: .notice--info}

Soft stability is illustrated in the previous example, where the stability rate $\tau_4 = 95.3$%.
This is a generalization of hard stability, which simply states that the explanation is *not* hard stable at radius $4$ because the stability rate is $< 100$%.
By making this shift to a probabilistic requirement, soft stability offers two key benefits:
1. **Model-agnostic certification**: The soft stability rate is efficiently computable for any classifier, whereas hard stability is only easy to certify for smoothed classifers.
2. **Practical guarantees**: The certificates for soft stability are much larger and more practically useful than those obtained from hard stability.


In the next sections, we provide detailed comparisons between hard and soft stability and demonstrate the practical benefits of our new approach.


## Technical details


<!--
- Here is a more detailed comparison of hard and soft stability
  + Restate definitions if we have to
- Challenges with hard stability certification
  + Have to use smoothed classifiers ("specialized architectures")
- Up-sell the benefits of soft stability certification
  + Model-agnostic (doesn't need smoothing)
  + Less conservative guarantees (as we'll see in the subsequent experiments)
  + Sample-efficient certification, as whown by the following algorithm
-->




We now expand on the nuanced differences between hard stability (previous work) and soft stability (this work).
Both are statements about how the prediction varies when one adds features to an explanation.
We use the term radius to measure perturbation, this is the standard convention in adversarial robustness literature.
However, hard stability tries to ensure that every perturbation up to some threshold does not change, but whereas soft stability measures how often it is maintained, or something.
We illustrate this in the following figure.

At some radius $r$, they answer the following questions:

* Hard stability at radius $r$: does any reveal of up to $r$ features guarantee that the prediction remains unchanged.
* Soft stability at radius $r$: how often does revealing up to $r$ features maintain the prediction?

As alluded to earlier, hard stability is a generalization of soft stability when the stability rate is exactly $100$%.
We illustrate this difference in the following figure.


{% include gallery id="gallery_hard_vs_soft" layout="" caption="**A visual example of certified radii by hard stability vs. soft stability.** \\
For an image of a penguin masked to show only the top 44% explanation by LIME, hard stability certifies that adding one patch won't change the prediction. In contrast, soft stability can certify adding up to 5 patches with a probabilistic guarantee." %}




Certifying hard stability is non-trivial.
This is because if the classifier lacks mathematically convenient prperties, then one must check a computationally intractable nuber of possible perturbasitons to ensure that all perturbations up to some radius do not cause the prediction flip.
The approach for doing this in our previous work is by using specialized architectures, in particular, smoothed classifiers.
We refer to a [later section](#mild-smoothing-improves-soft-stability) for details about how this smoothing procedure works.
The main idea is to take an existing classifier, and then modify it to have convenient proeprties, with which we wmay then quickyl check that perturbations up to blah blah balh.
However, this process is not good because the smoothed classifier is not very good at this accuracy business.

In contrast to the above very computationally stupid way of certifying hard stability, we can in fact certify soft stability --- in particular the stability rate --- through a very standard sampling process.
We describe this algortihm in the following manner.


**Algorithm for estimating the stability rate $\tau_r$.** \\
To estimate the soft stability at radius $r$ to accuracy $\varepsilon$ and confidence $1 - \delta$, it suffices to take $N \geq \frac{\log(2/\delta)}{2 \varepsilon^2}$ samples uniformly from the additive perturbations of size $\leq r$.
The stability rate estimator $\hat{\tau}_r$ will satisfy $\lvert \hat{\tau}_r - \tau_r \rvert \leq \varepsilon$ with probability $\geq 1 - \delta$.
{: .notice--danger}


The resulting estimated soft stability rate $\hat{\tau}_r$ is accurate to the true stability rate $\tau_r$ with high confidence.
In other words, with probability $\geq 1 - \delta$, we have that $\lvert \hat{\tau}_r - \tau_r \rvert \leq \varepsilon$.
The technical details follow by standard theoretical concentrations on sampling. 
We give a demonstration of this estimation process in our [tutorial notebook](https://github.com/helenjin/soft_stability/blob/main/tutorial.ipynb). 


There are three main computational benefits of estimating soft stability in this manner.
First, the estimation algorithm is model-agnostic, which means that soft stability can be certified for any model, not just smoothed ones --- in contrast to hard stability.
Second, this algorithm is sample-efficient: the number of samples depends only on the hyperparameters $\varepsilon$ and $\delta$, meaning that the runtime cost scales linearly with the cost of running the classifier.
Thirdly, as our subsequent experiments will show, soft stability certificates are much less conservative than hard stability, making them more practical for measuring the robustness of explanations.

Our summary on the trade-offs between hard stability and soft stability are summarized in the following table.



| Stability Type | Formal Guarantee | Computational Considerations|
| |:----------------:|:-----------:|
| Hard Stability | **All** perturbations up to $r$ features do not alter the prediction. | Maximum tolerance is intractable to certify, while a lower-bound requires the use of smoothed classifiers |
| Soft stability | Computes **how often** perturbations up to a radius $r$ maintain the prediction | High-confidence estimation is sample-efficient. |


Next, we give some experimental evaluations that compare soft and hard stability in practice.



## Experiments

We next consider how soft stability compares with hard stability in practice. 
We empirically evaluate on vision and language tasks.

Below, we show the stability rates we can attain on [Vision Transformer](https://huggingface.co/google/vit-base-patch16-224), when taking $1000$ examples from ImageNet.


{% include gallery id="gallery_soft_certifies_more" layout="" caption="**Soft stability certifies more than hard stability.** LIME and SHAP showing a sizable advantage over IntGrad, MFABA, and random baselines across all radii." %}

Next, we show the stability rates we can attain on [RoBERTa]() and [TweetEval](https://huggingface.co/datasets/cardiffnlp/tweet_eval).

{% include gallery id="gallery_soft_certifies_more_tweeteval" layout="" caption="**Soft stability certifies more than hard stability.** " %}

We observe that the attainable radii from soft stability are much larger, by up to two orders of magnitude, than those obtained by hard stability.
This trend is most pertinent in the vision task but is also present in the language task.
Furtheremore, for the vision task, we can see that soft stability effectively differentiates various explanation methods, in contrast to hard stability.
Note that a caveat of the soft stability estimation is that it is inherently probabilistic, which directly contrasts with the deterministic style of hard stability.
To boost the estimation confidence, one can take more samples to better approximate the true soft stability rate.


## Mild smoothing improves soft stability
Should we abandon smoothing?
It turns out no, not necessarily.
Although the algorithm for certifying does not require a smoothed classifier, we found that mildly smoothed models often have empirically improved stability rates.

<details>
<summary>Click for details</summary>
<div markdown="1">
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


</div>
</details>


## Conclusion
In this blog post, we explore a practical variant of stability guarantees that improves upon existing methods in the literature.

Powerful machine learning models are increasingly deployed in practice.
However, their opacity presents a major challenge when adopted in high-stake domains, where transparent and reliable explanations are needed in decision-making.
In healthcare, for instance, doctors require insights into the diagnostic steps to trust the model and integrate them into clinical practice effectively.
Similarly, in the legal domain, attorneys must ensure that decisions reached with the assistance of models meet stringent judicial standards.

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


