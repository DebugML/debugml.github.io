---
title: "Logicbreaks: A Framework for Understanding Subversion of Rule-based Inference"
layout: single
excerpt: "We study jailbreak attacks through propositional Horn inference."
header:
  overlay_filter: "0.75"
  overlay_image: /assets/images/logicbreaks/building_bombs.gif
  teaser: /assets/images/logicbreaks/building_bombs.gif
  actions:
    - label: "Paper"
      url: https://arxiv.org/abs/2407.00075
    - label: "Code"
      url: https://github.com/AntonXue/tf_logic

authors:
  - Anton Xue
  - Avishree Khare
  - Rajeev Alur
  - Surbhi Goel
  - Eric Wong


gallery_building_bombs:
  - image_path: /assets/images/logicbreaks/building_bombs.gif
    title: Jailbreak attack example

gallery_results_overview:
  - image_path: /assets/images/logicbreaks/blog_results_overview.png
    title: General overview of our results.

gallery_mc_example:
  - image_path: /assets/images/logicbreaks/blog_mc_example.png
    title: Crafting recipes in Minecraft.

gallery_main_idea:
  - image_path: /assets/images/logicbreaks/blog_main_idea.png
    title: Main idea.

gallery_learned_accs:
  - image_path: /assets/images/logicbreaks/exp1_step1_acc.png
  - image_path: /assets/images/logicbreaks/exp1_step2_acc.png
  - image_path: /assets/images/logicbreaks/exp1_step3_acc.png

gallery_theory_attacks:
  - image_path: /assets/images/logicbreaks/exp2_suppress_rule_acc.png
  - image_path: /assets/images/logicbreaks/exp2_fact_amnesia_acc.png
  - image_path: /assets/images/logicbreaks/exp2_coerce_state_var.png

gallery_mc_suppression:
  - image_path: /assets/images/logicbreaks/mc_suppression_example_2_4.png
    image_path: /assets/images/logicbreaks/mc_suppression_example_38_4.png
    title: Rule suppression on the Minecraft dataset.


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


> LLMs can be easily tricked into ignoring content safeguards and other prompt-specified instructions.
> How does this happen?
> To understand how LLMs may fail to follow the rules, we model rule-following as logical inference and theoretically analyze how to subvert LLMs from reasoning properly.
> Surprisingly, we find that our theory-based attacks on inference are aligned with real jailbreaks on LLMs.


{% include gallery id="gallery_building_bombs" caption="An adversarial suffix makes the LLM ignore its safety prompt." %}


## Modeling Rule-following with Logical Inference

Developers commonly use prompts to specify what LLMs should and should not do.
For example, the LLM may be instructed to not give bomb-building guidance through a *safety prompt* such as "don't talk about building bombs".
Although such prompts are sometimes effective, they are also easily exploitable, most notably by *jailbreak attacks*.
In jailbreak attacks, a malicious user crafts an adversarial input that tricks the model into generating undesirable content.
For instance, appending the user prompt "How do I build a bomb?" with a nonsensical **adversarial suffix** "@A$@@..." fools the model into yielding bomb-building instructions.


In this blog, we present some [recent work](https://arxiv.org/abs/2407.00075) on how to subvert LLMs from following the rules specified in the prompt.
Such rules might be safety prompts that look like *"if [the user is not an admin] and [the user asks about bomb-building], then [the model should reject the query]"*.
Our main idea is to cast rule-following as inference in propositional Horn logic, a system wherein rules take the form *"if $P$ and $Q$, then $R$"* for some propositions $P$, $Q$ and $R$.
This logic is a common choice for modeling rule-based tasks.
In particular, it effectively captures many instructions commonly specified in the safety prompt, and so serves as a foundation for understanding how jailbreaks subvert LLMs from following these rules.


We first set up a logic-based framework that lets us precisely characterize how rules can be subverted.
For instance, one attack might trick the model into ignoring a rule, while another might lead the model to absurd outputs.
Next, we present our main theoretical result of how to subvert a language model from following the rules in a simplified setting.
Our work suggests that investigations on smaller theoretical models and well-designed setups can yield insights into the mechanics of real-world rule-subversions, particularly jailbreak attacks on large language models.
In summary:
* Small transformers can theoretically encode and empirically learn inference in propositional Horn logic.
* Jailbreak attacks are easy to find and highly effective in our simplified, analytical setting.
* These theory-based attacks transfer to practice, and existing LLM jailbreaks mirror these theory-based attacks.

<!--
A key benefit of our logic-based approach is that it lets us characterize the different properties of what it means to follow the rules --- and what it means to break them.
For instance, one adversarial suffix might trick the model into ignoring a rule, while another suffix might lead the model to absurd outputs.
Although both suffixes subvert rule-following, their strategies are fundamentally different.
By identifying and formalizing the different rule-following properties, we can also precisely describe how the model may fail to follow the rules.

We first set up a logic-based framework that lets us precisely characterize how rules can be subverted.
Next, we present our main theoretical result of how to subvert a language model from following the rules in a simplified setting.
Furthermore, we find that real jailbreak attacks on LLMs also use strategies similar to our theory-based derivations.
Our work suggests that investigations on smaller theoretical models and well-designed setups can yield insights into the mechanics of real-world rule-subversions, particularly jailbreak attacks on large language models.
In summary:
* Small transformers can theoretically encode and empirically learn inference in propositional Horn logic.
* Jailbreak attacks are easy to find and highly effective in our simplified, analytical setting.
* These theory-based attacks transfer to practice, and existing LLM jailbreaks mirror these theory-based attacks.
-->

<!--
Our main approach is as follows.
We first formalize a logic-based framework for studying rule-following.
We then consider a simplified, theoretical setting to show that transformer-based language models can both encode and learn rule-following.
Next, we construct attacks against our analytical setup, and show that these attacks also transfer to reasoners trained on data.
Finally, we find that real jailbreak attacks on LLMs also use strategies similar to our theory-based derivations.
Our work suggests that investigations on smaller theoretical models and well-designed setups can yield insights into the mechanics of real-world rule-subversions, particularly jailbreak attacks on large language models.
In summary:
* Small transformers can theoretically encode and practically learn inference in propositional Horn logic.
* Theory-based attacks transfer to practice.
* Existing LLM jailbreaks mirror these theory-based attacks.
-->


{% include gallery id="gallery_results_overview" caption="An overview of our results. We devise jailbreak attacks in a simplified theoretical setting that transfer to learned reasoners. Moreover, real jailbreaks on real LLMs exhibit similar strategies as our theory-based setup." %}


## A Logic-based Framework for Rule-following

To study rule-following, we model it as inference in propositional Horn logic.
Moreover, although our starting motivation is to better understand jailbreak attacks, it will be helpful to more broadly consider *dependency relations*, which is especially well-suited to propositional Horn logic.
As a running example, consider the task of crafting items in [Minecraft](https://www.minecraft.net).
For instance, a player may have the following crafting recipes and starting items:


{% include gallery id="gallery_mc_example" caption="Crafting items in Minecraft. Given a recipe list and some starting items, what items can the player make?" %}


Encountered with this task, the player might formulate the following prompt:

**Prompt:**
Here are some crafting recipes:
If I have Sheep, then I can create Wool.
If I have Wool, then I can create String.
If I have Log, then I can create Stick.
If I have String and Stick, then I can create Fishing Rod.
Here are some items I have:
I have Sheep and Log as starting items.
Based on these items and recipes, what items can I create?
{: .notice--info}


To which the LLM might respond:



**Generation on Prompt:**
I have Sheep, and so I can create Wool.
I have Wool, and so I can create String.
I have Log, and so I can create Stick.
I have String and Stick, and so I can create Fishing Rod.
I cannot create any other items.
{: .notice--info}


How can we be sure that the LLM has responded correctly?
One way is to check whether its output matches what a logical reasoning algorithm might say.

<!--
But how can we be sure that the LLM has responded correctly?
One idea is to relate the LLM output to well-established logic algorithms.
Then, an LLM output is "correct" if it "sufficiently matches" such a reference algorithm.
Because these reference algorithms have nice mathematical properties, an LLM output that "matches" such an algorithm will also inherit the corresponding properties.
-->

### Rule-following via Forward Chaining

As a reference algorithm, we use [forward chaining](https://en.wikipedia.org/wiki/Forward_chaining), which is a well-known algorithm for inference in propositional Horn logic.
Given the task, the main idea is to first extract a set of rules $\Gamma$ and known facts $\Phi$ as follows:

$$
  \Gamma = \{A \to B, B \to C, D \to E, C \land E \to F\}, \;
  \Phi = \{A,D\}
$$

We have introduced propositions $A, B, \ldots, F$ to stand for the obtainable items.
For example, the proposition $B$ stands for "I have Wool", which we treat as equivalent to "I can create Wool", and the rule $C \land E \to F$ reads "If I have Wool and Stick, then I can create Fishing Rod".
The inference task is to find all the derivable propositions, i.e., that we can create Wool, Stick, and String, etc.
Forward chaining then iteratively applies the rules $\Gamma$ to the known facts $\Phi$ as follows:

$$
\begin{aligned}
  \{A,D\}
    &\xrightarrow{\mathsf{Apply}[\Gamma]} \{A,B,D,E\} \\
    &\xrightarrow{\mathsf{Apply}[\Gamma]} \{A,B,C,D,E\} \\
    &\xrightarrow{\mathsf{Apply}[\Gamma]} \{A,B,C,D,E,F\}.
\end{aligned}
$$

The core component of forward chaining is $\mathsf{Apply}[\Gamma]$, which performs a one-step application of all the rules in $\Gamma$.
The algorithm terminates when it reaches a *proof state* like $\\{A,B,C,D,E,F\\}$ from which no new facts can be derived.
The iterative nature of forward chaining is particularly amenable to LLMs, which commonly use techniques like chain-of-thought to generate their output step-by-step.


### Subversions on Rule-following

<!--
However, a major difference between LLM execution and forward chaining is that an LLM generates its output step-by-step, whereas forward chaining keeps track of all the derivable facts at each step.
-->

So what does it mean for an LLM to *not* follow the rules?
Following our earlier idea, we say that an LLM fails to follow the rules if its output does not "match" that of forward chaining.
**Crucially, we identify three ways in which the outputs may fail to match.**
First, recall that the original, unattacked generation looks as follows:


**Original Generation on Prompt:**
I have Sheep, and so I can create Wool.
I have Wool, and so I can create String.
I have Log, and so I can create Stick.
I have String and Stick, and so I can create Fishing Rod.
I cannot create any other items.
{: .notice--info}



An adversarial suffix can then specifically target these erroneous behaviors, described below.


**(1) Rule suppression**: a rule and its dependents are ignored.
Suppose we want to ignore the rule "if I have Wool, then I can create String".
In this case, we would like for a prompt appended with the adversarial suffix to generate the following, where we use ~~strikeout text~~ to denote the omitted output:

**Generation on Prompt + Adv. Suffix**:
I have Sheep, and so I can create Wool.
~~I have Wool, and so I can create String.~~
I have Log, and so I can create Stick.
~~I have String and Stick, and so I can create Fishing Rod.~~
I cannot create any other items.
{: .notice--danger}


**(2) Fact amnesia**: the model skips an intermediate step in its derivation.
For instance, if we want to make the model skip the derivation of Wool, an adversarial suffix-appended prompt should result in the following, where ~~strikeout text~~ similarly denotes omission.

**Generation on Prompt + Adv. Suffix:**
I have Sheep, ~~and so I can create Wool.
I have Wool,~~ and so I can create String.
I have Log, and so I can create Stick.
I have String and Stick, and so I can create Fishing Rod.
I cannot create any other items.
{: .notice--danger}



**(3) State coercion**: the model infers something absurd.
That is, we'd like to have the suffix-appended prompt generate anything we'd like to, no matter how ridiculous.
For instance:

**Generation on Prompt + Adv. Suffix:**
I have String, and so I can create Gray Dye.
I cannot create any other items.
{: .notice--danger}


## Subverting Inference in Transformers (Theory)

To better understand how adversarial suffixes affect LLMs, we first study how such models might reason in a simplified theoretical setting.
By studying rule-following in a simpler setting, we can more easily construct attacks that induce each of the three failure modes.
Interestingly, these theory-based attacks also transfer to models learned from data.


Our main findings are as follows.
First, we show that a transformer with only **one layer** and **one self-attention head** has the *theoretical capacity* to encode one step of inference in propositional Horn logic.
Moreover, we find that our simple theoretical construction is susceptible to attacks that target all three failure modes of inference.


<details>
<summary>Click here for details</summary>
<div markdown="1">

Our main encoding idea is as follows:
* Propositional Horn logic is Boolean-valued, so inference can be implemented via a Boolean circuit.
* A one-layer transformer has the theoretical capacity to approximate this circuit; more layers means more power.
* Therefore, a (transformer-based) language model can also perform propositional inference assuming that its weights behave like the "correct" Boolean circuit.
We illustrate this in the following.

{% include gallery id="gallery_main_idea" caption="The main theoretical encoding idea. A propositional Horn query may be equivalently formulated as Boolean vectors, which may then be solved via Boolean circuits. A language model has the theoretical capacity to encode/approximate such an idealized circuit." %}


More concretely, our encoding result is as follows.

**Theorem.** (Encoding, Informal)
For binarized prompts, a transformer with one layer, one self-attention head, and embedding dimension $d = 2n$ can encode one step of inference, where $n$ is the number of propositions.
{: .notice--success}


We emphasize that this is a result about **theoretical capacity**: it states that transformers of a certain size have the ability to perform one step of inference.
However, it is not clear how to certify whether such transformers are guaranteed to learn the "correct" set of weights.
Nevertheless, such results are useful because they allow us to better understand what a model is theoretically capable of.
Our theoretical construction is not the [only one](https://arxiv.org/abs/2205.11502), but it is the smallest to our knowledge.
A small size is generally an advantage for theoretical analysis and, in our case, allows us to more easily derive attacks against our theoretical construction.


Although we don't know how to provably guarantee that a transformer learns the correct weights, we can empirically evaluate the performance of learned models.
By fixing an architecture of one layer and one self-attention head while varying the number of propositions and embedding dimensions, we see that models subject to our theoretical constraints **can** learn inference to a high accuracy.

{% include gallery id="gallery_learned_accs" caption="Small transformers can learn propositional inference to high accuracy. Left, center, and right are the accuracies for $t = 1, 2, 3$ steps of inference, respectively. A model must correctly predict the state of all $n$ propositions up to $t$ steps to be counted as correct." %}

In particular, we observe that models of size $d \geq 2n$ can consistently learn propositional inference to high accuracy, whereas those at $d < 2n$ begin to struggle.
These experiments provide evidence that our theoretical setup of $d = 2n$ is not a completely unrealistic setup on which to study rule-following.
It is an open problem to better understand the training dynamics and to verify whether these models provably succeed in achieving the "correct" weights.


### Theory-based Attacks Manipulate the Attention

Our simple analytical setting allows us to derive attacks that can provably induce rule suppression, fact amnesia, and state coercion.
As an example, suppose that we would like to suppress some rule $\gamma$ in the (embedded) prompt $X$.
Our main strategy is to find an adversarial suffix $\Delta$ that, when appended to $X$, draws attention away from $\gamma$.
In other words, this rule-suppression suffix $\Delta$ acts as a "distraction" that makes the model forget that the rule $\gamma$ is even present.
This may be (roughly) formulated as follows:

$$
\begin{aligned}
  \underset{\Delta}{\text{minimize}}
    &\quad \text{The attention that $\mathcal{R}$ places on $\gamma$} \\
  \text{where}
    &\quad \text{$\mathcal{R}$ is evaluated on $\mathsf{append}(X, \Delta)$} \\
\end{aligned}
$$

As a technicality, we must also ensures that $\Delta$ draws attention away from only the targeted $\gamma$ and leaves the other rules unaffected.
In fact, for reach of the three failure modalities, it is possible to find such an adversarial suffix $\Delta$.

**Theorem.** (Theory-based Attacks, Informal)
For the model described in the encoding theorem, there exist suffixes that induce fact amnesia, rule suppression, and state coercion.
{: .notice--success}




We have so far designed these attacks against a *theoretical construction* in which we manually assigned values to every network parameter.
But how do such attacks transfer to *learned models*, i.e., models with the same size as specified in the theory, but trained from data?
Interestingly, the learned reasoners are also susceptible to theory-based rule suppression and fact amnesia attacks.


{% include gallery id="gallery_theory_attacks" caption="With some modifications, the theory-based rule suppression and fact amnesia attacks achieve a high attack success rate. The state coercion does not succeed even with our modifications, but attains a 'converging' behavior as evidenced by the diminishing variance. The 'Number of Repeats' is a measure of how 'strong' the attack is. Interestingly making the attack 'stronger' has diminishing returns against learned models." %}

</div>
</details>


## Real Jailbreaks Mirror Theory-based Ones
We have previously considered how theoretical jailbreaks might work against simplified models that take a binarized representation of the prompt.
It turns out that such attacks transfer to real jailbreak attacks as well.
For this task, we fine-tuned GPT-2 models on a set of Minecraft recipes curated from [GitHub](https://github.com/joshhales1/Minecraft-Crafting-Web/) --- which are similar to the running example above.
A sample input is as follows:


**Prompt:**
Here are some crafting recipes:
If I have Sheep, then I can create Wool.
If I have Wool, then I can create String.
If I have Log, then I can create Stick.
If I have String and Stick, then I can create Fishing Rod.
If I have Brick, then I can create Stone Stairs.
If I have Lapis Block, then I can create Lapis Lazuli.
Here are some items I have: I have Sheep and Log and Lapis Block.
Based on these items and recipes, I can create
the following:
{: .notice--info}


For attacks, we adapted the reference implementation of the [Greedy Coordinate Gradients](https://github.com/llm-attacks/llm-attacks) (GCG) algorithm to find adversarial suffixes.
Although GCG was not specifically designed for our setup, we found the necessary modifications straightforward.
Notably, the suffixes that GCG finds use similar strategies as ones explored in our theory.
As an example, the GCG-found suffix for rule suppression significantly reduces the attention placed on the targeted rule.
We show some examples below, where we plot the **difference** in attention between an attacked (with adv. suffix) and a non-attacked (without suffix) case.
Click the arrow keys to navigate!

<!--
{% include gallery id="gallery_mc_suppression" caption="The difference in attention weights between a generation with and without the adversarial suffix. When the suffix is present, the tokens of the targeted rule receive lower attention than when the suffix is absent." %}
-->


<div class="carousel-container">
  <div class="carousel">
    <div class="carousel-item active">
      <img src="/assets/images/logicbreaks/mc_suppression_example_2_4.png" alt="First slide">
    </div>
    <div class="carousel-item">
      <img src="/assets/images/logicbreaks/mc_suppression_example_38_4.png" alt="Second slide">
    </div>
    <div class="carousel-item">
      <img src="/assets/images/logicbreaks/mc_suppression_example_53_4.png" alt="Second slide">
    </div>
  </div>
  <a class="carousel-control prev" onclick="moveSlide(-1)">&#10094;</a>
  <a class="carousel-control next" onclick="moveSlide(1)">&#10095;</a>
</div>

<style>
.carousel-container {
  position: relative;
  max-width: 100%;
  margin: auto;
  overflow: hidden;
}

.carousel {
  display: flex;
  transition: transform 0.5s ease-in-out;
}

.carousel-item {
  min-width: 100%;
  box-sizing: border-box;
}

.carousel-control {
  position: absolute;
  top: 10%;
  transform: translateY(-50%);
  font-size: 1em;
  color: gray;
  text-decoration: none;
  padding: 0 0px;
  cursor: pointer;
}

.carousel-control.prev {
  left: 0px;
}

.carousel-control.next {
  right: 0px;
}
</style>

<script>
let currentSlide = 0;

function moveSlide(step) {
  const carousel = document.querySelector('.carousel');
  const items = document.querySelectorAll('.carousel-item');
  currentSlide = (currentSlide + step + items.length) % items.length;
  carousel.style.transform = 'translateX(' + (-currentSlide * 100) + '%)';
}
</script>


Although the above are only a few examples, we found a general trend in that GCG-found suffixes for rule suppression do, on average, significantly diminish attention on the targeted rule.
Similarities for real jailbreaks and theory-based setups also exist for our two other failure modes: for both fact amnesia and state coercion, GCG-found suffixes frequently contain theory-predicted tokens.
We report additional experiments and discussion in our paper, where our findings suggest a connection between real jailbreaks and our theory-based attacks.


Our paper also contains additional experiments with the larger Llama-2 model, where similar behaviors are observed, especially for rule suppression.



## Conclusion
We use propositional Horn logic as a framework to study how to subvert the rule-following of language models.
We find that attacks derived from our theory are mirrored in real jailbreaks against LLMs.
Our work suggests that analyzing simplified, theoretical setups can be useful for understanding LLMs.