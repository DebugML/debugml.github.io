---
title: "Logicbreaks: A Framework for Understanding Subversion of Rule-based Inference"
layout: single
excerpt: "We study jailbreak attacks through propositional Horn inference."
header:
  overlay_filter: "0.75"
  overlay_image: /assets/images/logicbreaks/building_bombs.gif
  teaser: /assets/images/logicbreaks/building_bombs.gif

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



> Prompts are commonly used to specify what large language models (LLMs) should and should not do.
> However, LLMs are often tricked into ignoring such instructions --- how are such powerful models so easily fooled?
> We use propositional Horn logic as a foundation to better understand why this is the case.
> In our theoretical investigation, we subvert the rule-following of simplified models by strategically manipulating the attention.
> Interestingly, we find that real LLM jailbreak share similar strategies as our theory-based ones.

{% include gallery id="gallery_building_bombs" caption="An adversarial suffix makes the LLM ignore its safety prompt." %}


## Logical Inference Can Model Rule-following

Developers and users commonly use prompts to specify what LLMs should and should not do.
Although this strategy is generally effective in aligning LLMs to match desired behaviors, it is also exploitable, most notably by *jailbreak attacks*.
Such attacks work by appending the prompt with a malicious sequence of tokens, known as the **adversarial suffix**, that causes the model to generate undesirable outputs.
As LLMs become increasingly widespread, there is also increasing interest in exploiting them.
However, the principles of why such exploits work, as well as how to engineer and prevent them, are poorly understood.


In this work, we study how to subvert LLMs from following the rules specified in the prompt.
To establish a theoretical foundation for such attacks, we characterize rule-following as inference in propositional Horn logic.
In this logic, rules take the form *"if $P$ and $Q$, then $R$"* for some propositions $P$, $Q$, and $R$, and 
This choice of logic is common for modeling rule-based tasks, which includes a large number of jailbreak attacks.
Propositions in such attacks may correspond to various conditions intended to guide output generation, such as *"the user asked about building bombs"* or *"the model should deny the user query"*.


In conjunction with the step-by-step output generation of LLMs, this framework allows us to characterize the different modalities in which inference may fail.
We first consider a simplified, theoretical setting to study how a transformer-based language model performs reasoning.
We then construct attacks against our theoretical setup, and find that these attacks also transfer to learned models of a similar size trained from data.
Finally and interestingly, we find that real jailbreak attacks on LLMs also use strategies similar to our theory-based derivations.
A high-level overview is illustrated in the following.

{% include gallery id="gallery_results_overview" caption="An overview of our results." %}

Our work suggests that investigations on smaller theoretical models and well-defined setups can yield insights into the mechanics of real-world rule-subversions, particularly jailbreak attacks on large language models.
In summary, our **results** are as follows:
* Small transformers can theoretically encode and practically learn inference in propositional Horn logic.
* Attacks designed from theory transfer to practice.
* Existing LLM jailbreaks mirror these theory-based attacks.



## Breaking Inference in Propositional Horn Logic

We model rule-following as inference in propositional Horn logic, a system in which inference rules have the form *"if $P$ and $Q$, then $R$"* for some propositions $P$, $Q$, and $R$.
Although our starting motivation is to understand jailbreak attacks, it will be helpful to more broadly consider dependency relations, which propositional Horn logic is well-suited to modeling.
As a running example, we consider the task of crafting items in Minecraft according to a recipe list.
For instance, a player may be given the following recipe and starting items:


{% include gallery id="gallery_mc_example" caption="Crafting items in Minecraft." %}


Encountered with such a task, the player may formulate the following prompt for an LLM:

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


In turn, we may abstract this prompt into the following set of inference rules $\Gamma$ and known facts $\Phi$:

$$
  \Gamma = \{A \to B, B \to C, D \to E, C \land E \to F\}, \;
  \Phi = \{A,D\}
$$

For example, the rule $C \land E \to F$ read *"If I have Wool and I have Stick, then I can create Fishing Rod"*, and the proposition $B$ stands for "I have Wool", which we treat as equivalent to "I can create Wool".
The inference task is to find all the derivable propositions.
A well-known algorithm for this is [forward chaining](https://en.wikipedia.org/wiki/Forward_chaining), which repeatedly applies the inference rules to the known facts until no new knowledge is derivable.
For the above example, this sequence of derivations is as follows:

$$
\begin{aligned}
  \{A,D\}
    &\xrightarrow{\mathsf{Apply}[\Gamma]} \{A,B,D,E\} \\
    &\xrightarrow{\mathsf{Apply}[\Gamma]} \{A,B,C,D,E\} \\
    &\xrightarrow{\mathsf{Apply}[\Gamma]} \{A,B,C,D,E,F\},
\end{aligned}
$$

where $\mathsf{Apply}[\Gamma]$ is a set-to-set function that implements a one-step application of $\Gamma$.
Because no knowledge may be derived starting from the *proof state* $\\{A,B,C,D,E,F\\}$, we may stop.
The iterative nature of forward chaining is particularly amenable to LLMs, which commonly use techniques like chain-of-thought to autoregressively generate their output.


So what does it mean to subvert rule-following in propositional Horn logic?
Intuitively, we take this to mean deviating from the derivation sequence of $\mathsf{Apply}[\Gamma]$.
In particular, there are three different ways in which such a deviation may occur.
Given a reasoner model $\mathcal{R}$ that is to mimic $\mathsf{Apply}[\Gamma]$, we categorize the three failure modes of inference as follows:


**(1) Rule suppression**: a rule and its dependents are ignored.
In the following, $E$ is absent from the second state, meaning that the rule $D \to E$ failed to trigger:

$$
  \{A,D\}
  \xrightarrow{\mathcal{R}} \{A,B,D\}
  \xrightarrow{\mathcal{R}} \{A,B,C,D\}
  \xrightarrow{\mathcal{R}} \cdots
$$



**(2) Fact amnesia**: the model forgets its previously derived information.
In the following $A$ is absent from the second state:

$$
  \{A,D\}
  \xrightarrow{\mathcal{R}} \{B,D,E\}
  \xrightarrow{\mathcal{R}} \{B,C,D,E\}
  \xrightarrow{\mathcal{R}} \cdots
$$


**(3) State coercion**: the model infers something absurd.
In the following, each inference step arrives at a conclusion that is absurd:

$$
  \{A,D\}
  \xrightarrow{\mathcal{R}} \{F\}
  \xrightarrow{\mathcal{R}} \{B,C,E\}
  \xrightarrow{\mathcal{R}} \cdots
$$



Of these three failure modalities, we focus on **rule suppression**, which is most closely related to jailbreak attacks, and refer to our paper for additional discussion on fact amnesia and state coercion.


## Subverting Inference in Transformers (Theory)

To better understand how these attacks might function in large language models, we first study a simplified theoretical setting.
We show that small transformers have the capacity to encode and learn propositional Horn inference, and that attacks designed against theory models transfer to models that learn to reason under the constraints imposed by the theory.


### Small Transformers Can Encode and Learn Inference

We first show that even small transformers have both the capacity to encode and learn propositional inference.
Our core idea is to use transformers to approximate the Boolean circuits traditionally used in inference tasks, such as those of propositional Horn inference.


{% include gallery id="gallery_main_idea" caption="The main theoretical idea." %}

Particularly, for a reasoning problem with at most $n$ propositions, we use an embedding dimension of $d = 2n$.
Following our running Minecraft example with $n = 6$ proposition, we embed the rule $C \land E \to F$ as follows:

$$
  (C \land E \to F)
  \; \cong \;
  (001010,000001)
  \in \{0,1\}^{12}
$$

Intuitively, the first $n = 6$ bits encode the antecedent of the rule, while the remaining $n$ bits encode the consequent.
Similarly, we also embed the proof states in $\\{0,1\\}^{12}$ as follows:

$$
\begin{aligned}
  \{A,D\} &\;\cong\; (000000,100100) \in \{0,1\}^{12} \\
  % \{A,B,D,E\} &\;\cong\; (000000,110110) \\
  % \{A,B,C,D,E\} &\;\cong\; (000000,111110) \\
  % \{A,B,C,D,E,F\} &\;\cong\; (000000,111111)
\end{aligned}
$$

In particular, our encoding is formatted as follows:

$$
  X_0 = 
  \mathsf{Encode}(\Gamma, \Phi)
  = \begin{bmatrix}
    100000 & 010000 \\
    010000 & 001000 \\
    000100 & 000010 \\
    001010 & 000001 \\
    000000 & 100100
  \end{bmatrix}
  \in \{0,1\}^{5 \times 12}
$$

where a reasoner model $\mathcal{R}$ starts from $X_0$ and autoregressively predicts the $n$ bits of the next state to generate $X_1 \in \\{0,1\\}^{6 \times d}$, $X_2 \in \\{0,1\\}^{7 \times d}$, and $X_3 \in \\{0,1\\}^{8 \times d}$ such that:

$$
\begin{aligned}
  \mathsf{Decode}(X_0) &= \{A,D\} \\
  \mathsf{Decode}(X_1) &= \{A,B,D,E\} \\
  \mathsf{Decode}(X_2) &= \{A,B,C,D,E\} \\
  \mathsf{Decode}(X_3) &= \{A,B,C,D,E,F\}
\end{aligned}
$$

In particular, a small reasoner model suffices to encode each step of this autoregressive inference procedure.


**Theorem.** (Encoding, Informal)
For binarized prompts, a transformer with one layer, one self-attention head, and embedding dimension $d = 2n$ can encode one step of inference, where $n$ is the number of propositions.
{: .notice--success}


A caveat is that this is a result about existence: it shows that transformers of a certain size have the sufficient capacity to encode one step of inference.
Our paper gives additional details of the proof, in which we explicitly specify all the parameter values of a transformer such that it is able to implement the inference circuit.
Our transformer encoding is not the only one in the [literature](https://arxiv.org/abs/2205.11502), but it is the smallest to our knowledge, which is generally an advantage for theoretical analysis.


So can transformer of such sizes *actually learn*?
Our experiments show that models at our theoretically-restricted sizes can still learn propositional inference to a high accuracy.

{% include gallery id="gallery_learned_accs" caption="Small transformers can learn propositional inference to high accuracy. Left, center, and right are the accuracies for $t = 1, 2, 3$ steps of inference, respectively. A model must correctly predict the state of all $n$ propositions up to $t$ steps to be counted as correct." %}


We observe that models of size $d \geq 2n$ can consistently learn propositional inference to high accuracy, whereas those at $d < 2n$ begin to struggle.
Although a one-layer and one-head transformer is fairly small, these empirical findings suggest that our theoretical construction is not a completely unrealistic setup on which to study attacks on rule-following.
We remark that it is an open problem to better understand the training dynamics and verify whether these learned models succeed in achieving a *correct* solution.


### Theory-based Attacks Transfer to Learned Models

We next show that our theory-based models can be attacked, and that such attacks transfer to learned models.
As stated before, a key advantage of our construction is its size, which lets us manually construct attacks against all three modalities of inference failure --- rule suppression, fact amnesia, and state coercion --- that are guaranteed to succeed.
As an example, suppose that we would like to suppress the activation of some target rule $(\alpha_{\mathsf{tgt}}, \beta_{\mathsf{tgt}}) \in \Gamma$.
Given an attack budget $p > 0$, the objective is to find an adversarial suffix $\Delta \in \mathbb{R}^{p \times d}$ that, when appended to $X_0 = \mathsf{Encode}(\Gamma, \Phi)$, causes the reasoner $\mathcal{R}$ to place low attention on the targeted rule.
We may roughly formulate this as the following problem:

$$
\begin{aligned}
  \underset{\Delta \in \mathbb{R}^{p \times d}}{\text{minimize}}
    &\quad \text{Attention that $\mathcal{R}$ places on $(\alpha_{\mathsf{tgt}}, \beta_{\mathsf{tgt}})$} \\
  \text{where}
    &\quad \text{$\mathcal{R}$ is evaluated on $\mathsf{append}(X_0, \Delta)$} \\
    &\quad X_0 = \mathsf{Encode}(\Gamma, \Phi)
\end{aligned}
$$

In fact, it is possible to explicitly find such a suffix $\Delta$ for all three modalities of attacks.

**Theorem.** (Theory-based Attacks, Informal)
For the model described in the encoding theorem, there exist suffixes that induce fact amnesia, rule suppression, and state coercion.
{: .notice--success}



We have so far designed these attacks against a *theoretical construction* in which we manually assigned values to every network parameter.
But how do such attacks transfer to *learned models*, i.e., models with the same size as specified in the theory, but trained from data?
Interestingly, the learned reasoners are also susceptible to theory-based rule suppression and fact amnesia attacks.


{% include gallery id="gallery_theory_attacks" caption="With some modifications, the theory-based rule suppression and fact amnesia attacks achieve a high attack success rate. The state coercion does not succeed even with our modifications, but attains a 'converging' behavior as evidenced by the diminishing variance. The number of 'repetitions' is a measure of how 'strong' the attack is." %}



## Real Jailbreaks Mirror Theory-based Ones
We have previously considered how theoretical jailbreaks might work against simplified models that take a binarized representation of the prompt.
It turns out that such attacks transfer to real jailbreak attacks as well.
For this task, we fine-tuned GPT-2 models on a set of Minecraft recipes curated from [GitHub](https://github.com/joshhales1/Minecraft-Crafting-Web/).
A sample input is as follows:


**Prompt:**
Here are some crafting recipes: If I have Sheep, then I can create Wool. If I have Wool,
then I can create String. If I have Log, then I can create Stick. If I have String and Stick,
then I can create Fishing Rod. If I have Brick, then I can create Stone Stairs. Here are
some items I have: I have Sheep and Log. Based on these items and recipes, I can create
the following:
{: .notice--info}

The attacker's objective is to find an adversarial suffix that induces some inference error.
For instance, the following would be a valid rule suppression behavior of *"If I have Wool, then I can create String"*:


**LLM(Prompt + XXX):**
I have Sheep, and so I can create Wool.
I have Log, and so I can create Stick.
I cannot create any other items.
{: .notice--info}


Observe that all dependents of Wool are absent from the output, as the next applicable rule is never triggered.
We use the [Greedy Coordinate Gradients](https://arxiv.org/abs/2307.15043) (GCG) algorithm to find such suffixes for language models.
Notably, the suffixes found by GCG also use an attention-suppression mechanism as explored in our theory.
For instance, below is an example of the *difference* in attention between an attacked (with suffix) and a non-attacked (without suffix) output generation.


{% include gallery id="gallery_mc_suppression" caption="The difference in attention weights between a generation with and without the adversarial suffix. When the suffix is present, the tokens of the targeted rule receive lower attention than when the suffix is absent." %}

We give additional details in our paper on how the attention is aggregated.
However, the general trend is that GCG finds suffixes that suppress the attention placed on the targeted rule.


<!--
In summary, our results here are as follows
* POINT 2
* POINT 3
-->


## Conclusion
We use propositional Horn logic as a framework to study how to subvert the rule-following of language models.
We find that attacks derived in a simple, theoretical setting are mirrored in real jailbreaks against LLMs.