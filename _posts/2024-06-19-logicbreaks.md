---
title: "Logicbreaks: A Framework for Understanding Subversion of Rule-based Inference"
layout: single
excerpt: "We study jailbreak attacks through propositional Horn inference."

authors:
  - Anton Xue
  - Avishree Khare
  - Rajeev Alur
  - Surbhi Goel
  - Eric Wong


gallery_building_bombs:
  - image_path: /assets/images/logicbreaks/building_bombs.gif
    title: Jailbreak attack example

gallery_exp1:
  - image_path: /assets/images/logicbreaks/exp1_step1_acc.png
  - image_path: /assets/images/logicbreaks/exp1_step2_acc.png
  - image_path: /assets/images/logicbreaks/exp1_step3_acc.png


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
> Using propositional Horn logic as a theoretical foundation, we seek to better understand why this is the case.
> We find that theory-based insights about the attention weights are evidenced in real jailbreak attacks.


{% include gallery id="gallery_building_bombs" caption="An adversarial suffix makes the LLM ignore its safety prompt." %}

## Logical Inference Can Model Rule-following

Developers and users commonly use prompts to specify what LLMs should and should not do.
Although this is generally effective in aligning LLMs to match desired behavior, this strategy is also exploitable, most notably by *jailbreak attacks*.
Such attacks work by appending the prompt with a malicious sequence of tokens, known as the **adversarial suffix**, that causes the model to generate undesirable outputs.
As LLMs become increasingly widespread, there is also increasing interest in exploiting them.
However, the principles of why such exploits work, as well as how to engineer and prevent them, are poorly understood.

To establish a theoretical foundation for such attacks, we study how to subvert LLMs from following the rules specified in the prompt.
We model rule-following as inference in propositional Horn logic, wherein rules take the form *"if $P$ and $Q$, then $R$"* for some propositions $P$, $Q$, and $R$.
This choice of logic is common for modeling rule-based tasks, which includes a large number of jailbreak attacks.
Propositions in such attacks may correspond to various conditions intended to guide output generation, such as *"the user asked about building bombs"* or *"the model should deny the user query"*.


We use a logic-centric approach to study the subversion of prompt-specified rules.
Our **results** are summarized as follows:
* Small transformers can theoretically encode and practically learn inference in propositional Horn logic.
* Attacks designed from theory transfer to practice.
* Existing LLM jailbreaks mirror these theory-based attacks.

Our work strongly suggests that investigations on smaller theoretical models and well-defined setups can yield insights into how rule-subversions, particularly jailbreak attacks, work on large language models.



## Breaking Inference in Propositional Horn Logic

We model rule-following as inference in propositional Horn logic, a system in which inference rules ahve the form *"if $P$ and $Q$, then $R$"* for some propositions $P$, $Q$, and $R$.
But what does it mean to subvert rule-following in this setting?
Although our starting motivation was jailbreak attacks, it will be helpful to consider dependency relations, which propositional Horn logic is particularly well-suited to modeling.

As a running example, we consider the task of crafting in Minecraft.
Given a set of starting items, a common activity is to construct new items according to a recipe list.


<!-- image of minecraft example -->


Given such a task, a user may formulate the following prompt:

> Here are some crafting recipes:
> If I have Sheep, then I can create Wool.
> If I have Wool, then I can create String.
> If I have Log, then I can create Stick.
> If I have String and Stick, then I can create Fishing Rod.
> Here are some items I have:
> I have Sheep and Log as starting items.
> Based on these items and recipes, what items can I create?


$$
\begin{aligned}
  \Gamma &= \{A \to B, A \to C, D \to E, C \land E \to F\}, \\
  \Phi &= \{A,D\}
\end{aligned}
$$


A well-known algorithm for inference is known as *forward chaining*, which repeatedly applies the inference rules to the known facts until no new knowledge may be derived.
For the example above, such a step of derivations looks as follows:


$$
\begin{aligned}
  \{A,D\}
    &\xrightarrow{\mathsf{Apply}[\Gamma]} \{A,B,D,E\} \\
    &\xrightarrow{\mathsf{Apply}[\Gamma]} \{A,B,C,D,E\} \\
    &\xrightarrow{\mathsf{Apply}[\Gamma]} \{A,B,C,D,E,F\}
\end{aligned}
$$



Using the inference properties of propositional Horn logic, we classify three different ways in which an LLM might not properly follow the rules.


**(1) Fact amnesia**: the model skips a reasoning step.
For example:

$$
  \{A,D\}
  \xrightarrow{\mathcal{R}} \{B,D,E\}
  \xrightarrow{\mathcal{R}} \{B,C,D,E\}
  \xrightarrow{\mathcal{R}} \cdots
$$

Observe that $A$ is absent from the second state.

**(2) Rule suppression**: a rule and its dependents are ignored.

$$
  \{A,D\}
  \xrightarrow{\mathcal{R}} \{A,B,D\}
  \xrightarrow{\mathcal{R}} \{A,B,C,D\}
  \xrightarrow{\mathcal{R}} \cdots
$$

Observe that $E$ is absent from the second state.


**(3) State coercion**: the model infers something absurd.

$$
  \{A,D\}
  \xrightarrow{\mathcal{R}} \{F\}
  \xrightarrow{\mathcal{R}} \{B,C,E\}
  \xrightarrow{\mathcal{R}} \cdots
$$


<!-- image of the 2x2 grid -->




<!--
{% include gallery id="gallery_chatgpt_error" caption="A mistake: ChatGPT can no longer derive *D* when asked if it is certain." %}
-->

Of these three failure modalities, we focus on **rule suppression**, which is most closely related to jailbreak attacks, and refer to our paper for additional discussion on fact amnesia and state coercion.



## Subverting Inference in Transformers (Theory)

To better understand these three attacks, particularly rule suppression, we first study how transformer-based language models might perform logical inference in a simplified, theoretical setup.
Because propositional reasoning is fundamentally Boolean-valued, it is not surprising that **Boolean circuits** can implement algorithms for inference steps.
Our main strategy is as follows:
1. Construct a Boolean circuit that can reason over a binarized representation of the prompt.
2. Construct a transformer that approximates this reference Boolean circuit.
3. Attack this theoretically-crafted transformer.

<!-- image of the workflow -->

It turns out that even small transformers have the capacity to encode inference.

**Theorem.** (Encoding, Informal)
For binarized prompts, a transformer with one layer, one self-attention head, and embedding dimension $d = 2n$ can encode one step of inference, where $n$ is the number of propositions.
{: .notice--info}


Importantly, models at this size can still learn propositional inference to a high accuracy.


{% include gallery id="gallery_exp1" caption="Small transformers can learn propositional inference to high accuracy. Shown are the accuracies for autoregressively running $t = 1, 2, 3$ steps of inference, respectively. A model must correctly predict the state of all $n$ propositions up to $t$ steps to be counted as correct." %}


This provides a simplified theoretical setup in which to study attacks on rule-following.
In particular, the simplicity of our theoretical model lets us manually construct attacks that are guaranteed to work against it.


**Theorem.** (Theory-based Attacks, Informal)
For the model described in the encoding theorem, there exist suffixes that induce fact amnesia, rule suppression, and state coercion.
{: .notice--info}

To design rule suppression, for example, the main strategy is to create a suffix that diverts attention away from the targeted rule.

<!-- image of rule suppression for the binarized case -->


Interestingly, reasoners trained from data are also susceptible to attacks devised against theory.



## Real Jailbreaks Mirror Theory-based Ones

We have previously considered how theoretical jailbreaks might work against simplified models that take a binarized representation of the prompt.