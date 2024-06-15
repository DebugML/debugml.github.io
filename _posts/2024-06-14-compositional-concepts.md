---
published: true
title:  "Towards Compositionality in Concept Learning"
excerpt: "A method for learning compositional concepts from pre-trained foundation models."
header:
  overlay_color: "#000"
  overlay_filter: "0.5"
  overlay_image: assets/images/compositional_concepts/composition_mondrian.jpg
  teaser: assets/images/compositional_concepts/composition_mondrian.jpg
  actions:
    - label: "Paper"
      url: "https://arxiv.org/abs/"
    - label: "Code"
      url: "https://github.com/adaminsky/compositional_concepts"
authors: 
  - Adam Stein
  - Aaditya Naik
  - Yinjun Wu
  - Mayur Naik
  - Eric Wong

fig1:
  - url: "/assets/images/compositional_concepts/fig1.jpg"
    image_path: "/assets/images/compositional_concepts/fig1.jpg"
    alt: ""
    title: ""
birds_hands:
  - url: "/assets/images/compositional_concepts/birds_hands.jpg"
    image_path: "/assets/images/compositional_concepts/birds_hands.jpg"
    alt: ""
    title: ""
framed_birds:
  - url: "/assets/images/compositional_concepts/framed_birds.jpg"
    image_path: "/assets/images/compositional_concepts/framed_birds.jpg"
    alt: ""
    title: ""
gt_orthogonality:
  - url: "/assets/images/compositional_concepts/gt_orthogonality.jpg"
    image_path: "/assets/images/compositional_concepts/gt_orthogonality.jpg"
    alt: ""
    title: ""
method:
  - url: "/assets/images/compositional_concepts/method.png"
    image_path: "/assets/images/compositional_concepts/method.png"
    alt: ""
    title: ""
nlp_qual1:
  - url: "/assets/images/compositional_concepts/nlp_qual_1.jpg"
    image_path: "/assets/images/compositional_concepts/nlp_qual_1.jpg"
    alt: ""
    title: ""
nlp_qual2:
  - url: "/assets/images/compositional_concepts/nlp_qual_2.jpg"
    image_path: "/assets/images/compositional_concepts/nlp_qual_2.jpg"
    alt: ""
    title: ""
cub_acc:
  - url: "/assets/images/compositional_concepts/CUB (1) (1).jpg"
    image_path: "/assets/images/compositional_concepts/CUB (1) (1).jpg"
    alt: ""
    title: ""
ham_acc:
  - url: "/assets/images/compositional_concepts/ham (1) (1).jpg"
    image_path: "/assets/images/compositional_concepts/ham (1) (1).jpg"
    alt: ""
    title: ""
news_acc:
  - url: "/assets/images/compositional_concepts/News1 (1) (1).jpg"
    image_path: "/assets/images/compositional_concepts/News1 (1) (1).jpg"
    alt: ""
    title: ""
truth_acc:
  - url: "/assets/images/compositional_concepts/Truth (1) (1).jpg"
    image_path: "/assets/images/compositional_concepts/Truth (1) (1).jpg"
    alt: ""
    title: ""
---
<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>


> *Concept-based interpretability linearly decomposes a model’s hidden representation into vectors corresponding to human-interpretable concepts, but do these concepts really compose through addition? In our recent paper, we investigate this question and find that foundation models represent compositional concepts, such as “white bird” and “small bird”, in a way supporting composition through addition, but existing concept learning methods do not find such concepts. We propose Compositional Concept Extraction (CCE) as a way to encourage the learning of such compositional concepts.*

To describe something complicated we often rely on explanations using simpler components. For instance, a [brachiosaurus](https://brachiolab.github.io/) is a dinosaur which looks like a mixture of a lizard and a giraffe, and a dog is an animal with four legs, a tail, fur, and a snout. This is the *principle of compositionality* at work!

A promising method for understanding deep neural networks is to similarly break down their complex behavior into human-understandable components called concepts. Interestingly, past work such as [TCAV](https://proceedings.mlr.press/v80/kim18d/kim18d.pdf) and [Posthoc Concept Bottleneck Models](https://openreview.net/pdf?id=nA5AZ8CEyow) successfully ascribed human-interpretable concepts such as “fur” and “snout” to the features learned by modern deep learning models. For example, the following example shows the "color: white" and "size: 3-5in" concepts for the CLIP model that are learned by prior work and by our method which is explained later.

<!-- <Concept examples here> -->
<!-- ![Concept failure](assets/fig1.jpg) -->
{% include gallery id="fig1" layout="" caption="" %}

Many prior works, such as [IBD](https://openaccess.thecvf.com/content_ECCV_2018/papers/Antonio_Torralba_Interpretable_Basis_Decomposition_ECCV_2018_paper.pdf) and recently [TextSpan](https://openreview.net/pdf?id=5Ca9sSzuDp), use these discovered concepts to approximately reconstruct the hidden space of a model. This means that an image of a dog should be roughly encoded as a sum of the concepts for “fur”, “snout”, “four legs”, and “tail”. While the qualitative examples of individual concepts, as shown above, seem to show that the individual concepts correspond closely to human interpretable concepts, adding them together to compose them does not actually compose the concepts for the PCA method. Our method, which we will introduce later does in fact discover concepts which compose through addition as we see in the figure above.

## Studying Compositional Concept Representations Using Synthetic Data

First, we define a concept as a set of *symbols*, such as the concept $$\{\text{“tail”}\}$$ which we denote as “tail” for simplicity. A *concept representation* is denoted $$R(c)$$ where $$R: \mathbb{C}\rightarrow\mathbb{R}^d$$ where $$\mathbb{C}$$ is the set of all concepts and $$\mathbb{R}^d$$ is an embedding space in some dimension $$d$$. Since concepts are defined as sets, we allow them to be composed through the union operator such that “four legs and tail” = “four legs” $$\cup$$ “tail”. Therefore, compositional concept representations mean that concept representations should compose through addition whenever concepts compose through the union, or that:
> For concepts $$c_i, c_j \in \mathbb{C}$$, the concept representation $$R: \mathbb{C}\rightarrow\mathbb{R}^d$$ is compositional if for some $$w_{c_i}, w_{c_j}\in \mathbb{R}^+$$,
> $$R(c_i \cup c_j) = w_{c_i}R(c_i) + w_{c_j}R(c_j)$$.

Given these definitions, we start from the case where we have data with known concepts (we know some $$c_i$$'s) and we study the representation of the concepts (the $$R(c_i)$$'s).

To understand how concepts are actually represented by pretrained models we resort to synthetic data where we can get representations for ground truth concepts. We consider the [CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/) dataset which consists of generated images containing an object with two attributes, color and shape, and each image corresponds to one of three shapes (cube, sphere, cylinder) and one of three colors (red, green, blue). As each image is generated with exactly one shape and one color, we have annotations for which color and shape each image contains, allowing us to derive ground truth concept representations for the shape and color concepts. After centering all the representations, we define the ground truth representation for a concept similar to [existing work](https://openaccess.thecvf.com/content/ICCV2023/papers/Trager_Linear_Spaces_of_Meanings_Compositional_Structures_in_Vision-Language_Models_ICCV_2023_paper.pdf) as the mean representation of all samples annotated with the concept.

Our main finding from the ground truth concept representations for each shape and color (6 total concepts) reveal that CLIP encodes concepts of different attributes as orthogonal, but that concepts of the same attribute need not be orthogonal. We make this empirical observation from the cosine similarities between all pairs of ground truth concepts, shown below. The concept pairs of the same attribute have non-zero cosine similarity, while cross-attribute pairs have nearly zero cosine similarity, implying orthogonality.

<!-- <Heatmap> -->
<!-- ![GT Orthogonality](assets/gt_orthogonality.jpg) -->
{% include gallery id="gt_orthogonality" layout="" caption="" %}

We now see why existing concept learning methods find concepts which do not compose correctly through addition. Existing methods either impose too strong or too weak of a constraint on the orthogonality of discovered concepts. For instance, PCA requires that all concepts are orthogonal to each other, but concepts like “red” and “blue” should not be orthogonal. On the other hand, methods such as ACE place no restrictions on concept orthogonality, which means concepts such as “red” and “cube” may not be orthogonal.

While we show that the ground truth concepts display certain orthogonality structure, does that mean that concept representations must also display such structure to be compositional through addition? In our paper, we prove the answer is yes in a simplified setting!

We show a toy example below where we first show how concepts which follow the described orthogonality structure compose correctly, while a set of concepts which do not follow such structure exhibit unexpected compositions. In addition, the individual concept representations still perfectly discriminate between the individual concepts, so looking at each concept independently would mislead us to believe we have learned the correct concepts.

<!-- <Toy example> -->

Given these findings, we next outline our method for finding compositional concepts better than existing approaches.

## Compositional Concept Extraction

{% include gallery id="method" layout="" caption="" %}

How do we learn compositional concept representations from a pretrained model? Our findings from the synthetic experiments as described above show that compositional concepts will be represented such that concepts from different attributes are orthogonal to each other while concepts of the same attribute may not be orthogonal. To create this compositionality structure, we use an unsupervised iterative orthogonal projection approach. As shown in the diagram above, we can learn the concepts within one attribute using a directional clustering approach such as spherical K-Means, and then we project away all these concepts so that concepts discovered in the next iteration are orthogonal.


## Discovering New Concepts

Since our method is unsupervised, we can apply it to larger-scale datasets where we do not know all the relevant concepts and see what concepts are discovered. Click through the below visualization for some examples of the disovered concepts:

For a dataset of bird images (CUB):
<div class="image-selector-visualization">
    <style>
        .image-selector-visualization {
            display: flex;
            flex-direction: column;
            align-items: center;
            font-family: 'Arial', sans-serif;
            color: #333;
            margin: 0;
            padding: 0;
        }
        .image-selector-visualization h1 {
            margin-top: 5px;
            color: #007bff;
        }
        .image-selector-container {
            display: flex;
            justify-content: space-around;
            width: 90%;
            /* margin: 10px auto; */
            /* margin-top: 0px; */
            max-width: 1200px;
        }
        .image-selector-column {
            text-align: center;
            background: #fff;
            padding: 10px;
            border-radius: 10px;
            flex: 1;
            margin: 2px;
        }
        .image-selector-column h2 {
            color: #555;
        }
        .image-selector-select {
            width: 100%;
            padding: 10px;
            margin: 10px 0;
            font-size: 12px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .image-selector-image {
            display: none;
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            transition: opacity 0.3s ease-in-out;
        }
        .image-selector-image.show {
            display: block;
            opacity: 1;
        }
        #image-selector-title1 {
            font-size: 16px;
            margin-top: 5px;
        }
        #image-selector-title2 {
            font-size: 16px;
            margin-top: 5px;
        }
        #image-selector-title3 {
            font-size: 16px;
            margin-top: 5px;
        }
    </style>

    <div class="image-selector-container">
        <div class="image-selector-column">
            <div id="image-selector-title1">Select C1</div>
            <select id="image-selector1" class="image-selector-select" onchange="updateImageSelectorImages()">
                <option value="">Choose one</option>
                <option value="1">White birds</option>
                <option value="16">Brown birds</option>
            </select>
            <img id="image-selector-image1-1" class="image-selector-image" src="/assets/images/compositional_concepts/cub_1.png" alt="Image 1 Option 1">
            <img id="image-selector-image1-16" class="image-selector-image" src="/assets/images/compositional_concepts/cub_16.png" alt="Image 1 Option 2">
        </div>
        <div class="image-selector-column">
            <div id="image-selector-title2">Select C2</div>
            <select id="image-selector2" class="image-selector-select" onchange="updateImageSelectorImages()">
                <option value="">Choose one</option>
                <option value="47">Birds with food in mouth</option>
                <option value="35">Frames around image</option>
            </select>
            <img id="image-selector-image2-47" class="image-selector-image" src="/assets/images/compositional_concepts/cub_47.png" alt="Image 2 Option 1">
            <img id="image-selector-image2-35" class="image-selector-image" src="/assets/images/compositional_concepts/cub_35.png" alt="Image 2 Option 2">
        </div>
        <div class="image-selector-column">
            <div id="image-selector-title3">C1 + C2</div>
            <img id="image-selector-result-image" class="image-selector-image" src="" alt="Resulting Image">
        </div>
    </div>

    <script>
        function updateImageSelectorImages() {
            // Get the values of the selectors
            const selector1Value = document.getElementById('image-selector1').value;
            const selector2Value = document.getElementById('image-selector2').value;

            // Get the title elements
            const title1 = document.getElementById('image-selector-title1');
            const title2 = document.getElementById('image-selector-title2');

            // Hide all images initially
            document.querySelectorAll('.image-selector-image').forEach(img => {
                img.classList.remove('show');
            });

            // Update titles and show images based on the selectors
            if (selector1Value) {
                title1.textContent = "C1: " + document.querySelector(`#image-selector1 option[value="${selector1Value}"]`).textContent;
                document.getElementById(`image-selector-image1-${selector1Value}`).classList.add('show');
            } else {
                title1.textContent = "Select C1";
            }

            if (selector2Value) {
                title2.textContent = "C2: " + document.querySelector(`#image-selector2 option[value="${selector2Value}"]`).textContent;
                document.getElementById(`image-selector-image2-${selector2Value}`).classList.add('show');
            } else {
                title2.textContent = "Select C2";
            }

            // Show the resulting image based on the combination of the two selectors
            if (selector1Value && selector2Value) {
                const resultImage = document.getElementById('image-selector-result-image');
                resultImage.src = `/assets/images/compositional_concepts/cub_${selector1Value}_${selector2Value}.png`;
                resultImage.classList.add('show');
            } else {
                document.getElementById('image-selector-result-image').classList.remove('show');
            }
        }
    </script>
</div>

<!-- <Qualitative examples> -->
<!-- ![Qual1](/assets/images/compositional_concepts/framed_birds.jpg) 
![Qual2](/assets/images/compositional_concepts/birds_hands.jpg) -->

<br>
Examples of concepts on language data:
<ul class="tab" data-tab="44bf2f41-34a3-4bd7-b605-29d394ac9b0f" data-name="tasks">
      <li class="active">
          <a href="#">Example 1</a>
      </li>
  
      <li class="">
          <a href="#">Example 2</a>
      </li>
</ul>
<ul class="tab-content" id="44bf2f41-34a3-4bd7-b605-29d394ac9b0f" data-name="tasks">
  
<li class="active">
<!-- <p class="notice"><strong>Math Reasoning</strong>: Given a math question, we want to obtain the answer as a real-valued number. Here, we use Python as the symbolic language and the Python Interpreter as the determinstic solver. Below is an example from <a href="https://github.com/openai/grade-school-math">GSM8K</a>, a dataset of grade-school math questions.</p> -->

{% include gallery id="nlp_qual1" layout="" caption="" %}
</li>

<li class="">
<!-- <p class="notice"><strong>Math Reasoning</strong>: Given a math question, we want to obtain the answer as a real-valued number. Here, we use Python as the symbolic language and the Python Interpreter as the determinstic solver. Below is an example from <a href="https://github.com/openai/grade-school-math">GSM8K</a>, a dataset of grade-school math questions.</p> -->

{% include gallery id="nlp_qual2" layout="" caption="" %}
</li>
</ul>





## CCE Concepts are More Compositional

How do we measure the compositionality of a set of concept representations? Compositionality has been evaluated in [existing work](https://openreview.net/pdf?id=HJz05o0qK7) on representation learning, and we adapt these metrics for concept learning. To measure compositionality, we assume that a dataset with labeled concepts is used and we evaluate how well the discovered concepts match the labeled concepts and their compositionality structure.

The compositionality score of discovered concepts on a dataset $$D$$ where each sample embedding $$z$$ has associated concepts $$C$$ is given by the following:

$$\min_{\Lambda \ge 0} \frac{1}{|D|}\sum_{(z, C)\in D} \left\|z - \sum_{i=1}^{|C|} \Lambda_{z, i}R(C_i)\right\|$$

For a sample such as an image of a blue cube with concepts “blue” and “cube”, this score represents how well the embedding of the image can be reconstructed as a sum of the discovered “blue” and “cube” concept representations.

Compositionality scores for all baselines and CCE are shown below for the CLEVR dataset as well as two other datasets introduced in our paper, where smaller scores are better. We see that for all datasets, CCE has the lowest compositionality score, implying higher compositionality of the discovered concepts.

|           | CLEVR             | CUB-sub           | Truth-sub         |
|:----------|:------------------|:------------------|:------------------|
| GT        | 3.162 $$\pm$$ 0.000 | 0.472 $$\pm$$ 0.000 | 3.743 $$\pm$$ 0.000 |
| PCA       | 3.684 $$\pm$$ 0.000 | 0.481 $$\pm$$ 0.000 | 3.988 $$\pm$$ 0.000 |
| ACE       | 3.496 $$\pm$$ 0.116 | 0.502 $$\pm$$ 0.008 | 3.727 $$\pm$$ 0.032 |
| DictLearn | 3.387 $$\pm$$ 0.007 | 0.503 $$\pm$$ 0.002 | 3.708 $$\pm$$ 0.007 |
| NML       | 3.761 $$\pm$$ 0.050 | 0.542 $$\pm$$ 0.001 | 3.812 $$\pm$$ 0.063 |
| CT        | 4.931 $$\pm$$ 0.001 | 0.546 $$\pm$$ 0.000 | 4.348 $$\pm$$ 0.000 |
| Random    | 4.927 $$\pm$$ 0.001 | 0.546 $$\pm$$ 0.000 | 4.348 $$\pm$$ 0.000 |
| CCE       | 3.163 $$\pm$$ 0.000 | 0.459 $$\pm$$ 0.004 | 3.689 $$\pm$$ 0.002 |


## CCE Concepts Result in Better Downstream Classification Accuracy

A central use-case for concepts is for interpretable classification with [posthoc concept-bottleneck models](https://openreview.net/pdf?id=nA5AZ8CEyow). For four datasets spanning image and text domains, we evaluate CCE concepts against baselines in terms of classification accuracy after training a posthoc concept-bottleneck model on the extracted concepts. We show classification accuracy with increasing numbers of extracted concepts in the below figure, and we see that CCE always achieves the highest accuracy or near-highest accuracy.

<!-- <Figure of downstream accuracy> -->
<ul class="tab" data-tab="44bf2f41-34a3-4bd7-b605-29d394ac9b0" data-name="tasks_acc">
      <li class="active">
          <a href="#">CUB</a>
      </li>
  
      <li class="">
          <a href="#">HAM</a>
      </li>
      <li class="">
          <a href="#">News</a>
      </li>
      <li class="">
          <a href="#">Truth</a>
      </li>
</ul>
<ul class="tab-content" id="44bf2f41-34a3-4bd7-b605-29d394ac9b0" data-name="tasks_acc">
  
<li class="active">
<!-- <p class="notice"><strong>Math Reasoning</strong>: Given a math question, we want to obtain the answer as a real-valued number. Here, we use Python as the symbolic language and the Python Interpreter as the determinstic solver. Below is an example from <a href="https://github.com/openai/grade-school-math">GSM8K</a>, a dataset of grade-school math questions.</p> -->

{% include gallery id="cub_acc" layout="" caption="" %}
</li>

<li class="">
<!-- <p class="notice"><strong>Math Reasoning</strong>: Given a math question, we want to obtain the answer as a real-valued number. Here, we use Python as the symbolic language and the Python Interpreter as the determinstic solver. Below is an example from <a href="https://github.com/openai/grade-school-math">GSM8K</a>, a dataset of grade-school math questions.</p> -->

{% include gallery id="ham_acc" layout="" caption="" %}
</li>
<li class="">
<!-- <p class="notice"><strong>Math Reasoning</strong>: Given a math question, we want to obtain the answer as a real-valued number. Here, we use Python as the symbolic language and the Python Interpreter as the determinstic solver. Below is an example from <a href="https://github.com/openai/grade-school-math">GSM8K</a>, a dataset of grade-school math questions.</p> -->

{% include gallery id="news_acc" layout="" caption="" %}
</li>
<li class="">
<!-- <p class="notice"><strong>Math Reasoning</strong>: Given a math question, we want to obtain the answer as a real-valued number. Here, we use Python as the symbolic language and the Python Interpreter as the determinstic solver. Below is an example from <a href="https://github.com/openai/grade-school-math">GSM8K</a>, a dataset of grade-school math questions.</p> -->

{% include gallery id="truth_acc" layout="" caption="" %}
</li>
</ul>

## Conclusion

Compositionality is a desired property of concept representations as concepts themselves are often compositional, but we show that existing concept learning methods do not always learn concept representations which compose through addition. After studying the representation of concepts in a synthetic setting we find two salient properties of compositional concept representations, and we propose a concept learning method, CCE, which leverages our insights to learn compositional concepts. CCE finds more compositional concepts than existing techniques, results in better downstream accuracy when used as the features in a posthoc concept-bottleneck model, and even discovers new compositional concepts as shown through our qualitative examples.

If interested, please check out the details in our paper [here]()! Our code is available [here](https://github.com/adaminsky/compositional_concepts), and you can easily apply CCE to your own dataset or adapt our code to create new concept learning methods.
