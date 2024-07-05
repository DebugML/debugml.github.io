---
published: true
title:  "Towards Compositionality in Concept Learning"
excerpt: "A method for learning compositional concepts from pre-trained foundation models."
header:
  overlay_color: "#000"
  overlay_filter: "0.5"
  overlay_image: assets/images/compositional_concepts/composition_mondrian.jpg
  teaser: assets/images/compositional_concepts/birds_hands.jpg
  actions:
    - label: "Paper"
      url: "https://arxiv.org/abs/2406.18534"
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
  - url: "/assets/images/compositional_concepts/cross_similarities_CUB_subset2.png"
    image_path: "/assets/images/compositional_concepts/cross_similarities_CUB_subset2.png"
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
image-selector-image1-0:
  - url: "/assets/images/compositional_concepts/cub_0.png"
    image_path: "/assets/images/compositional_concepts/cub_0.png"
    alt: ""
    title: ""
---
<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@3.7"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-chart-matrix@1.1"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels"></script>


> *Concept-based interpretability represents human-interpretable concepts such as "white bird" and "small bird" as vectors in the embedding space of a deep network. But do these concepts really compose together? It turns out that existing methods find concepts that behave unintuitively when combined. To address this, we propose Compositional Concept Extraction (CCE), a new concept learning approach that encourages concepts that linearly compose.*

To describe something complicated we often rely on explanations using simpler components. For instance, a small white bird can be described by separately describing what small birds and white birds look like. This is the *principle of compositionality* at work!

<figure>
    <style>
        .container {
            display: grid;
            grid-template-columns: auto 1fr auto 1fr auto 1fr;
            gap: 10px;
            align-items: center;
            text-align: center;
        }
        .section-title {
            writing-mode: vertical-rl;
            text-orientation: mixed;
            transform: rotate(180deg);
            font-weight: bold;
        }
        .img-container {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .img-container img {
            width: 150px;
            height: auto;
            margin-bottom: 5px;
        }
        .operation {
            font-size: 24px;
            font-weight: bold;
        }
        .column-title {
            font-weight: bold;
            margin-bottom: 10px;
        }
    </style>
    <div class="container">
        <!-- PCA Concepts Section -->
        <div>
            <div class="column-title">color: white</div>
            <div class="img-container">
                <img src="/assets/images/compositional_concepts/cub_pca_white_1.jpg" alt="PCA color: white image 1">
                <img src="/assets/images/compositional_concepts/cub_pca_white_2.jpg" alt="PCA color: white image 2">
            </div>
        </div>
        <div class="operation"><br>+</div>
        <div>
            <div class="column-title">size: 3-5in</div>
            <div class="img-container">
                <img src="/assets/images/compositional_concepts/cub_pca_small_1.jpg" alt="PCA size: 3-5in image 1">
                <img src="/assets/images/compositional_concepts/cub_pca_small_2.jpg" alt="PCA size: 3-5in image 2">
            </div>
        </div>
        <div class="operation"><br>=</div>
        <div>
            <div class="column-title">?</div>
            <div class="img-container">
                <img src="/assets/images/compositional_concepts/cub_pca_comp_1.jpg" alt="PCA result image 1">
                <img src="/assets/images/compositional_concepts/cub_pca_comp_2.jpg" alt="PCA result image 2">
            </div>
        </div>
    </div>
    <figcaption>PCA-based concepts for the CLIP model do not compose. The first column depicts the "white birds" concept by showing the two samples closest to the concept representation. The second column shows the "small birds" concept and the two closest images are small birds in this case. The last column shows the composition of the two preceding concept representations.</figcaption>
</figure>


Concept-based explanations [[Kim et. al.](https://proceedings.mlr.press/v80/kim18d/kim18d.pdf), [Yuksekgonul et. al.](https://openreview.net/pdf?id=nA5AZ8CEyow)] aim to map these human-interpretable concepts such as "small bird" and "white bird" to the features learned by deep networks. For example, in the above figure, we visualize the "white bird" and "small bird" concepts discovered in the hidden representations from [CLIP](https://arxiv.org/abs/2103.00020) using a [PCA](https://arxiv.org/pdf/2310.01405)-based approach on a dataset of bird images. The "white bird" concept is close to birds that are indeed white, while the "small bird" concept indeed captures small birds. However, the composition of these two PCA-based concepts results in a concept depicted in the above figure on the right which is *not* close to small and white birds.

Composition of the "white bird" and "small bird" concepts is expected to look like the following figure. The "white bird" concept is close to white bird images, the "small bird" concept is close to small bird images, and the composition of the two concepts is indeed close to images of small white birds!

<figure>
    <style>
        .container {
            display: grid;
            grid-template-columns: auto 1fr auto 1fr auto 1fr;
            gap: 10px;
            align-items: center;
            text-align: center;
            margin-bottom: 20px;
        }
        .section-title {
            writing-mode: vertical-rl;
            text-orientation: mixed;
            transform: rotate(180deg);
            font-weight: bold;
        }
        .img-container {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .img-container img {
            width: 150px;
            height: auto;
            margin-bottom: 5px;
        }
        .operation-container {
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .operation {
            font-size: 24px;
            font-weight: bold;
        }
        .column-title {
            font-weight: bold;
            margin-bottom: 10px;
        }
    </style>
    <div class="container">
        <!-- PCA Concepts Section -->
        <div>
            <br>
            <div class="column-title">color: white</div>
            <div class="img-container">
                <img src="/assets/images/compositional_concepts/cub_ours_white_1.jpg" alt="PCA color: white image 1">
                <img src="/assets/images/compositional_concepts/cub_ours_white_2.jpg" alt="PCA color: white image 2">
            </div>
        </div>
        <div class="operation-container">
            <br>
            <br>
            <div class="operation">+</div>
        </div>
        <div>
            <br>
            <div class="column-title">size: 3-5in</div>
            <div class="img-container">
                <img src="/assets/images/compositional_concepts/cub_ours_small_1.jpg" alt="PCA size: 3-5in image 1">
                <img src="/assets/images/compositional_concepts/cub_ours_small_2.jpg" alt="PCA size: 3-5in image 2">
            </div>
        </div>
        <div class="operation-container">
            <br>
            <br>
            <div class="operation">=</div>
        </div>
        <div>
            <div class="column-title">color: white <br> size: 3-5in</div>
            <div class="img-container">
                <img src="/assets/images/compositional_concepts/cub_ours_comp_1.jpg" alt="PCA result image 1">
                <img src="/assets/images/compositional_concepts/cub_ours_comp_2.jpg" alt="PCA result image 2">
            </div>
        </div>
    </div>
<figcaption>Our method (CCE) discovers concepts which compose. The "white birds" concept on the left indeed is close to images of white birds, the "small birds" concept in the middle is close to images of small birds, and the composition of these concepts is close to images of small and white birds.</figcaption>
</figure>

We achieve this by first understanding the properties of compositional concepts in the embedding space of deep networks and then proposing a method to discover such concepts.

## Compositional Concept Representations

To understand concept compositionality, we first need a definition of concepts.
Abstractly, the concept "small bird" is nothing more than the *symbols* used to type it.
Therefore, we define a concept as a set of symbols.
<!-- , such as the concept $$\{``\text{small bird"}\}$$ which we denote as $$``\text{small bird"}$$ for simplicity. -->

A *concept representation* maps between the symbolic form of the concept, such as $$``\text{small bird"}$$, into a vector in a deep network's embedding space. A concept representation is denoted $$R: \mathbb{C}\rightarrow\mathbb{R}^d$$ where $$\mathbb{C}$$ is the set of all concept names and $$\mathbb{R}^d$$ is an embedding space with dimension $$d$$.

To compose concepts, we take the union of their set-based representation. For instance, $$``\text{small bird"} \cup ``\text{white bird"} = ``\text{small white bird"}$$. Concept representations, on the other hand, compose through vector addition. Therefore, we define *compositional concept representations* to mean concept representations which compose through addition whenever their corresponding concepts compose through the union, or that:

**Definition:** For concepts $$c_i, c_j \in \mathbb{C}$$, the concept representation $$R: \mathbb{C}\rightarrow\mathbb{R}^d$$ is compositional if for some $$w_{c_i}, w_{c_j}\in \mathbb{R}^+$$,
$$R(c_i \cup c_j) = w_{c_i}R(c_i) + w_{c_j}R(c_j)$$.
{: .notice--info}


## Why Don't Traditional Concepts Compose?

Traditional concepts don't compose since existing concept learning methods over or under constrain concept representation orthogonality. For instance, PCA requires all concept representations to be orthogonal while methods such as [ACE](https://proceedings.neurips.cc/paper_files/paper/2019/file/77d2afcb31f6493e350fca61764efb9a-Paper.pdf) from Ghorbani et. al. place no restrictions on concept orthogonality.

We discover the expected orthogonality structure of concept representations using a dataset 
where each sample is annotated with concept names (we know some $$c_i$$'s) and we study the representation of the concepts (the $$R(c_i)$$'s).
We create such a setting by subsetting the bird data from [CUB](https://www.vision.caltech.edu/datasets/cub_200_2011/) to only contain birds of three colors (black, brown, or white) and three sizes (small, medium, or large) according to the dataset's finegrained annotations.

<!-- To understand how concepts are actually represented by pre-trained models we use a controlled data setting where we can get representations for ground truth concepts. We start with the bird dataset, called [CUB](https://www.vision.caltech.edu/datasets/cub_200_2011/), used up to this point consisting of different bird species annotated with finegrained attributes. To create a controlled setting, we subset the data to only contain birds of three colors (black, brown, or white) and three sizes (small, medium, or large) according to the finegrained annotations. -->

Each image now contains a bird annotated as exactly one size and one color, so we derive ground truth concept representations for the bird shape and size concepts. To do so, we center all the representations, and we define the ground truth representation for a concept similar to [existing work](https://openaccess.thecvf.com/content/ICCV2023/papers/Trager_Linear_Spaces_of_Meanings_Compositional_Structures_in_Vision-Language_Models_ICCV_2023_paper.pdf) as the mean representation of all samples annotated with the concept.

Our main finding from analyzing the ground truth concept representations for each bird size and color (6 total concepts) is that CLIP encodes concepts of different attributes (colors vs. sizes) as orthogonal, but that concepts of the same attribute (e.g. different colors) need not be orthogonal. We make this empirical observation from the cosine similarities between all pairs of ground truth concepts, shown below.


<!-- <Heatmap> -->
<!-- ![GT Orthogonality](assets/gt_orthogonality.jpg) -->
<!-- {% include gallery id="gt_orthogonality" layout="" caption="Cosine similarities of all pairs of concepts. We can see that concepts within an attribute (red, green, and blue or sphere, cube, and cylinder) have non-zero cosine similarity, while the cosine similarity of concepts from different attributes are all nearly zero." %} -->

<figure>
<div class="chartcontainer" style="width: 400px; height: 400px; margin-bottom: 10px; margin: auto">
    <canvas id="matrix-chart" width="300" height="300"></canvas>
</div>
<figcaption>Cosine similarities of all pairs of concepts in the controlled setting for the bird images dataset. Concepts within an attribute (brown, white, and black or small, medium, and large) have non-zero cosine similarity, while the cosine similarity of concepts from different attributes are close to zero. We find this orthogonality structure is important for the compositionality of concept representations.</figcaption>
</figure>
<script>
    const labels = ['brown', 'white', 'black', 'small', 'medium', 'large'];
    const data = [
        [1.00, -0.53, -0.26, 0.33, -0.26, -0.32],
        [-0.53, 1.00, -0.68, -0.28, 0.24, 0.26],
        [-0.26, -0.68, 1.00, 0.04, -0.06, -0.01],
        [0.33, -0.28, 0.04, 1.00, -0.87, -0.90],
        [-0.26, 0.24, -0.06, -0.87, 1.00, 0.56],
        [-0.32, 0.26, -0.01, -0.90, 0.56, 1.00]
    ];

    const chartData = data.flatMap((row, y) => 
        row.map((value, x) => ({x, y, v: value}))
    );

    const chart = new Chart('matrix-chart', {
        type: 'matrix',
        plugins: [ChartDataLabels],
        data: {
            datasets: [{
                label: 'Correlation Matrix',
                data: chartData,
                borderWidth: 1,
                borderColor: 'white',
                backgroundColor: (context) => {
                    const value = context.dataset.data[context.dataIndex].v;
                    const alpha = Math.abs(value);
                    return value < 0 
                        ? `rgba(0, 0, 255, ${alpha})`  // Blue for negative
                        : `rgba(0, 0, 255, ${alpha})`  // Blue for negative
                },
                width: ({chart}) => (chart.chartArea || {}).width / 6 - 1,
                height: ({chart}) => (chart.chartArea || {}).height / 6 - 1,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                x: {
                    ticks: {
                        callback: (value) => labels[value],
                    },
                    grid: {
                        display: false
                    }
                },
                y: {
                    offset: true,
                    reverse: true,
                    ticks: {
                        callback: (value) => labels[value],
                    },
                    grid: {
                        display: false
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        title: () => '',
                        label: (context) => {
                            const value = context.dataset.data[context.dataIndex].v;
                            return `${value.toFixed(2)}`;
                        }
                    }
                },
                datalabels: {
                        display: true,
                        color: 'black',
                        font: {
                            weight: 'bold'
                        },
                        formatter: (value) => value.v.toFixed(2),
                        textAlign: 'center',
                        textStrokeColor: 'white',
                        textStrokeWidth: 0,
                        anchor: 'center',
                        clip: true
                }
            }
        }
    });
</script>


**Observation:** The concept pairs of the same attribute have non-zero cosine similarity, while cross-attribute pairs have close to zero cosine similarity, implying orthogonality.
{: .notice--info}

<!-- We now see why existing concept learning methods find concepts which do not compose correctly through addition. Existing methods either impose too strong or too weak of a constraint on the orthogonality of discovered concepts. For instance, PCA requires that all concepts are orthogonal to each other, but concepts like "white" and "black" should not be orthogonal. On the other hand, methods such as [ACE](https://proceedings.neurips.cc/paper_files/paper/2019/file/77d2afcb31f6493e350fca61764efb9a-Paper.pdf) from Ghorbani et. al. place no restrictions on concept orthogonality, which means concepts such as "black" and "small" may not be orthogonal. -->

While the ground truth concept representations display this orthogonality structure, must all compositional concept representations mimick this structure? In our paper, we prove the answer is yes in a simplified setting!

Given these findings, we next outline our method for finding compositional concepts which follow this orthogonality structure.

## Compositional Concept Extraction

{% include gallery id="method" layout="" caption="Depiction of CCE. There are two high level components, LearnSubspace and LearnConcepts, which are performed jointly to discover a subspace and concepts within the subspace. Then the subspace is orthogonally projected from the model's embedding space, to ensure orthogonality, and we repeat the process." %}

Our findings from the synthetic experiments show that compositional concepts are represented such that different attributes are orthogonal while concepts of the same attribute may not be orthogonal. To create this structure, we use an unsupervised iterative orthogonal projection approach.

First, orthogonality between groups of concepts is enforced through orthogonal projection. Once we find one set of concept representations (which may correspond to different values of an attribute such as different colors) we project away the subspace which they span from the model's embedding space so that all further discovered concepts are orthogonal to the concepts within the subspace.

To find the concepts within a subspace, we jointly learn a subspace (with *LearnSubspace*) and a set of concepts (with *LearnConcepts*). The figure above illustrates the high level algorithm. Given a subspace $$P$$, the LearnConcepts step finds a set of concepts within $$P$$ which are well clustered. On the other hand, the LearnSubspace step is given a set of concept representations and tries to find an optimal subspace in which the given concepts are maximally clustered. Since these steps are mutually dependent, we jointly learn both the subspace $$P$$ and the concepts within the subspace.

The full algorithm operates by finding a subspace and concepts within the subspace, then projecting away the subspace from the model's embedding space and repeating. All subspaces are therefore mutually orthogonal, but the concepts within one subspace may not be orthogonal, as desired.

<!-- Running one iteration of CCE results in a subspace $$P$$ and a set of concepts within that subspace. For the next iteration of CCE, we remove the subspace $$P$$ from the embedding space and repeat the algorithm. This removal process guarantees that all concepts discovered in iteration $$i$$ are orthogonal to all concepts discovered in iterations $$j < i$$. This mirrors the orthogonality structure we previously described since concepts within one discovered subspace may not be orthonal, but the concepts in different subspaces will be orthogonal. Therefore, CCE is an unsupervised alorithm for finding concepts divided into orthogonal subspaces. -->


## Discovering New Compositional Concepts

We qualitatively show that on larger-scale datasets, CCE discovers compositional concepts. Click through the below visualizations for examples of the disovered concepts on image and language data.

For a dataset of bird images (CUB):
<figure>
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
            width: 100%;
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
                <!-- <option value="">Choose one</option> -->
                <option value="1">White birds</option>
                <option value="16">Brown birds</option>
                <option value="0">Small green birds</option>
                <option value="8">Woodpeckers</option>
                <option value="15">Birds with water</option>
                <option value="7">Birds in water</option>
            </select>
            <a href="/assets/images/compositional_concepts/cub_1.png">
            <img id="image-selector-image1-1" class="image-selector-image" src="/assets/images/compositional_concepts/cub_1.png" alt="Image 1 Option 1">
            </a>

            <a href="/assets/images/compositional_concepts/cub_16.png">
            <img id="image-selector-image1-16" class="image-selector-image" src="/assets/images/compositional_concepts/cub_16.png" alt="Image 1 Option 2">
            </a>

            <a href="/assets/images/compositional_concepts/cub_0.png">
            <img id="image-selector-image1-0" class="image-selector-image" src="/assets/images/compositional_concepts/cub_0.png" alt="Image 1 Option 3">
            </a>

            <a href="/assets/images/compositional_concepts/cub_8.png">
            <img id="image-selector-image1-8" class="image-selector-image" src="/assets/images/compositional_concepts/cub_8.png" alt="Image 1 Option 4">
            </a>

            <a href="/assets/images/compositional_concepts/cub_15.png">
            <img id="image-selector-image1-15" class="image-selector-image" src="/assets/images/compositional_concepts/cub_15.png" alt="Image 1 Option 5">
            </a>

            <a href="/assets/images/compositional_concepts/cub_7.png">
            <img id="image-selector-image1-7" class="image-selector-image" src="/assets/images/compositional_concepts/cub_7.png" alt="Image 1 Option 6">
            </a>
        </div>
        <div class="image-selector-column">
            <div id="image-selector-title2">Select C2</div>
            <select id="image-selector2" class="image-selector-select" onchange="updateImageSelectorImages()">
                <!-- <option value="">Choose one</option> -->
                <option value="47">Birds eating food</option>
                <option value="35">Frames around image</option>
            </select>
            <a href="/assets/images/compositional_concepts/cub_47.png">
            <img id="image-selector-image2-47" class="image-selector-image" src="/assets/images/compositional_concepts/cub_47.png" alt="Image 2 Option 1">
            </a>

            <a href="/assets/images/compositional_concepts/cub_35.png">
            <img id="image-selector-image2-35" class="image-selector-image" src="/assets/images/compositional_concepts/cub_35.png" alt="Image 2 Option 2">
            </a>
        </div>
        <div class="image-selector-column">
            <div id="image-selector-title3">C1 + C2<br><br><br></div>
            <a id="image-selector-result-a" href="">
            <img id="image-selector-result-image" class="image-selector-image" src="" alt="Resulting Image">
            </a>
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
                const resulta = document.getElementById('image-selector-result-a');
                const resultImage = document.getElementById('image-selector-result-image');
                resulta.href = `/assets/images/compositional_concepts/cub_${selector1Value}_${selector2Value}.png`;
                resultImage.src = `/assets/images/compositional_concepts/cub_${selector1Value}_${selector2Value}.png`;
                resultImage.classList.add('show');
            } else {
                document.getElementById('image-selector-result-image').classList.remove('show');
            }

        }
        document.addEventListener("DOMContentLoaded", function() {
            updateImageSelectorImages();
        });
    </script>
</div>
<figcaption>Interactive visualization of some discovered compositional concepts on the CUB dataset. The concepts in the first two columns compose to form the concept in the third column.</figcaption>
</figure>

<!-- <Qualitative examples> -->
<!-- ![Qual1](/assets/images/compositional_concepts/framed_birds.jpg) 
![Qual2](/assets/images/compositional_concepts/birds_hands.jpg) -->

For a dataset of text newsgroup postings:
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

<figure>
<div style="display: flex; flex-direction: column; width: 100%; max-width: 800px; margin: 20px auto; padding: 10px; box-sizing: border-box; position: relative; font-size: 14px;">
  <div style="display: flex; margin-bottom: 10px;">
    <div style="flex: 1; text-align: center; font-weight: bold;">Text Ending in "..."</div>
    <div style="flex: 1; text-align: center; font-weight: bold;">Sports</div>
    <div style="flex: 1; text-align: center; font-weight: bold;">Sports text ending in "..."</div>
  </div>
  <div style="display: flex; align-items: stretch;">
    <div style="flex: 1; display: flex; flex-direction: column; margin-right: 5px;">
      <div style="flex: 1; padding: 10px; background-color: #ffffff; margin-bottom: 5px; border: 1px solid; border-radius: 5px;">
        <p style="margin: 0;">Hopefully, he doesn't take it personal...</p>
      </div>
      <div style="flex: 1; padding: 10px; background-color: #ffffff; margin-bottom: 5px; border: 1px solid; border-radius: 5px;">
        <p style="margin: 5px 0 0 0;">Hi there, maybe you can help me...</p>
      </div>
    </div>
    <div style="display: flex; align-items: center; font-size: 24px; margin: 0 10px;">+</div>
    <div style="flex: 1; display: flex; flex-direction: column; margin: 0 5px;">
      <div style="flex: 1; padding: 10px; background-color: #fffacd; margin-bottom: 5px; border: 1px solid; border-radius: 5px;">
        <p style="margin: 0;">If I were Pat Burns I'd throw in the towel. The wings dominated every aspect of the game.</p>
      </div>
      <div style="flex: 1; padding: 10px; background-color: #fffacd; border: 1px solid; border-radius: 5px;">
        <p style="margin: 0;">Quebec dominated Habs for first 2 periods and only Roy kept this one from being rout, although he did blow 2nd goal.</p>
      </div>
    </div>
    <div style="display: flex; align-items: center; font-size: 24px; margin: 0 10px;">=</div>
    <div style="flex: 1; display: flex; flex-direction: column; margin-left: 5px;">
      <div style="flex: 1; padding: 10px; background-color: #e6f3ff; margin-bottom: 5px; border: 1px solid; border-radius: 5px;">
        <p style="margin: 0;">Grant Fuhr has done this to a lot better coaches than Brian Sutter...</p>
      </div>
      <div style="flex: 1; padding: 10px; background-color: #e6f3ff; border: 1px solid; border-radius: 5px;">
        <p style="margin: 0;">No, although since the Lavalliere weirdness, nothing would really surprise me. Jeff King is currently in the top 10 in the league in *walks*. Something is up...</p>
      </div>
    </div>
  </div>
</div>
<figcaption>Discovered concepts from the <a href="http://qwone.com/~jason/20Newsgroups/">Newsgroups</a> dataset. The "Text ending in ..." concept is close to text which all ends in "...", the "Sports" concept is close to articles about sports, and the compostion of these concepts is close to samples about sports that end in "...".</figcaption>
</figure>
</li>

<li class="">

<figure>
<div style="display: flex; flex-direction: column; width: 100%; max-width: 800px; margin: 20px auto; padding: 10px; box-sizing: border-box; position: relative; font-size: 14px;">
  <div style="display: flex; margin-bottom: 10px;">
    <div style="flex: 1; text-align: center; font-weight: bold;">Asking for suggestions</div>
    <div style="flex: 1; text-align: center; font-weight: bold;">Items for sale</div>
    <div style="flex: 1; text-align: center; font-weight: bold;">Asking for purchasing suggestions</div>
  </div>
  <div style="display: flex; align-items: stretch;">
    <div style="flex: 1; display: flex; flex-direction: column; margin-right: 5px;">
      <div style="flex: 1; padding: 10px; background-color: #ffffff; margin-bottom: 5px; border: 1px solid; border-radius: 5px;">
        <p style="margin: 0;">HELP!<br>I am trying to find software that will allow COM port redirection [...] Can anyone out their make a suggestion or recommend something.</p>
      </div>
      <div style="flex: 1; padding: 10px; background-color: #ffffff; margin-bottom: 5px; border: 1px solid; border-radius: 5px;">
        <p style="margin: 5px 0 0 0;">Hi all,<br>I am looking for a new oscilloscope [...] and would like suggestions on a low-priced source for them.</p>
      </div>
    </div>
    <div style="display: flex; align-items: center; font-size: 24px; margin: 0 10px;">+</div>
    <div style="flex: 1; display: flex; flex-direction: column; margin: 0 5px;">
      <div style="flex: 1; padding: 10px; background-color: #fffacd; margin-bottom: 5px; border: 1px solid; border-radius: 5px;">
        <p style="margin: 0;">Please reply to the seller below.<br>For Sale:<br>Sun SCSI-2 Host Adapter Assembly [...]</p>
      </div>
      <div style="flex: 1; padding: 10px; background-color: #fffacd; border: 1px solid; border-radius: 5px;">
        <p style="margin: 0;">Please reply to the seller below.<br>210M Formatted SCSI Hard Disk 3.5" [...]</p>
      </div>
    </div>
    <div style="display: flex; align-items: center; font-size: 24px; margin: 0 10px;">=</div>
    <div style="flex: 1; display: flex; flex-direction: column; margin-left: 5px;">
      <div style="flex: 1; padding: 10px; background-color: #e6f3ff; margin-bottom: 5px; border: 1px solid; border-radius: 5px;">
        <p style="margin: 0;">Which would YOU choose, and why?<br><br>Like lots of people, I'd really like to increase my data transfer rate from</p>
      </div>
      <div style="flex: 1; padding: 10px; background-color: #e6f3ff; border: 1px solid; border-radius: 5px;">
        <p style="margin: 0;">Hi all,<br>I am looking for a new oscilloscope [...] and would like suggestions on a low-priced source for them.</p>
      </div>
    </div>
  </div>
</div>
<figcaption>Discovered concepts from the <a href="http://qwone.com/~jason/20Newsgroups/">Newsgroups</a> dataset. The "Asking for suggestions" concept is close to text where someone asks others for suggestions, the "Items for sale" concept is close to ads which are listing items available for purchase, and the compostion of these concepts is close to samples where someone asks for suggestions about purchasing a new item.</figcaption>
</figure>

</li>
</ul>


<!-- ## CCE Concepts are Compositional -->

<!-- Compositionality has been evaluated for representation learning methods ([Andreas](https://openreview.net/pdf?id=HJz05o0qK7)), but we adapt the evaluation for concept learning methods. -->
<!-- To measure compositionality in concept learning, we need a dataset with labeled concepts. For an image of a small white bird with concepts "small bird" and "white bird", we measure how well a sum of the discovered "small bird" and "white bird" concepts can reconstruct the embedding of the image. -->

<!-- Generally, for a sample labelled with certain concepts, the compositionality score measures how the corresponding concept representations reconstruct the sample's embedding. -->
<!-- This is similar to the reconstruction metric for techniques such as PCA, but it only allows reconstruction with the concept representations of the concepts present in a sample. -->

CCE also finds concepts which are quantitatively compositional.
Compositionality scores for all baselines and CCE are shown below for the CUB dataset as well as two other datasets, where smaller scores mean greater compositionality. CCE discovers the most compositional concepts compared to existing methods.

<!-- |           | CLEVR             | CUB-sub           | Truth-sub         |
|:----------|:------------------|:------------------|:------------------|
| *GT*        | *3.162 $$\pm$$ 0.000* | *0.472 $$\pm$$ 0.000* | *3.743 $$\pm$$ 0.000* |
| [PCA](https://arxiv.org/pdf/2310.01405)       | 3.684 $$\pm$$ 0.000 | 0.481 $$\pm$$ 0.000 | 3.988 $$\pm$$ 0.000 |
| [ACE](https://proceedings.neurips.cc/paper_files/paper/2019/file/77d2afcb31f6493e350fca61764efb9a-Paper.pdf)       | 3.496 $$\pm$$ 0.116 | 0.502 $$\pm$$ 0.008 | 3.727 $$\pm$$ 0.032 |
| [DictLearn](https://aclanthology.org/2021.deelio-1.1.pdf) | 3.387 $$\pm$$ 0.007 | 0.503 $$\pm$$ 0.002 | 3.708 $$\pm$$ 0.007 |
| [NMF](https://openaccess.thecvf.com/content/CVPR2023/papers/Fel_CRAFT_Concept_Recursive_Activation_FacTorization_for_Explainability_CVPR_2023_paper.pdf)       | 3.761 $$\pm$$ 0.050 | 0.542 $$\pm$$ 0.001 | 3.812 $$\pm$$ 0.063 |
| [CT](https://openreview.net/pdf?id=kAa9eDS0RdO)        | 4.931 $$\pm$$ 0.001 | 0.546 $$\pm$$ 0.000 | 4.348 $$\pm$$ 0.000 |
| Random    | 4.927 $$\pm$$ 0.001 | 0.546 $$\pm$$ 0.000 | 4.348 $$\pm$$ 0.000 |
| CCE       | **3.163 $$\pm$$ 0.000** | **0.459 $$\pm$$ 0.004** | **3.689 $$\pm$$ 0.002** | -->

<style>
    .tabitem {
        display: none;
    }
    .tabitem.active {
        display: block;
    }
    .tab-buttons {
        margin-bottom: 20px;
    }
    .tab-buttons button {
        padding: 10px 20px;
        margin-right: 10px;
    }
</style>
<ul class="tab">
    <li id="tab-clevr" class="active" onclick="showTab('clevr')"><a href="#">CLEVR</a></li>
    <li id="tab-cub-sub" class="" onclick="showTab('cub-sub')"><a href="#">CUB-sub</a></li>
    <li id="tab-truth-sub" class="" onclick="showTab('truth-sub')"><a href="#">Truth-sub</a></li>
</ul>
<div id="clevr" class="tabitem active">
    <canvas id="clevrChart"></canvas>
</div>
<div id="cub-sub" class="tabitem">
    <canvas id="cubSubChart"></canvas>
</div>
<div id="truth-sub" class="tabitem">
    <canvas id="truthSubChart"></canvas>
</div>

<script>
    function showTab(tabId) {
        var tabs = document.querySelectorAll('.tabitem');
        tabs.forEach(function(tab) {
            tab.classList.remove('active');
        });
        document.getElementById('tab-clevr').classList.remove('active');
        document.getElementById('tab-cub-sub').classList.remove('active');
        document.getElementById('tab-truth-sub').classList.remove('active');

        document.getElementById(tabId).classList.add('active');
        document.getElementById('tab-' + tabId).classList.add('active');
    }

    var clevrCtx = document.getElementById('clevrChart').getContext('2d');
    var clevrChart = new Chart(clevrCtx, {
        type: 'bar',
        data: {
            labels: ['GT', 'PCA', 'ACE', 'DictLearn', 'NMF', 'CT', 'Random', 'CCE'],
            datasets: [{
                label: 'CLEVR',
                data: [3.162, 3.684, 3.496, 3.387, 3.761, 4.931, 4.927, 3.163],
                backgroundColor: [
                    'rgba(75, 192, 192, 0.2)',
                    'rgba(54, 162, 235, 0.2)',
                    'rgba(255, 206, 86, 0.2)',
                    'rgba(255, 159, 64, 0.2)',
                    'rgba(153, 102, 255, 0.2)',
                    'rgba(255, 99, 132, 0.2)',
                    'rgba(75, 192, 192, 0.2)',
                    'rgba(54, 162, 235, 0.2)'
                ],
                borderColor: [
                    'rgba(75, 192, 192, 1)',
                    'rgba(54, 162, 235, 1)',
                    'rgba(255, 206, 86, 1)',
                    'rgba(255, 159, 64, 1)',
                    'rgba(153, 102, 255, 1)',
                    'rgba(255, 99, 132, 1)',
                    'rgba(75, 192, 192, 1)',
                    'rgba(54, 162, 235, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            plugins: {
              legend: {
                display: false
              }
            },
            scales: {
                y: {
                    beginAtZero: false
                }
            }
        }
    });

    var cubSubCtx = document.getElementById('cubSubChart').getContext('2d');
    var cubSubChart = new Chart(cubSubCtx, {
        type: 'bar',
        data: {
            labels: ['GT', 'PCA', 'ACE', 'DictLearn', 'NMF', 'CT', 'Random', 'CCE'],
            datasets: [{
                label: 'CUB-sub',
                data: [0.472, 0.481, 0.502, 0.503, 0.542, 0.546, 0.546, 0.459],
                backgroundColor: [
                    'rgba(75, 192, 192, 0.2)',
                    'rgba(54, 162, 235, 0.2)',
                    'rgba(255, 206, 86, 0.2)',
                    'rgba(255, 159, 64, 0.2)',
                    'rgba(153, 102, 255, 0.2)',
                    'rgba(255, 99, 132, 0.2)',
                    'rgba(75, 192, 192, 0.2)',
                    'rgba(54, 162, 235, 0.2)'
                ],
                borderColor: [
                    'rgba(75, 192, 192, 1)',
                    'rgba(54, 162, 235, 1)',
                    'rgba(255, 206, 86, 1)',
                    'rgba(255, 159, 64, 1)',
                    'rgba(153, 102, 255, 1)',
                    'rgba(255, 99, 132, 1)',
                    'rgba(75, 192, 192, 1)',
                    'rgba(54, 162, 235, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
          plugins: {
              legend: {
                display: false
              }
            },
            scales: {
                y: {
                    beginAtZero: false
                }
            }
        }
    });

    var truthSubCtx = document.getElementById('truthSubChart').getContext('2d');
    var truthSubChart = new Chart(truthSubCtx, {
        type: 'bar',
        data: {
            labels: ['GT', 'PCA', 'ACE', 'DictLearn', 'NMF', 'CT', 'Random', 'CCE'],
            datasets: [{
                label: 'Truth-sub',
                data: [3.743, 3.988, 3.727, 3.708, 3.812, 4.348, 4.348, 3.689],
                backgroundColor: [
                    'rgba(75, 192, 192, 0.2)',
                    'rgba(54, 162, 235, 0.2)',
                    'rgba(255, 206, 86, 0.2)',
                    'rgba(255, 159, 64, 0.2)',
                    'rgba(153, 102, 255, 0.2)',
                    'rgba(255, 99, 132, 0.2)',
                    'rgba(75, 192, 192, 0.2)',
                    'rgba(54, 162, 235, 0.2)'
                ],
                borderColor: [
                    'rgba(75, 192, 192, 1)',
                    'rgba(54, 162, 235, 1)',
                    'rgba(255, 206, 86, 1)',
                    'rgba(255, 159, 64, 1)',
                    'rgba(153, 102, 255, 1)',
                    'rgba(255, 99, 132, 1)',
                    'rgba(75, 192, 192, 1)',
                    'rgba(54, 162, 235, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
          plugins: {
              legend: {
                display: false
              }
            },
            scales: {
                y: {
                    beginAtZero: false
                }
            }
        }
    });
</script>


## CCE Concepts Improve Downstream Classification Accuracy

<!-- A primary use-case for concepts is for interpretable classification with [Posthoc Concept-Bottleneck Models (PCBMs)](https://openreview.net/pdf?id=nA5AZ8CEyow). For four datasets spanning image and text domains, we evaluate CCE concepts against baselines in terms of classification accuracy after training a PCBM on the extracted concepts. We show classification accuracy with increasing numbers of extracted concepts in the figure below, and we see that CCE always achieves the highest accuracy or near-highest accuracy. -->

Do the concepts discovered by CCE improve downstream classification accuracy compared to baseline methods? We find that CCE does improve accuracy, as shown below on the CUB dataset when using 100 concepts.

<figure>
<canvas id="cubChart" width="800" height="400"></canvas>
<figcaption>Classification accuracy of a <a href="https://openreview.net/pdf?id=nA5AZ8CEyow">PCBM</a> using the concepts discovered by various approaches on the CUB dataset using exactly 100 concepts. CCE improves accuracy. In our paper, we include results on three additional datasets accross varying numbers of concepts to show that CCE improves performance in many difference scenarios and domains.</figcaption>
</figure>
<script>
    const ctx = document.getElementById('cubChart').getContext('2d');
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['CT', 'PCA', 'ACE', 'DictLearn', 'NMF', 'CCE'],
            datasets: [{
                label: 'CUB Score',
                data: [65.60, 72.71, 74.99, 75.33, 75.81, 76.49],
                backgroundColor: 'rgba(54, 162, 235, 0.8)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1,
                errorBars: {
                    'CT': 0.12,
                    'PCA': 0.01,
                    'ACE': 0.06,
                    'DictLearn': 0.07,
                    'NMF': 0.11,
                    'CCE': 0.47
                }
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Downstream classification accuracy on CUB',
                    font: {
                        size: 18
                    }
                },
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                label += context.parsed.y.toFixed(2) + ' ± ' + context.dataset.errorBars[context.label];
                            }
                            return label;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    title: {
                        display: true,
                        text: 'Accuracy'
                    },
                    min: 60,
                    max: 80
                },
                x: {
                    title: {
                        display: true,
                        text: 'Method'
                    }
                }
            }
        },
        plugins: [{
            id: 'errorBars',
            afterDatasetsDraw(chart, args, plugins) {
                const {ctx, data, chartArea: {top, bottom, left, right}, scales: {x, y}} = chart;

                ctx.save();
                ctx.strokeStyle = 'black';
                ctx.lineWidth = 2;

                data.datasets[0].data.forEach((datapoint, index) => {
                    const xPos = x.getPixelForValue(index);
                    const yPos = y.getPixelForValue(datapoint);
                    const errorBar = data.datasets[0].errorBars[data.labels[index]];
                    const yPosUpper = y.getPixelForValue(datapoint + errorBar);
                    const yPosLower = y.getPixelForValue(datapoint - errorBar);

                    ctx.beginPath();
                    ctx.moveTo(xPos, yPosUpper);
                    ctx.lineTo(xPos, yPosLower);
                    ctx.stroke();

                    ctx.beginPath();
                    ctx.moveTo(xPos - 5, yPosUpper);
                    ctx.lineTo(xPos + 5, yPosUpper);
                    ctx.stroke();

                    ctx.beginPath();
                    ctx.moveTo(xPos - 5, yPosLower);
                    ctx.lineTo(xPos + 5, yPosLower);
                    ctx.stroke();
                });

                ctx.restore();
            }
        }]
    });
</script>

In the paper, we show that CCE also improves classification performance on three other datasets spanning vision and language.

## Conclusion

Compositionality is a desired property of concept representations as human-interpretable concepts are often compositional, but we show that existing concept learning methods do not always learn concept representations which compose through addition. After studying the representation of concepts in a synthetic setting we find two salient properties of compositional concept representations, and we propose a concept learning method, CCE, which leverages our insights to learn compositional concepts. CCE finds more compositional concepts than existing techniques, results in better downstream accuracy, and even discovers new compositional concepts as shown through our qualitative examples.

Check out the details in our paper [here](https://arxiv.org/abs/2406.18534)! Our code is available [here](https://github.com/adaminsky/compositional_concepts), and you can easily apply CCE to your own dataset or adapt our code to create new concept learning methods.
