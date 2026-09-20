# TensorTonic Solutions

Welcome to my TensorTonic solutions repository!

Here you'll find my solutions to various machine learning and deep learning problems from [TensorTonic](https://tensortonic.com).

## What is TensorTonic?

TensorTonic is a platform where you can implement core algorithms of Machine Learning from scratch.

This repository contains my personal solutions to these problems, automatically synchronized from the platform.

<!-- tensortonic:start -->
# Kenneth LIN's TensorTonic Solutions

Verified machine learning implementations completed on [TensorTonic](https://www.tensortonic.com).

<p align="center">
  <img src="https://www.tensortonic.com/api/badge/kennethlin.svg" alt="TensorTonic Verified Solutions" width="100%" />
</p>

| Problem | Description | Link |
|---|---|---|
| Implement AdamW (Decoupled Weight Decay) | Implement one AdamW optimizer step in NumPy with first and second moments plus decoupled weight decay. | https://www.tensortonic.com/problems/adamw-optimizer |
| Angle Between 3D Vectors | Compute the angle between two 3D vectors in NumPy with clamped cosine values and safe handling of zero norms. | https://www.tensortonic.com/problems/angle-between-3d |
| Bilinear Interpolation | Resize a 2D image with bilinear interpolation by combining the four neighboring pixel values at each output coordinate. | https://www.tensortonic.com/problems/bilinear-interpolation |
| Compute Confusion Matrix with Normalization | Build a multiclass confusion matrix and optionally normalize counts by true-class rows or predicted-class columns. | https://www.tensortonic.com/problems/confusion-matrix-norm |
| Data Drift Detection | Detect feature drift by computing total variation distance between reference and production histograms. | https://www.tensortonic.com/problems/data-drift-detection |
| Implement Dropout (Training Mode) | Implement training-mode dropout in NumPy with random masking and inverted scaling of retained activations. | https://www.tensortonic.com/problems/dropout-training |
| Calculate Eigenvalues of a Matrix | Calculate the eigenvalues of a square matrix and return them in the format required by the numerical contract. | https://www.tensortonic.com/problems/eigenvalues |
| Expected Value (Discrete Distribution) | Compute the expected value of a discrete distribution from matched outcomes and normalized probabilities. | https://www.tensortonic.com/problems/expected-value-discrete |
| Implement Focal Loss | Compute mean binary focal loss from predicted probabilities using a configurable focusing parameter. | https://www.tensortonic.com/problems/focal-loss |
| Implement GELU Activation (Gaussian Error Linear Unit) | Implement the Gaussian Error Linear Unit activation element-wise using the required GELU approximation. | https://www.tensortonic.com/problems/gelu |
| Implement Global Average Pooling | Apply global average pooling to spatial feature maps by averaging each channel across its height and width. | https://www.tensortonic.com/problems/global-avg-pooling |
| Hit Rate at K | Calculate recommendation hit rate at K by checking whether each user's relevant items appear in top-ranked results. | https://www.tensortonic.com/problems/hit-rate-at-k |
| KNN Distance + Neighbor Lookup | Find the nearest neighbors of a query point by computing and ordering Euclidean distances to training samples. | https://www.tensortonic.com/problems/knn-distance |
| Logistic Regression Training Loop | Train binary logistic regression in NumPy using sigmoid probabilities, gradient descent, and learned weight and bias parameters. | https://www.tensortonic.com/problems/logistic-regression-training |
| Make Diagonal Matrix | Construct a square diagonal matrix from a one-dimensional vector while setting every off-diagonal entry to zero. | https://www.tensortonic.com/problems/make-diagonal |
| Implement Manhattan Distance | Compute Manhattan distance between equal-length vectors by summing absolute coordinate differences. | https://www.tensortonic.com/problems/manhattan-distance |
| Monitoring Metrics Selection | Compute the required monitoring metrics for classification, regression, or ranking prediction results. | https://www.tensortonic.com/problems/monitoring-metrics-selection |
| Novelty Score | Measure recommendation novelty from item popularity by averaging self-information across recommended items. | https://www.tensortonic.com/problems/novelty-score |
| Pad Sequences | Pad or truncate variable-length token ID sequences in NumPy with configurable maximum length and padding values. | https://www.tensortonic.com/problems/pad-sequences |
| Poisson Probability Mass Function & Cumulative Distribution Function | Compute Poisson probability mass and cumulative probabilities for a nonnegative event count and rate. | https://www.tensortonic.com/problems/poisson-pmf-cdf |
| Policy Gradient Loss | Compute policy-gradient loss from selected action probabilities and advantage estimates with stable logarithms. | https://www.tensortonic.com/problems/policy-gradient-loss |
| RMSProp Optimizer (Single Update Step) | Implement one RMSProp update in NumPy using an exponential squared-gradient average and adaptive scaling. | https://www.tensortonic.com/problems/rmsprop-optimizer |
| ROI Pooling | Pool variable-size regions of interest into fixed spatial output grids using per-bin maximum values. | https://www.tensortonic.com/problems/roi-pooling |
| Rotate 3D Point Around Z-Axis | Rotate a 3D point around the z-axis by a given angle while preserving its z coordinate. | https://www.tensortonic.com/problems/rotate-around-z |
| Sample Variance & Standard Deviation | Compute sample variance and standard deviation with Bessel's correction from a numeric collection. | https://www.tensortonic.com/problems/sample-var-std |
| Implement Sigmoid in NumPy | Implement a vectorized sigmoid activation in NumPy for scalars, lists, vectors, and matrices, including large positive and negative inputs. | https://www.tensortonic.com/problems/sigmoid-numpy |
| One-Sample t-Test | Compute a one-sample t-statistic in NumPy using the sample mean, Bessel-corrected deviation, and hypothesized mean. | https://www.tensortonic.com/problems/t-test-one-sample |
| Implement TF-IDF Vectorizer | Build TF-IDF document vectors from token counts and inverse document frequency across a text corpus. | https://www.tensortonic.com/problems/tfidf-vectorizer |
| Detect Train-Serving Skew | Detect train-serving skew by comparing offline and online feature values under configured tolerances. | https://www.tensortonic.com/problems/train-serving-skew |
| Value Iteration Step | Perform one Bellman optimality update across states and actions for a tabular Markov decision process. | https://www.tensortonic.com/problems/value-iteration-step |
| Compute 3D Vector Norm | Compute the Euclidean norm of a 3D vector from the square root of summed squared coordinates. | https://www.tensortonic.com/problems/vector-norm-3d |
| WordPiece Tokenization | Implement BERT WordPiece tokenization with greedy longest-match subwords, continuation prefixes, and unknown-token fallback. | https://www.tensortonic.com/research/bert/bert-wordpiece |
| Bottleneck Block | Build a ResNet bottleneck block using 1x1 channel reduction, 3x3 convolution, and 1x1 channel expansion. | https://www.tensortonic.com/research/resnet/resnet-bottleneck |
| Convolutional Block | Implement a ResNet convolutional block with a projected shortcut that matches changed spatial and channel dimensions. | https://www.tensortonic.com/research/resnet/resnet-conv-block |
| Identity Block | Implement a ResNet identity block with a three-layer bottleneck branch, batch normalization, ReLU, and an unchanged skip path. | https://www.tensortonic.com/research/resnet/resnet-identity-block |
| Skip Connection Analysis | Analyze ResNet skip connections by combining residual and identity tensors and tracking gradient flow through the addition. | https://www.tensortonic.com/research/resnet/resnet-skip-connection |

View my verified ML profile: [TensorTonic profile](https://www.tensortonic.com/profile/kennethlin)
<!-- tensortonic:end -->
