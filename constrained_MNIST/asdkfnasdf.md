Shraya and I just finished up co-working, and we did some stuff
We used VS Code live share which is the best thing I have tried so far for pair programming (as opposed to jupyter cloud sharing, deepnote, etc.).

The main thing we experimented with was a threshold activation function and simplexes that sum to some *n* instead of 1. Threshold activation functions would potentially be a simplex equivalent of ReLU, where they set every value below a certain threshold to 0 (where threshold=0 would just be ReLU and the identity function in a simplex). Other intuitions for this include that it would increase sparsity even farther (i.e., collapsing into subspaces even more), which could improve interpretability. The central issue is probably that, by setting lots of activations to 0, we might get vanishing activations across many layers and hard gradients, in general.

We were unable to get any variants of the threshold to train very well. We tried
 - A threshold of 10 without anything else
 - ...combined with a power normalization (**1/2)
 - ...combined with a re-summing to 100 (increasing non-zero values to accomodate)
 - Making the threshold a learnable parameter
We did confirm that a threshold of 0 gave identical results to no activation function, but anything else essentially did not train past random-chance accuracy. Making the threshold a learnable parameter did work a little bit, but there is some funky stuff involved (we had to use a soft, sigmoid style cut-off during training for gradient purposes), and we didn't track the evolution of the threshold, so any accuracy that we got potentially came from the threshold simply being moved to 0.

We did re-confirm the viability of plain 100-sum-simplexes (~89% accuracy consistently; stochastic layers; no activations; input normalization). All of the above experiments were done with a simplex that sums to 100.