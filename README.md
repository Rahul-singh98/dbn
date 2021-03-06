# DBN (Deep Belief Network)

When trained on a set of exnamples without supervision, a DBN can learn to probabilistically reconstruct its inputs. The layers then act as feature detectors.After this learning step, a DBN can be further trained with supervision to perform classification.
DBNs can be viewed as a composition of simple, unsupervised networks such as restricted Boltzmann machines (RBMs) or autoencoders,where each sub-network's hidden layer serves as the visible layer for the next. An RBM is an undirected, generative energy-based model with a "visible" input layer and a hidden layer and connections between but not within layers. This composition leads to a fast, layer-by-layer unsupervised training procedure, where contrastive divergence is applied to each sub-network in turn, starting from the "lowest" pair of layers (the lowest visible layer is a training set).

## Training :- 

The training method for RBMs proposed by Geoffrey Hinton for use with training "Product of Expert" models is called contrastive divergence (CD).Β CD provides an approximation to the maximum likelihood method that would ideally be applied for learning the weights.Β In training a single RBM, weight updates are performed with gradient descent via the following equation:

	π€ππ(t+1)=π€ππ(π‘)+π (πΏlog(π(π£)) / πΏπ€ππ)

where , p(v) is the probability vector , which is given by p(v) = 1/ π β βπβπΈ(π£,β)
Z is the partition function (used for normalizing) and 
E(v,h) is the energy function assigned to the state of the network. A lower energy indicates the network is in a more "desirable" configuration. The gradient πΏlog(π(π£)) / πΏπ€ππ  has the simple form β¨π£πβπβ©πππ‘πββ¨π£πβπβ©πππππ represent averages with respect to distribution p. The issue arises in sampling β¨π£πβπβ©πππππ because this requires extended alternating Gibbs sampling. CD replaces this step by running alternating Gibbs sampling for n steps (values of n =1 performs well) .After n steps  ,he data are sampled and that sample is used in place of β¨π£πβπβ©πππππ The CD procedure works as follows:
    β’ Initialize the visible units to a training vector.
    β’ Update the hidden units in parallel given the visible units:
 π(βπ=1β£π)=π(ππ+βπ π£π  π€ππ) 
	π is the sigmoid function
	ππ is the bias of hj

    β’ Update the visible units in parallel given the hidden units 
        β¦ π(π£π=1β£π»)=π(ππ+βπβππ€ππ)
where aj is the bias of vi . 
        β¦ And This step is known as the reconstruction step.

    β’ Re-update the hidden units in parallel given the reconstructed visible units using the same equation as in step 2.
    β’ Perform the weight update
        β¦ Ξπ€πππΌβ¨π£πβπβ©πππ‘πββ¨π£πβπβ©ππππππ π‘ππ’ππ‘πππ


Once an RBM is trained, another RBM is "stacked" atop it, taking its input from the final trained layer. The new visible layer is initialized to a training vector, and values for the units in the already-trained layers are assigned using the current weights and biases. The new RBM is then trained with the procedure above. This whole process is repeated until the desired stopping criterion is met.