# DBN (Deep Belief Network)

When trained on a set of exnamples without supervision, a DBN can learn to probabilistically reconstruct its inputs. The layers then act as feature detectors.After this learning step, a DBN can be further trained with supervision to perform classification.
DBNs can be viewed as a composition of simple, unsupervised networks such as restricted Boltzmann machines (RBMs) or autoencoders,where each sub-network's hidden layer serves as the visible layer for the next. An RBM is an undirected, generative energy-based model with a "visible" input layer and a hidden layer and connections between but not within layers. This composition leads to a fast, layer-by-layer unsupervised training procedure, where contrastive divergence is applied to each sub-network in turn, starting from the "lowest" pair of layers (the lowest visible layer is a training set).

## Training :- 

The training method for RBMs proposed by Geoffrey Hinton for use with training "Product of Expert" models is called contrastive divergence (CD). CD provides an approximation to the maximum likelihood method that would ideally be applied for learning the weights. In training a single RBM, weight updates are performed with gradient descent via the following equation:

	𝑤𝑖𝑗(t+1)=𝑤𝑖𝑗(𝑡)+𝜂 (𝛿log(𝑝(𝑣)) / 𝛿𝑤𝑖𝑗)

where , p(v) is the probability vector , which is given by p(v) = 1/ 𝑍 ∑ ℎ𝑒−𝐸(𝑣,ℎ)
Z is the partition function (used for normalizing) and 
E(v,h) is the energy function assigned to the state of the network. A lower energy indicates the network is in a more "desirable" configuration. The gradient 𝛿log(𝑝(𝑣)) / 𝛿𝑤𝑖𝑗  has the simple form ⟨𝑣𝑖ℎ𝑗⟩𝑑𝑎𝑡𝑎−⟨𝑣𝑖ℎ𝑗⟩𝑚𝑜𝑑𝑒𝑙 represent averages with respect to distribution p. The issue arises in sampling ⟨𝑣𝑖ℎ𝑗⟩𝑚𝑜𝑑𝑒𝑙 because this requires extended alternating Gibbs sampling. CD replaces this step by running alternating Gibbs sampling for n steps (values of n =1 performs well) .After n steps  ,he data are sampled and that sample is used in place of ⟨𝑣𝑖ℎ𝑗⟩𝑚𝑜𝑑𝑒𝑙 The CD procedure works as follows:
    • Initialize the visible units to a training vector.
    • Update the hidden units in parallel given the visible units:
 𝑝(ℎ𝑗=1∣𝑉)=𝜎(𝑏𝑗+∑𝑖 𝑣𝑖  𝑤𝑖𝑗) 
	𝜎 is the sigmoid function
	𝑏𝑗 is the bias of hj

    • Update the visible units in parallel given the hidden units 
        ◦ 𝑝(𝑣𝑗=1∣𝐻)=𝜎(𝑎𝑗+∑𝑖ℎ𝑖𝑤𝑖𝑗)
where aj is the bias of vi . 
        ◦ And This step is known as the reconstruction step.

    • Re-update the hidden units in parallel given the reconstructed visible units using the same equation as in step 2.
    • Perform the weight update
        ◦ Δ𝑤𝑖𝑗𝛼⟨𝑣𝑖ℎ𝑗⟩𝑑𝑎𝑡𝑎−⟨𝑣𝑖ℎ𝑗⟩𝑟𝑒𝑐𝑜𝑛𝑠𝑡𝑟𝑢𝑐𝑡𝑖𝑜𝑛


Once an RBM is trained, another RBM is "stacked" atop it, taking its input from the final trained layer. The new visible layer is initialized to a training vector, and values for the units in the already-trained layers are assigned using the current weights and biases. The new RBM is then trained with the procedure above. This whole process is repeated until the desired stopping criterion is met.