import numpy as np

# let's see what np.random.randn does ,suppose 
# hidden_layers = 100  ; visible_layers = 60
# W = np.random.randn(hidden_layers , visible_layers) / np.sqrt(visible_layers)

# print('W is {}'.format(W))
# print('Shape of W is {}'.format(W.shape))
# print("Square root of 60 is {} ".format(np.sqrt(visible_layers)))

# What happened here is random wights is initialized with size of hidden_layers as n 
# and visible_layers as m so W = n * m shape whose values are root_squared by n


# What does ceil does ??
# let's suppose we have a list of float values
# len_data = 1000 ; float_batch_size = 32
# ceil = np.ceil(len_data / float_batch_size)
# print(ceil)

# ceil rounds the float values in a positive value

# what does permutation does 
# let's see with an example

len_data= 1000
perm = np.random.permutation(len_data)
print(perm)
print(perm.shape)

# Permutation creates list of random and non-repeating numbers less 
# than the X of size X 

