# neuron network understanding and visualizing

## Introduction
this repo contain many fun topic about nn understanding and visualizing. the topic are as follows:
1. the visulization of what the filter are catching.
we can divided this topic into two situation.
* given no image, what the filter are trying to catch.

for the first layers, the filters channel is 3 (if the image is a common RGB image), so we can easily plot it. in early CNN model, the first layers filter size is relatively large (Alexnet 11x11). but later pople find that the stack of small filter can outpreform the large filter, so current CNN's filter size is commonly 3x3.

for the intermeidate layers, the filter is often (N x 3 x3), N is the number of filter often 128, 256, 512, 1024, since the N != 3, just simpley plot it as N gray image is not so nice, what we do is run gradient ascent on input image to max activate the filter to get insight into what the filter is trying to catching/

also we can run graient descent on class score to answer the quewstion: what is cat in network's mind?

* given an input image, which part of the image is max activating the filter or the class score?

the method we use is still gradient descent. 

in the process of using SGD to update the image, we use a lot of tricks, the tricks as listed as follows:
  1. total variance norm
  2. L x norm
  3. jitter
  4. blur the image


2. human attack

human attack's goal is we are trying to fool the neuron network, given an image of cat, what we do is using gradient descent on the image to make the nn think that: this image is a dog.

3. style transfer




## Result



## Refrence