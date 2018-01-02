# Uncertainty in Deep learning

This post sheds light on uncertainty in deep learning models. We all realise that deep learning algorithms grow in popularity and use by the greater engineering community. Maybe a deep learning model recommended this post to you, or the spotify song you are listening on the background. Soon, deep learning models might rise in more sensitive domains. Autonomous driving, medical decision making or jurisdiction might adapt models too, but what about the uncertainties that the models introduce in such applications?

Training a neural net gets you a point estimate of the weights. Testing a neural net gets you one sample of a softmax distribution. Think about the knowledge being lost in those two steps. Moreover, [we know that these softmax outputs can be easily fooled](http://karpathy.github.io/2015/03/30/breaking-convnets/). An imperceptible change might change a classification from _schoolbus 0.95_ to _Ostrich 0.95_. This project will not focus on these adversarial examples, but their existence motivates us to consider a more elaborate view on neural networks.

This project will compare on three approaches to uncertainty in neural networks. I talked to various researchers over the past months. There is no conclusion for one approach for obtaining uncertainties. However, all researchers agreed these three approaches point in the right direction.

# Overview of bootstrapping, MCMC and variational inference

Our three approaches are bootstrapping, MCMC and variational inference. Before we dive into the details of each, this section will sketch an overarching structure in which to understand these approaches. 

The bootstrap follows from the assumption that _there is one correct parameter for our model and we estimate it from a random data source_. How can we use this assumption to compute the uncertainty in our parameter? We will subsample the training set many times and estimate one parameter from each. We are uncertain about our model, so we maintain this set of estimated parameters. At test-time, we average the outputs from the model with each parameter as our prediction. The variance in our outputs represents the uncertainty. [chapter 8 of elements of statistical learning explains the bootstrap in more detail](https://web.stanford.edu/~hastie/ElemStatLearn/)

Another way to view the learning proces is that _there is one dataset and we learn a distribution over our model parameters_. Bayes rule exemplifies this reasoning. We distill our knowledge of the world in the prior. In the likelihood we update the distribution according to the data. This gives rise to a posterior distribution over parameters. However, this distribution is intractable to compute. Therefore, we resort to two approximations for this process. 

  * __Monte Carlo sampling_ Rather than evaluating the distribution, we draw samples from it. Then we can evaluate any function of the distribution via these samples. 
  * __Variational inference_ Rather than evaluating the distribution, we find a close approximation to it. This approximation will have a form that we can easily perform calculations over.

Both approximations come with disadvantages. For Monte Carlo methods, our estimate may _vary_ from the expected value if we have few samples. More samples will reduce this variance. For variational inference we will exactly find the best approximation. However, we will not know if there is a _bias_ between our approximate distribution and the true distribution. In other words, Monte Carlo methods have variance, variational inference has bias. 

# Sampling, averages and uncertainty
All three approaches results in multiple samples of the parameter vector. Our interest lies in the output for a sample and its uncertainty. How do we get these quantities from the parameter samples?

Our model outputs a softmax distribution. Therefore, we take the average over all these softmax distributions.

<img alt="$\bar{f}(x) = \sum_{\theta_i \in \{\theta\}} softmax(x;\theta_i)$" src="https://rawgit.com/RobRomijnders/bayes_nn/master/svgs/83985905d03c0411ae529d5b616d253b.svg?invert_in_darkmode" align=middle width="221.861805pt" height="27.34248pt"/>

For many applications, we need a decision. This will be the bin with the highest softmax value, <img alt="$\delta(x) = \arg max \bar{f}(x)$" src="https://rawgit.com/RobRomijnders/bayes_nn/master/svgs/029b5a8c0794b3d7099af9ca56f70ca6.svg?invert_in_darkmode" align=middle width="142.385925pt" height="27.34248pt"/>

Our estimate of the uncertainty is less clear. We are working with softmax distribution which has no common uncertainty number associated. In the literature, I came across three options

  * **Softmax value**: in this case, the value of the softmax at the decision is used to represent uncertainty. so <img alt="$\gamma = max \bar{f}(x)$" src="https://rawgit.com/RobRomijnders/bayes_nn/master/svgs/a61892f5736d156c83867d6f40a15b39.svg?invert_in_darkmode" align=middle width="95.856585pt" height="27.34248pt"/>
  * **Variance in the softmax**: in this case, the variance of the softmax values in the different outputs is used to represent uncertainty. Define the set of all softmax values, <img alt="$S = \{softmax_j(x; \theta_i)| j=\delta(x), \theta_i \in {\theta} \}$" src="https://rawgit.com/RobRomijnders/bayes_nn/master/svgs/8c79c835ae04f9cc0145857f7c1b8ff0.svg?invert_in_darkmode" align=middle width="275.993355pt" height="24.6576pt"/>. Then the uncertainty is the variance in this set, <img alt="$\gamma = var(S)$" src="https://rawgit.com/RobRomijnders/bayes_nn/master/svgs/0fec8f99d852f36973058e9264821ee9.svg?invert_in_darkmode" align=middle width="80.274315pt" height="24.6576pt"/>. [Section 4 of this paper](https://arxiv.org/pdf/1511.02680.pdf)
  * **Entropy in the average softmax**: in this case, the entropy of the average distribution represents the uncertainty. So <img alt="$\gamma = \sum_{k=1}^K  \bar{f}_k(x) log(\bar{f}_k(x))$" src="https://rawgit.com/RobRomijnders/bayes_nn/master/svgs/13c457c985c15aa7dab630d5442019c0.svg?invert_in_darkmode" align=middle width="187.210155pt" height="32.25618pt"/>. [Section 5.3 of this paper](https://arxiv.org/pdf/1506.02142.pdf)

In this project, we implement all three of them.

# Details on the implementations

## Bootstrapping
In bootstrappping, we sample multiple datasets with replacement. The `Dataloader` object has a function `bootstrap_yourself` to resample the training set for a bootstrap. Then the model is trained `num_runs` times to obtain the set of parameters

## MCMC
We use Langevin dynamics to obtain samples from the posterior over parameters. This implementation exactly follows [this paper by Teh and Welling](https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf). After a `burn_in` period, it will save a parameter vector every `steps_per_epoch` steps.

## Variational Inference
Honestly, I currently lack some understanding of the variational approach. The implementation follows the papers [here](https://arxiv.org/abs/1506.02142) and [here](https://arxiv.org/abs/1703.04977). At the moment, I understand this literature as fundamental approach that leads to an intuitive implementation. We are all familiar with dropout and its dropping of weights in a neural network. We can interpret this as fitting a two spike distribution to the parameter posterior (per weight) while constraining one spike at zero. We obtain samples from this distribution by sampling from these spikes. That amounts to running the model many times with different dropout masks. 
_I hope to update this section if I gain more understanding. The researchers I chatted with on this project also pointed me to [this](https://arxiv.org/abs/1505.05424) paper_

# Experiment
So how to assess uncertainty in image classification? There is no uncalibrated measure of uncertainty for any image, as that would assume a model of the full (history of) the world. However, we can assess images for which we know that uncertainty increases. We take two approaches, injecting Gaussian noise or rotating the image. 

# Results

We experiment with different noise levels or angles of rotation and record the corresponding uncertainty metrics. At perturbation method, we take `num_experiments` experiments on `batch_size_test` images.

![risks_experiments](https://github.com/RobRomijnders/bayes_nn/blob/master/bayes_nn/im/risks.png?raw=true)

These diagrams plot the risk numbers against the experiment variable. For differen injected noise and different rotation angles, we see the entropy, mean and standard deviation of the softmax. You can make this diagram with `plot_risk.py`

We also want intuition for the mutilation and its effect on the uncertainty. Therefore, we made these GIFs where the mutilations increase. Red and green titles indicate incorrect/correct classifications. 

![noise_gif](https://github.com/RobRomijnders/bayes_nn/blob/master/bayes_nn/im/noise/uncertainty_noise.gif?raw=true)
![rotation_gif](https://github.com/RobRomijnders/bayes_nn/blob/master/bayes_nn/im/rotation/uncertainty_rotation.gif?raw=true)

## Observations

In these results, there are some interesting observations

  * When rotating the images, the error quickly shoots up. At 90 degrees rotation the model misclassifies 80% of the images. It's interesting to see how the uncertainty numbers behave under such large error.
  * The entropy of `mc_dropout` is larger than the other two MC types. In parallel, we notice that its mean softmax value is lower.
  * Even though the entropy and mean softmax of Bootstrapping and Langevin samples are comparable, the standard deviation is lower. 

# Discussion
At this point, we leave many open ends for this project. No researcher I contacted on this expressed a conclusion on the uncertainties in neural networks. Lots of research needs to be done in this area. I hope these diagrams give you a starting point to think about these questions too.

As always, I am curious to any comments and questions. Reach me at romijndersrob@gmail.com



# Further reading

  * The original paper to propose dropout variational functions to approximate the posterior [Dropout as a Bayesian approximation](https://arxiv.org/abs/1506.02142)
  * Outlining the difference between aleatoric and epistemic uncertainty [What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?](https://arxiv.org/abs/1703.04977)
  * Nice Reddit thread on discussing validity of dropout as an estimator of epistemic uncertainty [[D] What is the current state of dropout as Bayesian approximation?](https://www.reddit.com/r/MachineLearning/comments/7bm4b2/d_what_is_the_current_state_of_dropout_as/)
  * On Hamiltonian Monte Carlo [MCMC using Hamiltonian dynamics](https://arxiv.org/abs/1206.1901)
  * The original paper to outline the sampling procedure in Langevin dynamics [Bayesian Learning via Stochastic Gradient Langevin Dynamics](https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf)
  * [Chapter 8 from the Elements of statistical learning on Bootstrapping](https://web.stanford.edu/~hastie/ElemStatLearn/)
