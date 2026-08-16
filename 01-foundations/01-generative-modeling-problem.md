# 01 — Generative Modeling: Problem Formulation

> **Course context:** IIT Madras — Deep Generative Models, W1_L2
> **Focus:** Mathematical formulation of the generative modeling problem, probabilistic assumptions, model distributions, and the role of neural function approximators.

---

## 1. Starting Point: Data as Samples from an Unknown Distribution

Suppose we observe a dataset

$$
\mathcal{D}={x_1,x_2,\ldots,x_n}.
$$

The lowercase symbols $x_1,\ldots,x_n$ denote the **observed data points**.

Behind these observations, we model corresponding random variables

$$
X_1,X_2,\ldots,X_n.
$$

A standard assumption in statistical learning is

$$
X_1,\ldots,X_n\overset{\mathrm{i.i.d.}}{\sim}P_X.
$$

Here, $P_X$ denotes the unknown **data-generating distribution**.

The important distinction is

$$
\mathcal{D}\neq P_X.
$$

The dataset is only a finite sample from $P_X$; it is not the distribution itself.

The true distribution $P_X$ is generally unknown.

This distinction is fundamental to generative modeling:

> We observe samples, but we want to learn something about the distribution that could have generated them.

---

## 2. Random Variables, Observations, and the Data Space

The symbol $X$ represents a random variable, while $x$ represents a particular realization of that random variable.

Conceptually,

$$
X\sim P_X
$$

and observing the random variable once gives a realization

$$
X=x.
$$

For modern machine learning, $X$ is often high-dimensional.

For example, an RGB image of resolution $64\times64$ has three color channels and can be represented as an element of

$$
\mathbb{R}^{64\times64\times3}.
$$

After flattening the image,

$$
X\in\mathbb{R}^{12288}.
$$

For raw 8-bit digital images, a more literal representation would be

$$
X\in{0,\ldots,255}^{12288}.
$$

After normalization, one may instead use

$$
X\in[0,1]^{12288}.
$$

Thus, even a relatively small image corresponds to a point in a very high-dimensional space.

The difficulty is not only that this space is large. The probability distribution over this space can also be extremely complicated.

---

## 3. What Does the i.i.d. Assumption Mean?

The notation

$$
X_1,\ldots,X_n\overset{\mathrm{i.i.d.}}{\sim}P_X
$$

contains two distinct assumptions:

1. **independence**, and
2. **identical distribution**.

### 3.1 Identically Distributed

Each random variable follows the same marginal distribution:

$$
X_i\sim P_X\qquad\text{for every }i.
$$

This does **not** mean

$$
X_i=X_j.
$$

Two observations may look completely different while still being samples from the same probability distribution.

For example, two images of entirely different objects can both be realizations of the same broad image distribution.

### 3.2 Independent

Independence means that the joint distribution factorizes.

In measure notation,

$$
P_{X_1,\ldots,X_n}=P_X^{\otimes n}.
$$

Informally, when probability mass functions or densities exist, this becomes

$$
p(x_1,\ldots,x_n)=\prod_{i=1}^{n}p_X(x_i).
$$

The assumption says that observing one sample does not change the probability law of another sample.

---

## 4. Why Make an Independence Assumption?

The i.i.d. assumption should not be interpreted as a statement that real-world observations are always genuinely independent.

It is a **modeling assumption**.

Real datasets may contain dependence because of:

* consecutive frames from the same video,
* repeated photographs of the same individual,
* multiple documents from the same source,
* temporal correlations,
* spatial correlations,
* duplicated or near-duplicated observations.

Nevertheless, independence assumptions make statistical analysis much more tractable.

One major consequence is that expectations under the unknown distribution can be approximated using sample averages.

For a suitable function $f$,

$$
\mathbb{E}*{X\sim P_X}[f(X)]\approx\frac{1}{n}\sum*{i=1}^{n}f(x_i).
$$

Under appropriate conditions, the Law of Large Numbers explains why this empirical average approaches the population expectation as the number of samples grows.

This will become extremely important later because many generative-model objectives contain expectations with respect to distributions that we cannot evaluate analytically.

---

## 5. The Empirical Distribution

A useful mathematical object connecting the finite dataset with the unknown population distribution is the **empirical distribution**

$$
\widehat{P}*n=\frac{1}{n}\sum*{i=1}^{n}\delta_{x_i},
$$

where $\delta_{x_i}$ denotes a point mass concentrated at the observation $x_i$.

The empirical distribution assigns probability $1/n$ to each observed data point.

For any suitable function $f$,

$$
\mathbb{E}_{X\sim\widehat{P}*n}[f(X)]=\frac{1}{n}\sum*{i=1}^{n}f(x_i).
$$

The empirical distribution is not the same object as the true distribution:

$$
\widehat{P}_n\neq P_X
$$

for a finite dataset in general.

However, it provides a sample-based approximation to $P_X$.

This is one of the fundamental bridges between probability theory and statistical learning.

---

## 6. Connection: Independence in Information Theory

The idea of independence appears far beyond generative modeling.

For independent random variables,

$$
p(x_1,\ldots,x_n)=\prod_{i=1}^{n}p(x_i).
$$

For independent discrete random variables, entropy becomes additive:

$$
H(X_1,\ldots,X_n)=\sum_{i=1}^{n}H(X_i).
$$

Memoryless sources in information theory are based on this type of factorization.

However, information theory does **not** require all random variables to be independent.

It also studies dependent systems through concepts such as:

* conditional entropy,
* mutual information,
* joint entropy,
* entropy rate,
* stochastic processes.

The broader lesson is that independence assumptions provide mathematical structure, but dependence itself is also something probability theory can model explicitly.

---

## 7. Connection: Conditional Independence in Hidden Markov Models

Hidden Markov Models provide another useful comparison.

Suppose the latent states are

$$
Z_1,Z_2,\ldots,Z_T
$$

and observations are

$$
X_1,X_2,\ldots,X_T.
$$

A first-order HMM assumes the Markov property

$$
P(Z_t\mid Z_1,\ldots,Z_{t-1})=P(Z_t\mid Z_{t-1}).
$$

Thus, once $Z_{t-1}$ is known, earlier latent states do not provide additional information about $Z_t$ under the model.

The emission assumption states that the current observation depends on the current hidden state:

$$
P(X_t\mid Z_1,\ldots,Z_T,X_1,\ldots,X_{t-1})=P(X_t\mid Z_t).
$$

Therefore, an HMM does not assume that everything is mutually independent.

Instead, it introduces a carefully designed structure of **conditional independence**.

This suggests a broader modeling principle:

> Complex probabilistic systems often become tractable by specifying which variables are allowed to depend on which other variables.

Independence is therefore not merely a simplifying trick; it is part of the structural language used to define probabilistic models.

---

## 8. Why Can Very Different Samples Come from One Distribution?

At first, the assumption

$$
X_i\sim P_X
$$

may appear strange for highly diverse datasets.

An image dataset may contain:

* people,
* animals,
* buildings,
* vehicles,
* landscapes,
* different colors,
* different lighting conditions,
* different viewing angles.

Why describe all of these using a single $P_X$?

The answer is that a probability distribution does not need to be simple or unimodal.

A single distribution may contain many regions of high probability, or **modes**.

One conceptual representation is a mixture:

$$
p_X(x)=\sum_{k=1}^{K}\pi_k,p(x\mid C=k),
$$

where

$$
\pi_k\geq0
$$

and

$$
\sum_{k=1}^{K}\pi_k=1.
$$

Different components may represent different subpopulations of the data.

This mixture representation is only one possible model; the real data distribution does not need to have this exact finite-mixture form.

The important point is:

> A single probability distribution can itself have extremely rich internal structure.

The challenge of modern generative modeling is precisely that $P_X$ may be an enormously complicated distribution over a high-dimensional space.

---

## 9. The Generative Modeling Problem

We observe

$$
\mathcal{D}={x_1,\ldots,x_n}
$$

with the modeling assumption

$$
X_1,\ldots,X_n\overset{\mathrm{i.i.d.}}{\sim}P_X.
$$

However, $P_X$ itself is unknown.

We therefore introduce a family of model distributions

$$
{P_\theta:\theta\in\Theta},
$$

where:

* $\Theta$ is the parameter space,
* $\theta$ denotes a particular parameter configuration,
* $P_\theta$ is the distribution induced by those parameters.

The learning goal is to choose a parameter value $\theta^\star$ such that

$$
P_{\theta^\star}\approx P_X.
$$

This is the central statistical viewpoint behind deep generative modeling.

---

## 10. Explicit and Implicit Generative Models

There is an important distinction between the **distribution** $P_\theta$ and a probability density or mass function $p_\theta(x)$.

Some models provide an explicit probability model for the observations.

For such models, quantities related to

$$
p_\theta(x)
$$

can be evaluated directly or through a tractable formulation.

These are commonly called **explicit generative models**.

Other models primarily define a procedure for sampling from $P_\theta$, while $p_\theta(x)$ itself may be unavailable or computationally intractable.

These are commonly called **implicit generative models**.

Therefore, the most general goal of generative modeling is not necessarily

> recover an explicit analytical formula for $p_X(x)$.

A more general statement is:

> Construct a generative mechanism whose induced distribution resembles the unknown data-generating distribution.

---

## 11. Latent Variables and Generative Mappings

A common approach begins with a simple latent random variable

$$
Z\sim P_Z.
$$

For example,

$$
Z\sim\mathcal{N}(0,I).
$$

We then introduce a parameterized function

$$
g_\theta:\mathcal{Z}\rightarrow\mathcal{X}
$$

and define the generated random variable

$$
\widehat{X}=g_\theta(Z).
$$

The function $g_\theta$ may be a neural network.

Notice an important distinction:

* $g_\theta$ may be deterministic,
* but $Z$ is random.

Therefore,

$$
\widehat{X}=g_\theta(Z)
$$

is itself a random variable.

Hence, $\widehat{X}$ has an induced probability distribution, which we denote by

$$
P_\theta.
$$

---

## 12. The Pushforward Distribution

The previous construction has a precise mathematical interpretation.

The model distribution is the **pushforward** of $P_Z$ through $g_\theta$:

$$
P_\theta=(g_\theta)_*P_Z.
$$

Here, $(g_\theta)_*$ denotes the pushforward operator.

For a measurable subset $A$ of the data space,

$$
P_\theta(A)=P_Z\bigl(g_\theta^{-1}(A)\bigr).
$$

The expression $g_\theta^{-1}(A)$ here denotes the **preimage** of the set $A$, not necessarily an inverse function.

Intuitively, to determine how much probability $P_\theta$ assigns to a region $A$ of the output space, we ask:

> How much latent probability is mapped by $g_\theta$ into that region?

Thus the generator transforms a relatively simple latent distribution into a potentially much more complicated output distribution:

$$
P_Z\longrightarrow P_\theta.
$$

This gives a precise probabilistic interpretation of a neural generator.

---

## 13. A Technical Distinction: Distribution vs. Density

It is mathematically safer to work with the distribution

$$
P_\theta
$$

rather than automatically assuming that a conventional density

$$
p_\theta(x)
$$

always exists.

Suppose

$$
Z\in\mathbb{R}^{100}
$$

while

$$
X\in\mathbb{R}^{12288}.
$$

Then a deterministic generator may have the form

$$
g_\theta:\mathbb{R}^{100}\rightarrow\mathbb{R}^{12288}.
$$

For sufficiently regular mappings, such as typical Lipschitz neural networks, the generated observations may occupy a much lower-dimensional subset of the ambient output space.

In such a case, the induced probability measure can be singular with respect to the ordinary Lebesgue measure on $\mathbb{R}^{12288}$.

Therefore, an ordinary density over the entire ambient space may not exist.

This distinction is particularly important when studying implicit generative models such as GANs.

---

## 14. Why Neural Networks Are Useful Here

The generative problem requires a family of functions expressive enough to represent extremely complicated transformations.

A deep neural network can be written conceptually as a composition

$$
f_\theta=f_L\circ f_{L-1}\circ\cdots\circ f_1.
$$

Its parameters $\theta$ control a very large family of possible functions.

Under standard assumptions, neural networks can approximate broad classes of functions to arbitrary accuracy when sufficient capacity is available.

This does not mean that every function is easy to learn, or that optimization will always find the desired approximation.

It means that neural networks provide a highly expressive **parameterization**.

Instead of manually deriving one enormous analytical expression for a complicated mapping, we optimize the parameters of

$$
f_\theta
$$

from data.

In generative modeling, the same idea allows

$$
g_\theta
$$

to transform a simple latent distribution into a highly structured model distribution.

---

## 15. Connection: Neural Networks as Function Approximators

The same perspective appears in scientific computing.

Suppose a physical system has a solution

$$
u(x,t)
$$

determined by a differential equation.

An analytical closed-form solution may be unavailable, and repeatedly running a numerical solver may be computationally expensive.

One possible strategy is to learn an approximation

$$
u_\theta(x,t)\approx u(x,t).
$$

After training, $u_\theta$ may serve as a **surrogate model**.

The neural network is not replacing the mathematics of the original problem.

Rather, it provides a parameterized approximation to a difficult function or solution map.

This can be useful when the original computation is:

* high-dimensional,
* repeatedly evaluated,
* computationally expensive,
* data-driven,
* or difficult to represent analytically.

Related ideas appear in:

* surrogate modeling,
* scientific machine learning,
* physics-informed neural networks,
* neural operators.

---

## 16. Differentiability and Smoothness

Neural networks do not automatically produce smooth functions.

For example,

$$
\mathrm{ReLU}(x)=\max(0,x)
$$

is continuous but is not differentiable at

$$
x=0.
$$

A feedforward ReLU network is generally piecewise linear.

If a problem requires higher-order derivatives, smoother activation functions may be more appropriate.

For example, $\tanh$ is differentiable to all orders.

This matters in applications involving differential equations because automatic differentiation can compute quantities such as

$$
\frac{\partial u_\theta}{\partial x}
$$

or

$$
\frac{\partial^2u_\theta}{\partial x^2}.
$$

Therefore, the advantage of neural parameterizations is not that they automatically make every problem smooth.

A more precise statement is:

> Neural networks allow us to choose flexible computational parameterizations whose differentiability properties can be matched to the problem.

---

## 17. Connection: Why Modern LLMs Became Powerful Even Though Neural Networks Already Existed
## 17. Connection: Why Modern LLMs Became Powerful Even Though Neural Networks Already Existed

Neural networks are not new.

Statistical language modeling is also not new.

The modern success of large language models therefore cannot be explained simply by saying that neural networks were invented.

Consider a token sequence

$$
x_1,x_2,\ldots,x_T.
$$

Probability theory gives the chain-rule factorization

$$
p(x_1,\ldots,x_T)=\prod_{t=1}^{T}p(x_t\mid x_1,\ldots,x_{t-1}).
$$

Using compact sequence notation, this can also be written as

$$
p(x_{1:T})=\prod_{t=1}^{T}p(x_t\mid x_{1:t-1}).
$$

Here, $x_{1:t-1}$ denotes all tokens that appear before position $t$.

This factorization itself is not a modern invention. It follows directly from the probability chain rule.

Earlier statistical language models also attempted to estimate these conditional probabilities, but they typically imposed a much shorter context.

For example, a trigram language model approximates

$$
p(x_t\mid x_1,\ldots,x_{t-1})\approx p(x_t\mid x_{t-2},x_{t-1}).
$$

Equivalently, using compact notation,

$$
p(x_t\mid x_{1:t-1})\approx p(x_t\mid x_{t-2},x_{t-1}).
$$

Thus, the probabilistic problem already existed:

> Given the previous tokens, estimate the probability distribution of the next token.

Modern neural language models address the same general probabilistic problem using much more expressive parameterized functions.

Instead of relying on short fixed-order statistics, a neural language model learns a conditional distribution of the form

$$
p_\theta(x_t\mid x_{1:t-1}).
$$

The parameter vector $\theta$ may contain millions or billions of learned parameters.

The dramatic improvement of modern LLMs came from the combination of several developments:

* better architectures,
* very large datasets,
* massive computational resources,
* improved optimization algorithms,
* scalable training infrastructure,
* effective self-supervised objectives,
* increasing model capacity.

The Transformer architecture was particularly important because it made large-scale parallel training practical while allowing the model to represent interactions across long contexts.

From a statistical perspective, however, the fundamental objective remains recognizable.

We observe sequences sampled from an unknown data distribution and construct a parameterized probability model intended to approximate that distribution:

$$
P_\theta\approx P_X.
$$

For autoregressive language modeling, the model distribution is constructed through conditional probabilities:

$$
p_\theta(x_{1:T})=\prod_{t=1}^{T}p_\theta(x_t\mid x_{1:t-1}).
$$

Therefore, one useful interpretation of modern LLM progress is:

> The underlying probabilistic formulation was already known. What changed dramatically was our ability to parameterize, optimize, and scale extremely complicated conditional probability models.

Neural networks supplied increasingly expressive function approximators, while advances in architecture, data, compute, and optimization made those approximations practically learnable at unprecedented scale.






--------------------------------------------------------------------------------

Neural networks are not new.

Statistical language modeling is also not new.

The modern success of large language models therefore cannot be explained simply by saying that neural networks were invented.

Consider a token sequence

$$
x_1,x_2,\ldots,x_T.
$$

Probability theory gives the chain-rule factorization

$$
p(x_1,\ldots,x_T)=\prod_{t=1}^{T}p(x_t\mid x_1,\ldots,x_{t-1}).
$$

Using the shorthand $x_{<t}$,

$$
p(x_{1:T})=\prod_{t=1}^{T}p(x_t\mid x_{<t}).
$$

This factorization itself is not a modern invention; it follows from the probability chain rule.

Earlier statistical language models approximated these conditionals using restricted context.

For example, a trigram model approximates

$$
p(x_t\mid x_{<t})\approx p(x_t\mid x_{t-2},x_{t-1}).
$$

Modern neural language models use highly expressive parameterized functions to estimate much richer conditional distributions.

The dramatic improvement of modern LLMs came from the combination of several developments:

* better architectures,
* very large datasets,
* massive computational resources,
* improved optimization,
* scalable training infrastructure,
* effective training objectives,
* increasing model capacity.

The Transformer architecture was especially important because it enabled highly parallel training while modeling long-range contextual interactions effectively.

Thus, one useful statistical interpretation is:

> The basic probabilistic problem did not fundamentally change. Our ability to parameterize and optimize extremely complicated probability models changed dramatically.




---

## 18. From Classical Parametric Statistics to Deep Models

A classical statistical model may assume a simple parametric family.

For example,

$$
X\sim\mathcal{N}(\mu,\sigma^2)
$$

with parameter vector

$$
\theta=(\mu,\sigma).
$$

Only a small number of parameters must be estimated.

A deep model may instead have

$$
\theta\in\mathbb{R}^{m}
$$

where $m$ may contain millions or billions of parameters.

Conceptually, however, the statistical structure remains related:

$$
\mathrm{data}\longrightarrow\mathrm{model\ family}\longrightarrow\mathrm{parameter\ estimation}\longrightarrow\mathrm{fitted\ model}.
$$

Deep learning does not eliminate statistical modeling.

Rather, it greatly expands the complexity of the parameterized function families that we can practically optimize.

---

## 19. Measuring the Difference Between Two Distributions

Once we have a model distribution $P_\theta$, we need some mathematical criterion for comparing it with the true distribution $P_X$.

Let

$$
\Delta(P_X,P_\theta)
$$

denote a generic discrepancy measure.

The learning problem can then be written abstractly as

$$
\theta^\star\in\underset{\theta\in\Theta}{\mathrm{arg,min}};\Delta(P_X,P_\theta).
$$

The goal is to obtain a model for which

$$
P_{\theta^\star}\approx P_X.
$$

The word **discrepancy** is intentionally general.

Not every useful comparison between probability distributions is a mathematical metric.

A metric $d$ must satisfy properties such as symmetry:

$$
d(P,Q)=d(Q,P),
$$

and the triangle inequality:

$$
d(P,R)\leq d(P,Q)+d(Q,R).
$$

Many important quantities used in generative modeling do not satisfy these properties.

---

## 20. Divergences Are Not Necessarily Distances

A major example is the Kullback-Leibler divergence.

In general,

$$
D_{\mathrm{KL}}(P\Vert Q)\neq D_{\mathrm{KL}}(Q\Vert P).
$$

Therefore, KL divergence is not symmetric.

It is consequently not a metric.

Nevertheless, it satisfies non-negativity:

$$
D_{\mathrm{KL}}(P\Vert Q)\geq0,
$$

and under the usual conditions,

$$
D_{\mathrm{KL}}(P\Vert Q)=0
$$

if and only if

$$
P=Q
$$

almost everywhere with respect to the relevant measure.

This illustrates why the term **divergence** is more appropriate than simply calling every distributional comparison a distance.

The direction of a divergence can also matter:

$$
D(P\Vert Q)
$$

need not behave like

$$
D(Q\Vert P).
$$

This distinction will become important when studying different generative objectives.

---

## 21. The Generic Recipe for Generative Modeling

The mathematical structure can now be summarized.

There exists an unknown data-generating distribution

$$
X\sim P_X.
$$

We observe samples

$$
X_1,\ldots,X_n\overset{\mathrm{i.i.d.}}{\sim}P_X.
$$

These produce the finite dataset

$$
\mathcal{D}={x_1,\ldots,x_n}.
$$

We choose a parameterized model family

$$
{P_\theta:\theta\in\Theta}.
$$

We define a suitable discrepancy between the data distribution and the model distribution:

$$
\Delta(P_X,P_\theta).
$$

Then we optimize

$$
\theta^\star\in\underset{\theta\in\Theta}{\mathrm{arg,min}};\Delta(P_X,P_\theta).
$$

The desired outcome is

$$
P_{\theta^\star}\approx P_X.
$$

If the model is based on a latent generator, we sample

$$
Z\sim P_Z
$$

and compute

$$
\widehat{X}=g_{\theta^\star}(Z).
$$

Therefore,

$$
\widehat{X}\sim P_{\theta^\star}.
$$

Only in an idealized case where

$$
P_{\theta^\star}=P_X
$$

would the generated distribution exactly equal the true data-generating distribution.

---

## 22. The Fundamental Difficulty

The previous formulation appears simple:

$$
\theta^\star\in\underset{\theta\in\Theta}{\mathrm{arg,min}};\Delta(P_X,P_\theta).
$$

But it immediately creates a fundamental problem.

The objective involves

$$
P_X,
$$

yet $P_X$ is precisely the distribution that we do **not** know.

We only possess samples

$$
x_1,\ldots,x_n.
$$

So how can an objective involving $P_X$ be evaluated or optimized?

This is one of the central questions of generative modeling.

In many cases, the solution is to transform a population-level objective into something that can be estimated from samples.

For example, expectations under $P_X$ may sometimes be approximated using

$$
\mathbb{E}*{X\sim P_X}[f(X)]\approx\frac{1}{n}\sum*{i=1}^{n}f(x_i).
$$

But different generative-model families solve the broader problem in different ways.

Important approaches include:

* maximum likelihood,
* variational inference,
* adversarial learning,
* score matching,
* diffusion objectives.

---

## 23. Where This Leads Next

The current lecture leaves us with the key question:

> How can we compare $P_\theta$ with an unknown $P_X$ using only samples from $P_X$?

This motivates the study of divergences between probability distributions.

The next step is to investigate **f-divergences**, which provide a general family containing several important divergence measures.

From there, the next mathematical problem becomes:

> Can a divergence be rewritten or estimated in a form that is usable when we only have samples?

This leads naturally to **variational representations of divergences** and eventually to several important generative-model training objectives.

---

## Key Takeaways

1. The observed dataset $\mathcal{D}$ is a finite sample, not the true distribution $P_X$.

2. The i.i.d. assumption provides a useful statistical structure, but it is a modeling assumption rather than a universal fact about real-world data.

3. Independence and conditional independence appear throughout probability, information theory, HMMs, and many other probabilistic models.

4. A single distribution $P_X$ can represent extremely heterogeneous and multimodal data.

5. Generative modeling introduces a parameterized family $P_\theta$ and attempts to find a member that approximates $P_X$.

6. A deterministic neural generator can still induce a probability distribution because its latent input is random.

7. Formally, a latent generator induces a pushforward distribution:

$$
P_\theta=(g_\theta)_*P_Z.
$$

8. Neural networks are useful because they provide expressive parameterized function families, not because they remove the underlying mathematical problem.

9. Modern LLMs build on older probabilistic ideas; their recent capabilities largely result from scalable parameterization, architecture, data, compute, and optimization.

10. The central unresolved problem is that $P_X$ is unknown, so generative learning objectives must ultimately be estimable from finite samples.

---

## Conceptual Summary

The overall problem can be viewed as

$$
\mathcal{D}\longrightarrow P_X\ \mathrm{unknown}\longrightarrow P_\theta\ \mathrm{parameterized}\longrightarrow\mathrm{optimization}\longrightarrow P_{\theta^\star}\approx P_X.
$$

Deep generative modeling combines two major ingredients:

**Probability theory**, which tells us what distributions, samples, expectations, and divergences mean;

and

**neural function approximation**, which gives us expressive parameterized mechanisms capable of representing extremely complicated transformations and model distributions.

The next question is no longer merely how to represent $P_\theta$.

It is how to **mathematically compare and optimize distributions when the true distribution is available only through samples**.
