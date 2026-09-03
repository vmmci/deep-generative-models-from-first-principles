# 02 — f-Divergence: Comparing Probability Distributions

> **Course context:** IIT Madras — Deep Generative Models, W1_L3  
>
> **Focus:** Comparing the unknown data distribution with a generative model using divergence measures, with emphasis on the mathematical structure of f-divergences.

---

## 1. From Generative Modeling to Distribution Comparison

The previous formulation of the generative modeling problem begins with an unknown data-generating distribution

$$
X\sim P_X.
$$

We observe only a finite dataset

$$
\mathcal{D}=\{x_1,\ldots,x_n\},
$$

under the modeling assumption

$$
X_1,\ldots,X_n\overset{\mathrm{i.i.d.}}{\sim}P_X.
$$

We then introduce a parameterized model distribution

$$
P_\theta.
$$

For example, an implicit generative model may begin with a simple latent variable

$$
Z\sim P_Z
$$

and generate

$$
\widehat{X}=g_\theta(Z),
$$

which induces the model distribution

$$
\widehat{X}\sim P_\theta.
$$

The goal is to choose the parameters so that

$$
P_\theta\approx P_X.
$$

This immediately creates a mathematical question:

> What does it mean for two probability distributions to be close?

We therefore need a mathematical measure of discrepancy between

$$
P_X
$$

and

$$
P_\theta.
$$

---

## 2. The Abstract Optimization Problem

Let

$$
D(P_X\Vert P_\theta)
$$

represent some discrepancy between the true distribution and the model distribution.

The learning objective can then be written abstractly as

$$
\theta^\star\in\underset{\theta}{\mathrm{arg\,min}}\;D(P_X\Vert P_\theta).
$$

The ideal goal is

$$
P_{\theta^\star}\approx P_X.
$$

If the discrepancy satisfies

$$
D(P\Vert Q)\geq0
$$

and becomes zero when the distributions coincide, then minimizing it provides a natural statistical learning principle.

However, there is no single universal notion of discrepancy between probability distributions.

Different divergence measures emphasize different kinds of mismatch.

---

## 3. Why Compare Probability Ratios?

Assume for now that two distributions $P$ and $Q$ admit densities

$$
p(x)
$$

and

$$
q(x).
$$

At a particular point $x$, a natural local comparison is the density ratio

$$
r(x)=\frac{p(x)}{q(x)}.
$$

This ratio has a direct probabilistic interpretation.

If

$$
r(x)=1,
$$

then the two densities agree at $x$.

If

$$
r(x)>1,
$$

then $P$ assigns relatively more density to that region than $Q$.

If

$$
r(x)<1,
$$

then $Q$ assigns relatively more density to that region.

Therefore,

$$
\frac{p(x)}{q(x)}
$$

describes a local mismatch between the two probability laws.

A divergence must convert these local mismatches into a global measure.

---

## 4. From Local Ratios to a Global Discrepancy

Choose a function

$$
f:(0,\infty)\rightarrow\mathbb{R}.
$$

We use $f$ to transform the density ratio into a penalty:

$$
f\left(\frac{p(x)}{q(x)}\right).
$$

The role of $f$ is therefore to determine **how different probability-ratio mismatches are penalized**.

To summarize these penalties over the entire distribution, take an expectation under $Q$.

This leads to the f-divergence.

---

## 5. Definition of f-Divergence

In the density-based formulation,

$$
D_f(P\Vert Q)=\int_{\mathcal{X}}q(x)f\left(\frac{p(x)}{q(x)}\right)dx.
$$

Equivalently,

$$
D_f(P\Vert Q)=\mathbb{E}_{X\sim Q}\left[f\left(\frac{p(X)}{q(X)}\right)\right].
$$

This expectation form gives a useful interpretation:

> Sample a point according to $Q$, examine the local density ratio $p/q$, convert that ratio into a penalty using $f$, and average the penalty.

The f-divergence is therefore an **expected penalty of probability-ratio mismatch**.

---

## 6. A Mathematical Note on the Density Form

The expression

$$
\frac{p(x)}{q(x)}
$$

assumes that suitable densities exist with respect to a common reference measure.

A more general mathematical treatment uses probability measures and the Radon-Nikodym derivative.

When $P$ is absolutely continuous with respect to $Q$, the ratio is written abstractly as

$$
\frac{dP}{dQ}.
$$

Then the same idea becomes

$$
D_f(P\Vert Q)=\mathbb{E}_{X\sim Q}\left[f\left(\frac{dP}{dQ}(X)\right)\right].
$$

For the current discussion, the density notation is sufficient, but the measure-theoretic form clarifies that f-divergence is fundamentally a statement about probability measures rather than only about ordinary density functions.

---

## 7. What Properties Should the Function f Have?

A standard construction assumes that $f$ is convex and satisfies

$$
f(1)=0.
$$

These conditions are not arbitrary.

They provide the mathematical structure that allows the resulting quantity to behave like a divergence.

The condition

$$
f(1)=0
$$

has an immediate interpretation.

If

$$
P=Q,
$$

then

$$
\frac{p(x)}{q(x)}=1
$$

almost everywhere.

Therefore,

$$
f\left(\frac{p(x)}{q(x)}\right)=f(1)=0,
$$

which gives

$$
D_f(P\Vert Q)=0.
$$

---

# 8. What Does Convexity Mean?

Convexity is a property of the global shape of a function.

A function $f$ is convex if, for any $u_1,u_2$ in its domain and any

$$
\lambda\in[0,1],
$$

we have

$$
f\left(\lambda u_1+(1-\lambda)u_2\right)\leq\lambda f(u_1)+(1-\lambda)f(u_2).
$$

Geometrically, the straight line joining two points on the graph lies above the graph of a convex function.

A familiar example is

$$
f(u)=u^2.
$$

A convex function does **not** need to be smooth.

For example,

$$
f(u)=|u|
$$

is convex but is not differentiable at

$$
u=0.
$$

Similarly, a function can be smooth without being globally convex.

For example,

$$
f(u)=\sin u
$$

is smooth but is not convex over the whole real line.

Therefore,

> Convexity, differentiability, and smoothness are different mathematical properties.

---

## 9. Detecting Convexity with Derivatives

If $f$ is twice differentiable on an interval, a useful sufficient characterization is

$$
f''(u)\geq0.
$$

For example, consider

$$
f(u)=u^2.
$$

Then

$$
f''(u)=2>0,
$$

so the function is strictly convex.

In contrast,

$$
f(u)=-u^2
$$

has

$$
f''(u)=-2<0,
$$

so it is concave.

The second-derivative condition is useful when derivatives exist, but it is not the definition of convexity itself.

---

# 10. Why Convexity Matters for f-Divergence

The central reason convexity appears in the definition is **Jensen's inequality**.

For a convex function $f$,

$$
f\left(\mathbb{E}[Y]\right)\leq\mathbb{E}[f(Y)].
$$

Define the random variable

$$
R(X)=\frac{p(X)}{q(X)},
$$

where

$$
X\sim Q.
$$

Then

$$
D_f(P\Vert Q)=\mathbb{E}_Q[f(R)].
$$

Now compute the expectation of the density ratio:

$$
\mathbb{E}_Q[R]=\int q(x)\frac{p(x)}{q(x)}dx.
$$

The $q(x)$ terms cancel:

$$
\mathbb{E}_Q[R]=\int p(x)dx.
$$

Because $p$ is a probability density,

$$
\int p(x)dx=1.
$$

Therefore,

$$
\mathbb{E}_Q[R]=1.
$$

Applying Jensen's inequality gives

$$
\mathbb{E}_Q[f(R)]\geq f\left(\mathbb{E}_Q[R]\right).
$$

Hence,

$$
D_f(P\Vert Q)\geq f(1).
$$

Since

$$
f(1)=0,
$$

we obtain

$$
\boxed{D_f(P\Vert Q)\geq0}.
$$

This is the key reason convexity matters.

The logic is

$$
\boxed{\mathrm{Convexity}\Longrightarrow\mathrm{Jensen's\ inequality}\Longrightarrow D_f(P\Vert Q)\geq0}.
$$

Convexity is therefore not required because we want a smooth function for differentiation.

It is required because it gives the divergence important probabilistic and optimization properties.

---

## 11. When Does Zero Divergence Imply Equal Distributions?

We always have

$$
P=Q\Longrightarrow D_f(P\Vert Q)=0
$$

under the standard normalization

$$
f(1)=0.
$$

The reverse implication requires additional conditions.

For example, if $f$ is strictly convex at the relevant point, then under suitable regularity assumptions,

$$
D_f(P\Vert Q)=0
$$

implies

$$
P=Q.
$$

Therefore, the statement

$$
D_f(P\Vert Q)=0\iff P=Q
$$

should not be asserted for an arbitrary convex function without additional conditions.

---

# 12. f-Divergence Is a Family, Not a Single Divergence

The definition

$$
D_f(P\Vert Q)=\int q(x)f\left(\frac{p(x)}{q(x)}\right)dx
$$

does not define only one discrepancy.

Different choices of $f$ produce different divergence measures.

Symbolically,

$$
f\longrightarrow D_f.
$$

The choice of $f$ controls how different kinds of probability mismatch are weighted.

Therefore,

> Different choices of $f$ produce different notions of what it means for two probability distributions to be close.

---

# 13. Example 1: Forward KL Divergence

Choose

$$
f(u)=u\log u.
$$

Substituting into the f-divergence gives

$$
D_f(P\Vert Q)=\int q(x)\frac{p(x)}{q(x)}\log\left(\frac{p(x)}{q(x)}\right)dx.
$$

Canceling $q(x)$ gives

$$
D_f(P\Vert Q)=\int p(x)\log\left(\frac{p(x)}{q(x)}\right)dx.
$$

Therefore,

$$
\boxed{D_f(P\Vert Q)=D_{\mathrm{KL}}(P\Vert Q)}.
$$

The Kullback-Leibler divergence is therefore one special case of the general f-divergence family.

---

## 14. Why Is the KL Generating Function Convex?

For

$$
f(u)=u\log u,
$$

the first derivative is

$$
f'(u)=\log u+1.
$$

The second derivative is

$$
f''(u)=\frac{1}{u}.
$$

Since the domain is

$$
u>0,
$$

we have

$$
f''(u)>0.
$$

Thus,

$$
f(u)=u\log u
$$

is strictly convex on its domain.

Also,

$$
f(1)=1\log1=0.
$$

This makes it suitable as an f-divergence generating function.

---

# 15. KL Divergence and Information Theory

KL divergence has the expectation form

$$
D_{\mathrm{KL}}(P\Vert Q)=\mathbb{E}_{X\sim P}\left[\log\left(\frac{p(X)}{q(X)}\right)\right].
$$

It can also be written as

$$
D_{\mathrm{KL}}(P\Vert Q)=\mathbb{E}_P[\log p(X)]-\mathbb{E}_P[\log q(X)].
$$

For discrete distributions, this gives the relation

$$
D_{\mathrm{KL}}(P\Vert Q)=H(P,Q)-H(P),
$$

where

$$
H(P)=-\mathbb{E}_P[\log p(X)]
$$

is entropy, and

$$
H(P,Q)=-\mathbb{E}_P[\log q(X)]
$$

is cross-entropy.

This gives an information-theoretic interpretation:

> KL divergence measures the extra mismatch introduced when data generated according to $P$ are represented using the probability model $Q$.

---

# 16. Connection to Maximum Likelihood Estimation

In generative modeling, let

$$
P=P_X
$$

and

$$
Q=P_\theta.
$$

Then

$$
D_{\mathrm{KL}}(P_X\Vert P_\theta)=\mathbb{E}_{P_X}[\log p_X(X)]-\mathbb{E}_{P_X}[\log p_\theta(X)].
$$

The first term does not depend on the model parameters $\theta$.

Therefore,

$$
\underset{\theta}{\mathrm{arg\,min}}\;D_{\mathrm{KL}}(P_X\Vert P_\theta)
$$

is equivalent to

$$
\underset{\theta}{\mathrm{arg\,max}}\;\mathbb{E}_{P_X}[\log p_\theta(X)].
$$

However, $P_X$ is unknown.

We only have samples

$$
x_1,\ldots,x_n\sim P_X.
$$

The expectation can therefore be approximated empirically:

$$
\mathbb{E}_{P_X}[\log p_\theta(X)]\approx\frac{1}{n}\sum_{i=1}^{n}\log p_\theta(x_i).
$$

This leads to

$$
\theta^\star\approx\underset{\theta}{\mathrm{arg\,max}}\;\sum_{i=1}^{n}\log p_\theta(x_i).
$$

This is the maximum likelihood principle.

Therefore, under this formulation,

$$
\boxed{\mathrm{Maximum\ Likelihood}\Longleftrightarrow\mathrm{Forward\ KL\ Minimization}}.
$$

This provides a direct bridge between classical statistical estimation and modern generative modeling.

---

# 17. Example 2: Reverse KL Divergence

Choose

$$
f(u)=-\log u.
$$

Then

$$
D_f(P\Vert Q)=\int q(x)\left[-\log\left(\frac{p(x)}{q(x)}\right)\right]dx.
$$

Rearranging the logarithm,

$$
D_f(P\Vert Q)=\int q(x)\log\left(\frac{q(x)}{p(x)}\right)dx.
$$

Therefore,

$$
\boxed{D_f(P\Vert Q)=D_{\mathrm{KL}}(Q\Vert P)}.
$$

Thus, different choices of the same generating framework can reverse the direction of KL divergence.

---

## 18. KL Divergence Is Not Symmetric

In general,

$$
D_{\mathrm{KL}}(P\Vert Q)\neq D_{\mathrm{KL}}(Q\Vert P).
$$

Therefore, KL divergence is not a metric.

The order of the two distributions is not merely notational.

Changing the order changes the objective.

This is particularly important for multimodal probability distributions.

---

## 19. Forward KL and Reverse KL: Intuition

Suppose the true distribution contains several probability modes.

When minimizing

$$
D_{\mathrm{KL}}(P_X\Vert P_\theta),
$$

regions where

$$
p_X(x)>0
$$

but

$$
p_\theta(x)
$$

is extremely small can produce a large penalty.

This often creates a tendency to cover regions where the data distribution places probability mass.

For this reason, forward KL is commonly associated with **mode-covering behavior**.

In contrast, when minimizing

$$
D_{\mathrm{KL}}(P_\theta\Vert P_X),
$$

the model is strongly penalized for placing probability mass in regions where the true distribution assigns very little probability.

This can create a tendency to concentrate on a subset of high-probability regions.

For this reason, reverse KL is commonly associated with **mode-seeking behavior**.

These descriptions provide useful intuition, although actual behavior also depends on the model family, parameterization, and optimization procedure.

---

# 20. Example 3: Total Variation

Choose

$$
f(u)=\frac{1}{2}|u-1|.
$$

Then

$$
D_f(P\Vert Q)=\int q(x)\frac{1}{2}\left|\frac{p(x)}{q(x)}-1\right|dx.
$$

Simplifying,

$$
D_f(P\Vert Q)=\frac{1}{2}\int|p(x)-q(x)|dx.
$$

This is the Total Variation distance:

$$
\boxed{D_{\mathrm{TV}}(P,Q)=\frac{1}{2}\int|p(x)-q(x)|dx}.
$$

An important point is that the generating function

$$
f(u)=\frac{1}{2}|u-1|
$$

is convex but not differentiable at

$$
u=1.
$$

This demonstrates again that

> f-divergence requires convexity, not necessarily smoothness.

Total Variation also has an event-based probabilistic interpretation:

$$
D_{\mathrm{TV}}(P,Q)=\sup_A|P(A)-Q(A)|.
$$

It measures the largest disagreement between the two distributions over any measurable event $A$.

---

# 21. Example 4: Jensen-Shannon Divergence

Define the mixture distribution

$$
M=\frac{1}{2}(P+Q).
$$

The Jensen-Shannon divergence is

$$
D_{\mathrm{JS}}(P,Q)=\frac{1}{2}D_{\mathrm{KL}}(P\Vert M)+\frac{1}{2}D_{\mathrm{KL}}(Q\Vert M).
$$

Unlike KL divergence, Jensen-Shannon divergence is symmetric:

$$
D_{\mathrm{JS}}(P,Q)=D_{\mathrm{JS}}(Q,P).
$$

It is also bounded.

Jensen-Shannon divergence becomes particularly important later when studying the theoretical formulation of Generative Adversarial Networks.

---

# 22. Why Are There Many Different Divergences?

There is no unique mathematical definition of what it means for two probability distributions to be close.

Different divergences penalize different discrepancies differently.

The choice can affect:

- sensitivity to missing modes,
- sensitivity to probability mass placed in incorrect regions,
- optimization behavior,
- gradient quality,
- whether densities must be evaluated,
- whether only samples are required,
- symmetry,
- numerical stability.

Therefore,

$$
\boxed{\mathrm{Different\ divergence}\Longrightarrow\mathrm{different\ learning\ behavior}}.
$$

The choice of divergence is part of the modeling problem, not merely a change of notation.

---

# 23. The Four Questions That Arise

The generative modeling formulation now produces several important questions.

### Question 1: How can the divergence be computed?

We want to evaluate something like

$$
D_f(P_X\Vert P_\theta).
$$

But the true data distribution $P_X$ is unknown.

We only have observations sampled from it.

---

### Question 2: Which divergence should be chosen?

Different divergences have different mathematical and statistical properties.

Therefore, the choice of discrepancy can affect the behavior of the learned model.

---

### Question 3: How should the generator be chosen?

If

$$
\widehat{X}=g_\theta(Z),
$$

then the architecture of $g_\theta$ determines the family of transformations available to the model.

The architecture therefore determines, in part, which output distributions $P_\theta$ can be represented.

---

### Question 4: How can optimization be performed over many parameters?

Modern neural networks may contain millions or billions of parameters.

We therefore need computational methods capable of optimizing objectives such as

$$
\mathcal{L}(\theta).
$$

This leads to gradient-based optimization, automatic differentiation, forward propagation, and backpropagation.

---

# 24. The Central Difficulty: The True Density Is Unknown

Return to

$$
D_f(P_X\Vert P_\theta)=\int p_\theta(x)f\left(\frac{p_X(x)}{p_\theta(x)}\right)dx.
$$

To evaluate this expression directly, we would need the density ratio

$$
\frac{p_X(x)}{p_\theta(x)}.
$$

But

$$
p_X(x)
$$

is generally unknown.

We only possess samples

$$
x_1,\ldots,x_n\sim P_X.
$$

For implicit generative models, an additional difficulty may arise:

$$
p_\theta(x)
$$

may also be unavailable even though we can generate samples from $P_\theta$.

For example, we may know how to perform

$$
Z\sim P_Z
$$

followed by

$$
\widehat{X}=g_\theta(Z),
$$

without having a tractable expression for

$$
p_\theta(x).
$$

This creates a central problem:

> How can a divergence based on probability densities be used when the distributions are available primarily through samples?

This question motivates alternative estimators and variational formulations of distributional discrepancies.

---

# 25. Probability Theory, Statistics, and Deep Learning

This formulation reveals an important separation of roles.

### Probability Theory

Probability defines objects such as

$$
P_X,\qquad P_\theta,\qquad \mathbb{E},\qquad D_f.
$$

It tells us what random variables, distributions, expectations, and divergences mean.

### Statistics

Statistics addresses the fact that

$$
P_X
$$

is unknown and must be inferred from finite samples.

For example,

$$
\mathbb{E}_{P_X}[h(X)]
$$

may be approximated by

$$
\frac{1}{n}\sum_{i=1}^{n}h(x_i).
$$

### Neural Networks

Neural networks provide expressive parameterized functions such as

$$
g_\theta.
$$

They determine flexible model families that can represent complicated transformations.

### Optimization

Optimization learns the parameters

$$
\theta
$$

by minimizing or maximizing a statistical objective.

The complete picture is therefore

$$
\boxed{\mathrm{Probability}\longrightarrow\mathrm{Statistical\ Objective}\longrightarrow\mathrm{Neural\ Parameterization}\longrightarrow\mathrm{Optimization}}.
$$

---

# 26. Conceptual View of f-Divergence

The f-divergence construction can be understood through the following sequence.

Start with two probability distributions:

$$
P
\qquad\text{and}\qquad
Q.
$$

Compare them locally using

$$
\frac{p(x)}{q(x)}.
$$

Transform the local ratio using

$$
f\left(\frac{p(x)}{q(x)}\right).
$$

Average that penalty under $Q$:

$$
\mathbb{E}_Q\left[f\left(\frac{p(X)}{q(X)}\right)\right].
$$

This produces

$$
D_f(P\Vert Q).
$$

The choice of $f$ determines what type of mismatch is emphasized.

Therefore,

$$
\boxed{\mathrm{density\ ratio}\longrightarrow\mathrm{penalty}\longrightarrow\mathrm{expectation}\longrightarrow\mathrm{divergence}}.
$$

---

# 27. Key Takeaways

1. Generative modeling requires a mathematical notion of discrepancy between the true distribution $P_X$ and the model distribution $P_\theta$.

2. f-divergence defines a broad family of such discrepancies:

$$
D_f(P\Vert Q)=\mathbb{E}_Q\left[f\left(\frac{p(X)}{q(X)}\right)\right].
$$

3. The density ratio

$$
\frac{p(x)}{q(x)}
$$

measures local disagreement between two probability laws.

4. The function $f$ determines how that disagreement is penalized.

5. Convexity is not the same as smoothness or differentiability.

6. Convexity is important because Jensen's inequality gives

$$
D_f(P\Vert Q)\geq0.
$$

7. The normalization condition

$$
f(1)=0
$$

ensures that identical distributions have zero divergence.

8. KL divergence, reverse KL, Total Variation, and Jensen-Shannon divergence are related examples within this broader distribution-comparison framework.

9. Forward KL minimization is closely connected to maximum likelihood estimation.

10. Different divergences imply different notions of distributional similarity and can lead to different learning behavior.

11. The central practical difficulty is that the true density $p_X(x)$ is generally unknown.

12. This creates the next major problem in deep generative modeling: constructing objectives that can be optimized using finite samples and neural-network parameterizations.

---

# Final Perspective

The generative modeling problem can now be refined from

$$
P_\theta\approx P_X
$$

into the optimization problem

$$
\theta^\star\in\underset{\theta}{\mathrm{arg\,min}}\;D_f(P_X\Vert P_\theta).
$$

The f-divergence tells us mathematically how two probability distributions differ.

But defining the divergence is only the beginning.

The deeper computational question is:

> How can this distribution-level objective be turned into something that can actually be estimated from samples and optimized over neural-network parameters?

That question connects directly to the next computational foundations of the course: forward propagation, backpropagation, automatic differentiation, and gradient-based optimization.
