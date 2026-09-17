# 03 — Variational Divergence Minimization

> **Course context:** IIT Madras — Deep Generative Models, W1_L4  
>
> **Focus:** Rewriting an f-divergence as a variational optimization problem that can be estimated from samples rather than direct access to unknown densities.

---

## 1. Starting Point

We observe

$$
\mathcal{D}=\{x_1,\ldots,x_n\},\qquad X_1,\ldots,X_n\overset{\mathrm{i.i.d.}}{\sim}P_X,
$$

where the true data-generating distribution $P_X$ is unknown.

A generative model defines a parameterized distribution $P_\theta$. For an implicit generator,

$$
Z\sim P_Z,\qquad \widehat{X}=g_\theta(Z),\qquad \widehat{X}\sim P_\theta.
$$

The abstract objective is

$$
\theta^\star\in\underset{\theta}{\mathrm{arg\,min}}\;D_f(P_X\Vert P_\theta).
$$

The difficulty is that we usually have **samples** from $P_X$ and $P_\theta$, but not direct analytical access to their densities.

---

## 2. Key Statistical Idea: Integrals as Expectations

Suppose

$$
I=\int_{\mathcal X}h(x)p_X(x)\,dx.
$$

This is simply

$$
I=\mathbb{E}_{X\sim P_X}[h(X)].
$$

Given i.i.d. samples $X_1,\ldots,X_n\sim P_X$, the expectation can be approximated by

$$
\widehat{I}_n=\frac{1}{n}\sum_{i=1}^{n}h(X_i).
$$

Under the usual integrability assumptions, the Law of Large Numbers gives

$$
\frac{1}{n}\sum_{i=1}^{n}h(X_i)\longrightarrow\mathbb{E}_{P_X}[h(X)].
$$

So an inaccessible population quantity can often become computable once it is written as an expectation under a distribution from which we can sample.

$$
\boxed{\mathrm{distribution}\longrightarrow\mathrm{expectation}\longrightarrow\mathrm{sample\ average}}
$$

---

## 3. Why Direct f-Divergence Is Still Difficult

Recall

$$
D_f(P_X\Vert P_\theta)=\int p_\theta(x)f\left(\frac{p_X(x)}{p_\theta(x)}\right)dx.
$$

Equivalently,

$$
D_f(P_X\Vert P_\theta)=\mathbb{E}_{P_\theta}\left[f\left(\frac{p_X(X)}{p_\theta(X)}\right)\right].
$$

The outer integral is an expectation, but the quantity inside still contains

$$
\frac{p_X(x)}{p_\theta(x)}.
$$

This density ratio may be unavailable even if we can draw samples from both distributions.

Therefore, the problem is not only the integral itself. The main obstacle is the **unknown density ratio**.

We need a reformulation that removes explicit dependence on that ratio.

---

## 4. Convex Conjugates

Let $f$ be a proper convex function. Its convex conjugate is

$$
f^\star(t)=\sup_u\{tu-f(u)\}.
$$

Fenchel-Young inequality states

$$
f(u)+f^\star(t)\geq tu.
$$

Therefore,

$$
f(u)\geq tu-f^\star(t).
$$

Under the standard closedness and regularity assumptions of convex duality,

$$
f(u)=\sup_t\{tu-f^\star(t)\}.
$$

This means that a convex function can be reconstructed as the supremum of affine lower bounds.

That fact is the mathematical tool that lets us decouple the nonlinear function $f$ from the density ratio.

---

## 5. Applying Convex Duality to the Density Ratio

Set

$$
u(x)=\frac{p_X(x)}{p_\theta(x)}.
$$

Then

$$
f\left(\frac{p_X(x)}{p_\theta(x)}\right)=\sup_t\left\{t\frac{p_X(x)}{p_\theta(x)}-f^\star(t)\right\}.
$$

Instead of using one scalar $t$ for every $x$, allow the optimizing value to depend on $x$ through a function

$$
T:\mathcal X\rightarrow\mathbb R.
$$

For any admissible $T$, Fenchel-Young gives

$$
f\left(\frac{p_X(x)}{p_\theta(x)}\right)\geq T(x)\frac{p_X(x)}{p_\theta(x)}-f^\star(T(x)).
$$

Multiplying by $p_\theta(x)$,

$$
p_\theta(x)f\left(\frac{p_X(x)}{p_\theta(x)}\right)\geq p_X(x)T(x)-p_\theta(x)f^\star(T(x)).
$$

Integrating,

$$
D_f(P_X\Vert P_\theta)\geq\int p_X(x)T(x)dx-\int p_\theta(x)f^\star(T(x))dx.
$$

Recognizing both terms as expectations,

$$
D_f(P_X\Vert P_\theta)\geq\mathbb{E}_{P_X}[T(X)]-\mathbb{E}_{P_\theta}[f^\star(T(X))].
$$

---

## 6. The Variational Lower Bound

Because the previous inequality holds for every admissible function $T$,

$$
D_f(P_X\Vert P_\theta)\geq\sup_{T\in\mathcal T}\left\{\mathbb{E}_{P_X}[T(X)]-\mathbb{E}_{P_\theta}[f^\star(T(X))]\right\}.
$$

This is the central variational form.

Instead of directly evaluating an expression that requires the unknown ratio $p_X/p_\theta$, we optimize over a function $T$.

With a sufficiently rich function space and the appropriate regularity assumptions, the bound becomes exact:

$$
D_f(P_X\Vert P_\theta)=\sup_T\left\{\mathbb{E}_{P_X}[T(X)]-\mathbb{E}_{P_\theta}[f^\star(T(X))]\right\}.
$$

If $T$ is restricted to a smaller function family, then the expression is generally only a lower bound.

This is why the formulation is called **variational**: the divergence is represented through optimization over functions.

---

## 7. The Optimal Variational Function

When $f$ is differentiable and the required regularity conditions hold, the optimal function satisfies

$$
T^\star(x)=f'\left(\frac{p_X(x)}{p_\theta(x)}\right).
$$

So the optimal $T$ implicitly contains information about the density ratio.

The important computational point is that we can try to **learn** this function without explicitly evaluating that ratio.

---

## 8. Why the Variational Form Is Sample-Computable

The reformulated objective contains

$$
\mathbb{E}_{P_X}[T(X)]
$$

and

$$
\mathbb{E}_{P_\theta}[f^\star(T(X))].
$$

Both can be approximated using samples.

For real observations

$$
x_1,\ldots,x_n\sim P_X,
$$

we estimate

$$
\mathbb{E}_{P_X}[T(X)]\approx\frac{1}{n}\sum_{i=1}^{n}T(x_i).
$$

For generated observations

$$
\widehat{x}_1,\ldots,\widehat{x}_m\sim P_\theta,
$$

we estimate

$$
\mathbb{E}_{P_\theta}[f^\star(T(X))]\approx\frac{1}{m}\sum_{j=1}^{m}f^\star(T(\widehat{x}_j)).
$$

Thus the inaccessible density-ratio problem becomes a sample-based optimization problem.

---

## 9. Parameterizing the Function T

Optimizing over all possible functions is not computationally feasible.

We therefore introduce a parameterized family

$$
T_\phi(x),
$$

where $\phi$ denotes its parameters.

A neural network is a natural choice because it provides a flexible function approximator.

Define

$$
\mathcal V(\phi,\theta)=\mathbb{E}_{P_X}[T_\phi(X)]-\mathbb{E}_{P_\theta}[f^\star(T_\phi(X))].
$$

Using samples,

$$
\widehat{\mathcal V}(\phi,\theta)=\frac{1}{n}\sum_{i=1}^{n}T_\phi(x_i)-\frac{1}{m}\sum_{j=1}^{m}f^\star(T_\phi(\widehat{x}_j)).
$$

For fixed generator parameters $\theta$, we maximize this objective with respect to $\phi$:

$$
\sup_\phi\widehat{\mathcal V}(\phi,\theta).
$$

The role of $T_\phi$ is to make the variational estimate as tight as possible.

---

## 10. Divergence Minimization Becomes a Min-Max Problem

The original objective is

$$
\underset{\theta}{\min}\;D_f(P_X\Vert P_\theta).
$$

Using the variational representation leads to the structure

$$
\underset{\theta}{\min}\;\underset{\phi}{\max}\;\left\{\mathbb{E}_{P_X}[T_\phi(X)]-\mathbb{E}_{P_\theta}[f^\star(T_\phi(X))]\right\}.
$$

The two parameter sets have different roles:

- $\phi$ tries to maximize the estimated discrepancy.
- $\theta$ tries to minimize it by moving $P_\theta$ closer to $P_X$.

This min-max structure is an important mathematical bridge to adversarial generative modeling.

---

## 11. Where Approximation Error Enters

In practice, there can be several gaps between the exact divergence and the quantity we optimize.

### Restricted Function Class

The exact optimizer $T^\star$ may not belong to the chosen neural-network family $T_\phi$.

### Finite Samples

Population expectations are replaced by empirical averages.

### Numerical Optimization

The optimizer may not reach the exact supremum over $\phi$ or minimum over $\theta$.

So theoretical representation, statistical estimation, and numerical optimization are separate layers of the problem.

---

## 12. Connection to Probability Theory

The entire construction depends on the identity

$$
\int h(x)p(x)dx=\mathbb{E}_{P}[h(X)].
$$

The Law of Large Numbers then gives

$$
\frac{1}{n}\sum_{i=1}^{n}h(X_i)\longrightarrow\mathbb{E}_{P}[h(X)].
$$

This is why expectations are so valuable in machine learning: even when the underlying distribution is unknown, expectations can often be estimated from samples.

The conceptual path is

$$
\boxed{\mathrm{unknown\ density}\longrightarrow\mathrm{expectation}\longrightarrow\mathrm{sample\ average}}.
$$

---

## 13. Connection to Convex Analysis

The second major ingredient is convex duality:

$$
f^\star(t)=\sup_u\{tu-f(u)\},
$$

and

$$
f(u)=\sup_t\{tu-f^\star(t)\}.
$$

This replaces a nonlinear function of the density ratio with an optimization problem over an auxiliary variable or function.

So the essential mathematical transformation is

$$
\boxed{\mathrm{nonlinear\ density\ ratio\ expression}\longrightarrow\mathrm{optimization\ over\ functions}}.
$$

---

## 14. Why Neural Networks Enter Naturally

The derivation introduces a function $T$ that must be optimized.

Probability theory defines the distributions and expectations.

Convex analysis gives the variational representation.

Neural networks provide a tractable parameterization $T_\phi$.

Gradient-based optimization then learns $\phi$ and $\theta$.

The roles can be summarized as

$$
\boxed{\mathrm{Probability}\longrightarrow\mathrm{expectations}}
$$

$$
\boxed{\mathrm{Convex\ analysis}\longrightarrow\mathrm{variational\ form}}
$$

$$
\boxed{\mathrm{Neural\ networks}\longrightarrow\mathrm{function\ parameterization}}
$$

$$
\boxed{\mathrm{Optimization}\longrightarrow\mathrm{learned\ parameters}}.
$$

---

## 15. Key Takeaways

1. The direct f-divergence may be unusable because it depends on the unknown ratio $p_X/p_\theta$.

2. Integrals against probability densities can be rewritten as expectations and estimated from samples.

3. Convex conjugacy gives

$$
f^\star(t)=\sup_u\{tu-f(u)\}
$$

and, under standard assumptions,

$$
f(u)=\sup_t\{tu-f^\star(t)\}.
$$

4. Fenchel-Young inequality produces a variational lower bound on the f-divergence.

5. The lower bound has the form

$$
\mathbb{E}_{P_X}[T(X)]-\mathbb{E}_{P_\theta}[f^\star(T(X))].
$$

6. Both expectations can be approximated using real and generated samples.

7. Parameterizing $T$ as $T_\phi$ converts function-space optimization into parameter optimization.

8. The resulting generative objective has a min-max structure:

$$
\underset{\theta}{\min}\;\underset{\phi}{\max}\;\mathcal V(\phi,\theta).
$$

9. Restricting $T$, using finite samples, and imperfect numerical optimization all create approximation gaps.

10. This derivation connects probability, statistics, convex analysis, neural function approximation, and adversarial generative learning.

---

## 16. Where This Leads Next

At this point the distribution-level objective has been converted into a parameterized optimization problem.

The next computational question is:

> How do we efficiently compute gradients through these parameterized functions and update millions of parameters?

That leads naturally to **forward propagation, backpropagation, automatic differentiation, and gradient-based optimization**.
