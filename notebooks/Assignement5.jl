### A Pluto.jl notebook ###
# v0.19.36

using Markdown
using InteractiveUtils

# ╔═╡ 37bd1fe6-eb92-11ee-11b5-c79f0842f455
md"""
# Assignment 5

## Problem 1

$f(x_1, x_2) = (x_1 - 4)^2 + x_2^2$

Subject to

$\begin{align*}
- x_1 & \leq 0 \\
- x_2 & \leq 0 \\
x_1 + x_2 - 2 & \leq 0
\end{align*}$

![Immagine del set di vincoli](link-alla-tua-immagine)

$L(x_1, x_2, \lambda_1, \lambda_2, \lambda_3) = (x_1 - 4)^2 + x_2^2 + \lambda_1(-x_1) + \lambda_2(-x_2) + \lambda_3(x_1 + x_2 - 2)$

The function $f$ is convex, & the constraints too (see each $c_i$) and the set is bounded (see figure) $\Rightarrow$ unique solution

"""

# ╔═╡ ae8744a8-81c0-475a-81b5-7b5c39d44a5d
md"""
**KKT**

*Stationarity*

$\frac{\partial L}{\partial x_1} = 2(x_1 - 4) - \lambda_1 + \lambda_3 = 0$ 

$\frac{\partial L}{\partial x_2} = 2x_2 - \lambda_2 + \lambda_3 = 0$

*Primal feasibility*
$\begin{cases}
x_1 \leq 0 \\
x_2 \leq 0 \\
x_1 + x_2 - 2 \leq 0
\end{cases}$

*Dual feasibility*
$\begin{cases}
\lambda_1 \geq 0 \\
\lambda_2 \geq 0 \\
\lambda_3 \geq 0
\end{cases}$

*Complementarity* ("slackness")
$\begin{cases}
\lambda_1 x_1 = 0 \\
\lambda_2 x_2 = 0 \\
\lambda_3(x_1 + x_2 - 2) = 0
\end{cases}$ 

"""

# ╔═╡ c76da9b1-a57e-40ad-8a47-33d92b3522c5
md"""
### Discussione delle condizioni di ammissibilità

**Caso 1**: $\lambda_1 \neq 0, \lambda_2 = 0 = \lambda_3$

Solo il primo constraint è attivo, si risolve a

$f(x_1=0, x_2) = x_2^2 + 16$

che ha minimo in $x_2 = 0 \Rightarrow f = 16$

=> Anche il secondo vincolo è attivo.

**Caso 2**: $\lambda_1 = 0, \lambda_2 \neq 0, \lambda_3 = 0$

Solo il secondo vincolo è attivo, si risolve a

$f(x_1, x_2=0) = (x_1 - 4)^2$

che ha minimo in $x_1 = 4 \Rightarrow$ fuori da $C$!

=> dentro C il minimo è in $x_1 = 2 \Rightarrow f = 4$

=> anche il terzo vincolo è attivo

**Caso 3**: $\lambda_1 = 0, \lambda_2 = 0, \lambda_3 \neq 0$

$f\left(x_1, \frac{x_2}{x_1+x_2-2}=1\right) = (x_1-4)^2 + \left(\frac{x_2}{x_1+x_2-2}-x_1\right)^2$

$= 2x_1^2 + 20 - 8x_1 = 2(x_1^2 - 4x_1 + 4)$

min in $x_1 = 3 \Rightarrow$ outside $C$

"""

# ╔═╡ c3f81986-e050-4256-95be-ee07ce22d6f8
md"""
inside $C$ the min is in $x_1 = 2$

=> vincolo 2 attivo => $f = 4$

**Caso 4**: $\lambda_1 \neq 0, \lambda_2 \neq 0, \lambda_3 = 0$

Come caso 1, $x_1 = x_2 = 0$

**Caso 5**: $\lambda_1 \neq 0, \lambda_2 = 0, \lambda_3 \neq 0$

È il punto $x_1 = 0, x_2 = 2$

$f = 20$

**Caso 6**: $\lambda_1 = 0, \lambda_2 \neq 0, \lambda_3 \neq 0$

Come caso 3$^\circ$ escluso $C$

**Caso 7**: $\lambda_1 \neq 0, \lambda_2 \neq 0, \lambda_3 \neq 0$

Impossibile

**Caso 8**: $\lambda_1 = 0, \lambda_2 = 0, \lambda_3 = 0$

Nessun vincolo attivo, min in un certo $x$

(Nota: la parte di testo non matematica in alto a destra è parzialmente visibile e sembra essere un commento o una nota.)
min $f$ unconstrained $(x_1=4, x_2=0)$

inside $C \Rightarrow$ minimo è in

$x_2 = 0, x_1 = 2 \Rightarrow f = 4$

outside $C \Rightarrow$ torna a un caso precedente

constraint, torna a un caso precedente

**CONCLUSIONI**

Il minimo è in

$x_2 = 0, x_1 = 2 \Rightarrow f = 4$

"""

# ╔═╡ Cell order:
# ╠═37bd1fe6-eb92-11ee-11b5-c79f0842f455
# ╠═ae8744a8-81c0-475a-81b5-7b5c39d44a5d
# ╠═c76da9b1-a57e-40ad-8a47-33d92b3522c5
# ╠═c3f81986-e050-4256-95be-ee07ce22d6f8
