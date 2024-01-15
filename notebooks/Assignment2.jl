### A Pluto.jl notebook ###
# v0.19.35

using Markdown
using InteractiveUtils

# ╔═╡ 6476e2c3-4f65-4df3-84e3-a3f1aa2864f5
using LinearAlgebra, Plots, Statistics, Random, Polynomials

# ╔═╡ 72d1aade-524a-45f0-bdca-c5cd2456dfa5
md"""
## Problem 1
"""

# ╔═╡ 965e4ec5-d54c-4f2d-999b-4cee64672069
md"""
### Task 1
"""

# ╔═╡ 4352df2d-210e-436d-bc00-528767d8afb2
md"""

At each observation $(x_i, y_i)$, we define a residual, $y_i - f(x_i)$. Another possible formulation for our problem is to minimize:

$$R(\alpha_1, \ldots , \alpha_n) = \sum_{i=1}^{m} [y_i - f(x_i)]^2,$$

over all possible choices of parameters $\alpha_1, \ldots , \alpha_n$. We can rewrite the problem in the form $R = \mathbf{r}^T \mathbf{r}$, where

"""

# ╔═╡ 9b48e10a-c7c6-4e8d-968f-d6774416a942
md"""
$$\mathbf{r} = 
\begin{bmatrix}
y_1 \\
y_2 \\
\vdots \\
y_{m-1} \\
y_m 
\end{bmatrix}
-
\begin{bmatrix}
\phi_1(x_1) & \phi_2(x_1) & \ldots & \phi_n(x_1) \\
\phi_1(x_2) & \phi_2(x_2) & \ldots & \phi_n(x_2) \\
\vdots & \vdots & \ddots & \vdots \\
\phi_1(x_{m-1}) & \phi_2(x_{m-1}) & \ldots & \phi_n(x_{m-1}) \\
\phi_1(x_m) & \phi_2(x_m) & \ldots & \phi_n(x_m) \\
\end{bmatrix}
\begin{bmatrix}
\alpha_1 \\
\alpha_2 \\
\vdots \\
\alpha_n
\end{bmatrix}$$

"""

# ╔═╡ c378495c-3030-49f9-8031-486b4b4b4417
md"""
Recalling that $\mathbf{r}^T \mathbf{r} = \| \mathbf{r} \|^2$, and renaming the variables, we can express our problem as the linear least-squares problem:

$$\arg\min_{\mathbf{x} \in \mathbb{R}^n} \| \mathbf{b} - A\mathbf{x} \|_2$$

with
"""

# ╔═╡ b18a53e4-64ad-4edd-8f9b-83f92c79e35f
md"""

$$\mathbf{b} = \begin{bmatrix}
y_1 \\
y_2 \\
\vdots \\
y_{m-1} \\
y_m 
\end{bmatrix}\qquad A = \begin{bmatrix}
\phi_1(x_1) & \phi_2(x_1) & \ldots & \phi_n(x_1) \\
\phi_1(x_2) & \phi_2(x_2) & \ldots & \phi_n(x_2) \\
\vdots & \vdots & \ddots & \vdots \\
\phi_1(x_{m-1}) & \phi_2(x_{m-1}) & \ldots & \phi_n(x_{m-1}) \\
\phi_1(x_m) & \phi_2(x_m) & \ldots & \phi_n(x_m) \\
\end{bmatrix}$$

and $\mathbf{x}$ in the form requested.


"""

# ╔═╡ aaa78df6-9ebe-4867-b8fc-778d33e84606
md"""
### Task 2
"""

# ╔═╡ b847ef7a-171f-404a-8761-3eaaa0e5f697
md"""

Given the generic expression for $\phi_k(x)$, we can rewrite A as the *Vandermonde* matrix V:

$$V = \begin{bmatrix}
1 & (x_1) & \ldots & (x_1)^{n-1} \\
1 & (x_2) & \ldots & (x_2)^{n-1} \\
\vdots & \vdots & \ddots & \vdots \\
1 & (x_{m-1}) & \ldots & (x_{m-1})^{n-1} \\
1 & (x_m) & \ldots & (x_m)^{n-1} \\
\end{bmatrix}$$

We wish to prove that rank(V) = n if $x_i \neq x_j$ for $i \neq j$. The rank corresponds to the maximal number of linearly independent columns of V. 

Suppose that V is *not* of rank n. Then, the columns should not be linearly indipendent, that is, it would be possible to find some vector $\mathbf{c} = (c_1, \dots, c_n)^T: V\mathbf{c} = \mathbf{0}$. 

This expression would give rise to a polynomial $p(x) = c_1 + c_2 * x + \dots + c_n * x^{n-1}$. In order for $V\mathbf{c} = \mathbf{0}$ to be satisfied, this would mean that all the $x_i$ are roots (as they are distinct from each other). But the cardinality of ${x}$ is $m>n$, so a polynomial of degree n-1 would have more than n-1 roots, which is not possible. 

Therefore if $x_i \neq x_j$ for $i \neq j$, the matrix V must be of full rank, rank(V) = n. 
"""

# ╔═╡ 034e39df-dcbf-42ec-b84e-1e5dc3f43aae
md"""
### Task 3
"""

# ╔═╡ 5a7b400d-145b-4cd0-a1cd-067f4d95d072
md"""
Let's define the data for our problem as arrays, and compute the Vandermonde matrix:
"""

# ╔═╡ d024eff2-fbc0-403c-b950-fb9e9d7ed97d
begin
x = [8, 10, 12, 16, 20, 30, 40, 60, 100]
y = [0.88, 1.22, 1.64, 2.72, 3.96, 7.66, 11.96, 21.56, 43.16]
V = [ x[i]^j for i in 1:length(x), j in 0:2 ]  
#@show size(V)
end

# ╔═╡ 04c64c37-febc-414f-8b15-eb25ffdf459b
md"""
Here we define the function for computing a solution through the Cholesky decomposition:
"""

# ╔═╡ c9e70e07-eb26-4184-ac78-442966ea4744
md"""
First, we define the functions `forwardsub(L,b)` and `backsub(U,b)` to solve the lower triangular linear system with matrix `L` and right-hand side vector `b` and the upper triangular linear system with matrix `U` and right-hand side vector `b` respectively.
"""

# ╔═╡ 29fe2bda-d242-49ad-bcec-fdc6747fb25d
function forwardsub(L,b)
     n = size(L,1)
     x = zeros(n)
    x[1] = b[1]/L[1,1]
    for i in 2:n
        s = sum( L[i,j]*x[j] for j in 1:i-1 )
        x[i] = ( b[i] - s ) / L[i,i]
    end
    return x
end

# ╔═╡ 3c55be81-ff87-4a83-9807-0fd999d6da40
function backsub(U,b)
     n = size(U,1)
    x = zeros(n)
    x[n] = b[n]/U[n,n]
    for i in n-1:-1:1
		s = sum( U[i,j]*x[j] for j in i+1:n )
        x[i] = ( b[i] - s ) / U[i,i]
    end
	return x
end

# ╔═╡ d8ad5b17-8a8e-4cff-9c50-da4f98afb62f
md"""
Then we define `lsnormal(A,b)` to solve a linear least-squares problem by the normal equations. This function returns the minimizer of ||b-Ax||.
"""

# ╔═╡ 10e65287-1069-42d0-859a-6e6605226985
function lsnormal(A,b)
	# NOTE: We know that C is square and SPD!
    C = A'*A;  d = A'*b;
	# get cholesky decomposition
	# in upper triangular form
	# TODO: printout in format long
    R = cholesky(C).U 
	# solve (R^T)Rx=d
    w = forwardsub(R',d)           # solves R^T w = d              
    x = backsub(R,w)               # solves R x = w      
    return x
end

# ╔═╡ 9a9368c6-e7cb-46f7-82c5-26a08bd8a3c4
md"""
And now solve!
"""

# ╔═╡ 91b1bef0-faff-448a-bac6-dafa9dfc51ae
begin
	α_C = lsnormal(V, y)
	println(α_C)
	α_QR = V \ y
	println(α_QR)
	# compare the two solutions
	#  This corresponds to requiring equality of about half of the significant digits. 
    println("The two solutions are approximately equal: ", α_C ≈ α_QR)
	# compute the difference between the two solutions
    println("The difference between the two solutions is: ", norm(α_C - α_QR, 2))
end

# ╔═╡ 6591340b-c688-41c1-8d82-7c370fad3b4c
md"""
We plot the results below:
"""

# ╔═╡ d234abdf-a14a-449f-80d9-bf53f36264ff
begin
p_C = Polynomial(α_C)
p_QR = Polynomial(α_QR)

f_C = x -> p_C(x)
f_QR = x -> p_QR(x)
scatter(x,y,label="data",
    xlabel="x-var",ylabel="y-var",leg=:bottomright)
plot!(f_C,7,101,label="Cholesky Solution")
plot!(f_QR,7,101,label="QR Solution")

end

# ╔═╡ 0d2c2305-757c-49a8-ba6a-050462409f00
md"""
### Task 4

We compute the residuals for the two solutions. Notice that the resiudal of interest is $\mathbf{r} = \mathbf{d} - C*\alpha$. This residual measures the error in the solution of the system itself, rather than the fit of the model to the observed data (which would be $\mathbf{r}=\mathbf{y}−V\mathbf{x}$ in a regression context).
"""

# ╔═╡ 5bfc6d26-0da9-4e50-8f8d-0b2b42df72ba
begin
C = V'*V
d = V'*y
r_C = d - C*α_C
r_QR = d - C*α_QR
println("The norm of the residual for the Cholesky solution is: ", norm(r_C, 2))
println("The norm of the residual for the QR solution is: ", norm(r_QR, 2))
end

# ╔═╡ 8176c4f4-07bb-457b-9514-58227cf4221e
md"""
And now we compute the residuals for the approximate solution $\hat{\mathbf{\alpha}}$:
"""

# ╔═╡ 70be3242-51db-4566-a440-bed44ff4f101
begin
α_hat = [-1.919, 0.2782, 0.001739]
@show size(C)
r_hat = d - C*α_hat
println(r_hat)
println("The norm of the residual for the approximate solution is: ", norm(r_hat, 2))
end

# ╔═╡ ce2da4cf-5418-4043-887a-9dd5eebc388a
md"""
We can notice that the residual for this approximate solution has a much bigger norm compared to those of the solutions computed by Cholseky or QR Factorization. Why is this the case? Clearly, this solution is significantly less accurate than the one obtained via, e.g., QR decomposition. We can also use the following inequality to explain our findings:

$$\kappa^{-1} \frac{\| r \|_2}{\| d \|_2} \leq \frac{\| x - \hat{x} \|_2}{\| x \|_2} \leq \kappa \frac{\| r \|_2}{\| d \|_2}$$

Note that $\kappa$, the conditioning number of C, is particularly high. This is due to the ill-conditioned nature of V, which propagates to C as:

$$\kappa(V^T V) = \kappa(V)^2.$$

"""

# ╔═╡ 37453bfc-099d-4541-a76d-9d971231681c
begin
κ = cond(C)
κ_V = cond(V)
@show κ
@show κ_V, κ_V^2
println("κ approximately equal κ_V**2 with atol=1e-3: ", isapprox(κ, κ_V^2, atol=1e-3))

norm_r = norm(r_hat)
norm_d = norm(d)

LB = (norm_r/norm_d)/κ
@show LB
UB = κ*(norm_r/norm_d)
@show UB
end

# ╔═╡ 3e1b226d-cde2-4a16-b9fd-8b73e2086108
md"""
## Problem 2
"""

# ╔═╡ 839634e0-268a-41b2-afe1-290876cb4af5
md"""
### Task 1

Let $A = Q_1 R_1$ be the reduced QR factorization of A. 

Then, suppose that $R_1$ is singular: this would mean that $\exists \; \mathbf{v}\neq\mathbf{0}: R_1\mathbf{v}=\mathbf{0}$. But then  $A\mathbf{v} = Q_1 R_1\mathbf{v} = \mathbf{0}$, and this is not possible as A is of full rank. Thus, $R_1$ must be non-singular. 

The columns of $Q_1$ are orthonormal, by the property of the QR factorization, and they form a basis for the columns of $A$ . Since $A$ has full rank, the columns of $A$, and therefore the columns of $Q_1$, span Ran(A). Being orthonormal and spanning Ran(A), they form an orthonormal basis for Ran(A).

Null($A^T$) is the orthogonal complement of Ran($A$). Then, the extension of the orthonormal set $(\mathbf{q_1}, \dots, \mathbf{q_n})$ to a basis of $\mathbb{R}^m$, that is $(\mathbf{q_{n+1}}, \dots, \mathbf{q_m})$, is an orthonormal basis of Null($A^T$).
"""

# ╔═╡ a26fd903-7541-40d4-9e95-443a57fe5a0c
md"""
### Task 2

$$A^T = \begin{bmatrix}1.07 & 1.07 & 1.07 \\
1.10 & 1.11 & 1.15
\end{bmatrix}$$

and

$$A^T A = \begin{bmatrix}3.43 & 3.60 \\
3.60 & 3.76
\end{bmatrix}$$

We first notice that $A^T A$ is square and symmetric. To check if it is positive define, we need to check if the eignevalues are positive. The discriminant is $|A^T A| = 3.76*3.43 - 3.60*3.60 = -0.06$, so $A^T A$ cannot be positive definite.
"""

# ╔═╡ 40d89559-f9a0-46c2-8caf-6f3339929620
md"""
### Task 3
"""

# ╔═╡ 88a78727-afc2-4de9-8395-5f3deb16cfe2
md"""
We want to decompose:

$$A = \begin{bmatrix}
1.07 & 1.10 \\
1.07 & 1.11 \\
1.07 & 1.15
\end{bmatrix}$$

The steps to do it are as follows:

2. **Calculate the first Householder vector $z$:**

We know that $z$ is the first column of $A$:
$$z = \begin{bmatrix}
1.07 \\
1.07 \\
1.07
\end{bmatrix}$$
3. **Compute the first Householder reflector $v$:**

The first reflector $v$ is:
$$v = \frac{z - \| z \| \cdot e_1}{\| z - \| z \| \cdot e_1 \|} = \begin{bmatrix}
-0.46\\
0.628\\
0.628
\end{bmatrix}$$
4. **Construct the first Householder matrix $P_1$:**

The first Householder matrix $P_1$ is:
$$P_1 = I - 2vv^T =  \begin{bmatrix}
0.577 & 0.578 & 0.578 \\
0.578 & 0.211 & -0.789 \\
0.578 & -0.789 & 0.211
\end{bmatrix}$$
5. **Apply $P_1$ to $A$ to obtain $A_1$:**

The result is:
$$A_1 = P_1 A =  \begin{bmatrix}
1.85 & 1.94 \\
-5.02e-17 & -0.0373 \\
-7.6e-17 & 0.00266
\end{bmatrix}$$
6. **Repeat the process for the submatrix of $A_1$ to compute the second reflector:**

For the submatrix, we have:
$$z_1 = \begin{bmatrix}
-0.0373\\
0.00266
\end{bmatrix}, \quad v_1 = \begin{bmatrix}
-0.999\\
0.0356
\end{bmatrix}$$
7. **Construct the second Householder matrix $P_2$:**

The second Householder matrix $P_2$ is:
$$P_2 = \begin{bmatrix}
-1.0 & 0.0711\\
0.0711 & 0.997
\end{bmatrix}$$
8. **Form the full $Q_2$ matrix from $P_2$:**

The full $Q_2$ matrix is:
$$Q_2 = \begin{bmatrix}
1.0 & 0.0 & 0.0 \\
0.0 & -1.0 & 0.0711\\
0.0 & 0.0711 & 0.997
\end{bmatrix}$$
9. **Apply $Q_2$ to $A_1$ to obtain the upper triangular matrix $R$:**

The final upper triangular matrix $R$ is:
$$R = \begin{bmatrix}
1.85 & 1.94 \\
4.48e-17 & 0.0375 \\
-7.93e-17 & -1.0e-8
\end{bmatrix}$$

Remember that $Q = (Q_2 Q_1)^T = Q_1^T Q_2^T$, so:

$$Q = \begin{bmatrix}
0.577 & -0.537 & 0.617 \\
0.578 & -0.267 & -0.772 \\
0.578 & 0.804 & 0.154
\end{bmatrix}$$

When we multiply $Q$ by $R$ and round to two decimal places, we get:
$$Q R= \begin{bmatrix}
1.07 & 1.10 \\
1.07 & 1.11 \\
1.07 & 1.15
\end{bmatrix}$$

Which is precisely the original matrix $A$!
"""

# ╔═╡ 224fa724-812c-4c67-8eed-88ab20b880cd
md"""
### Task 4
"""

# ╔═╡ c12e7827-637c-42fe-a03e-965c65daf3ef
md"""
Given a reduced QR decomposition of a $m\times n$ real matrix $A=Q_1 R_1$, with $Q_1\in\mathbb{R}^{m\times n}$ and $R_1\in\mathbb{R}^{n\times n}$, we want to find the reduced QR decomposition of the $m\times (n+1)$ matrix

$A_+ = \begin{bmatrix}A & \mathbf{b}\end{bmatrix}$

in terms of the reduced QR decomposition of $A$.
"""

# ╔═╡ 08286a34-cf70-4ae5-862c-20cde4d03ea5
md"""
To do so, we start from the full QR decomposition of $A$

$A = QR = \begin{bmatrix} Q_1 & Q_2 \end{bmatrix} \begin{bmatrix} R_1 \\ 0 \end{bmatrix}$

Here $Q\in\mathbb{R}^{m\times m}$ and $R\in\mathbb{R}^{m\times m}$. The columns of $Q_1$ form an orthonormal basis of the span of $A$, while the columns of $Q_2$ form an orthonormal basis of the kernel of $A$.
"""

# ╔═╡ afb8c3f3-b05c-4c2b-a916-8cb94c302649
md"""
Now, let's consider the full QR decomposition of $A_+$ by applying consecutive Householder transformations: $H_1, H_2, \dots, H_n, H_{n+1}$. Since the first $n$ columns of $A_+$ are the same of $A$, the first $n$ Householder transformations are the same for the two matrices, hence

$H_n \cdots H_1 A_+ = Q^T A_+ = 
\begin{bmatrix}
R_1 & Q_1^T \mathbf{b} \\
0 & Q_2^T \mathbf{b}
\end{bmatrix}
\equiv 
\begin{bmatrix}
R_1 & \mathbf{z} \\
0 & \mathbf{a}
\end{bmatrix}$

Where $\mathbf{z}\in\mathbb{R}^n$ and $\mathbf{a}\in\mathbb{R}^{(m-n)}$.
"""

# ╔═╡ 19f13630-6fe8-4201-97ee-0b41d733a934
md"""
The last reflector is then of the form

$H_{n+1} = \begin{bmatrix}
I & 0 \\
0 & \hat{H}_{n+1}
\end{bmatrix}$

with

$\hat{H}_{n+1} = I - 2\mathbf{u}_{n+1}\mathbf{u}_{n+1}^T\in\mathbb{R}^{(m-n)\times(m-n)}$

and $\mathbf{u}_{n+1}$ being the usual vector of unitary norm constructed from $\mathbf{a}$ 

"""

# ╔═╡ 1cc06e5b-94bd-4faf-b082-6ce72c309803
md"""
With these premises, we are able to get the full size QR decomposition of $A_+$:

$H_{n+1} \cdots H_1 A_+ 
= H_{n+1} Q^T A_+
=
\begin{bmatrix}
Q_1^T \\
\hat{H}_{n+1}Q_2^T
\end{bmatrix} A_+
\equiv Q_+^T A_+ 
\equiv R_+
=
\begin{bmatrix}
R_1 & \mathbf{z} \\
0 & \rho \\
0 & 0
\end{bmatrix}$

In the reduced form then $Q_{1_+}$ is a $\mathbb{R}^{m\times(n+1)}$ matrix while $R_{1_+}\in\mathbb{R}^{(n+1)\times(n+1)}$ and they can be written as

$Q_{1_+}=\begin{bmatrix} Q_1^T & \mathbf{q}_{n+1}\end{bmatrix} 
\qquad 
R_{1_+}=\begin{bmatrix}
R_1 & \mathbf{z} \\
0 & \rho \\
\end{bmatrix}$

where $\mathbf{q}_{n+1}$ is the first column of $Q_2\hat{H}_{n+1}^T$ while $\rho=\text{sgn}(a_1) ||\mathbf{a}||_2$.
"""

# ╔═╡ 400e00db-717f-48d1-817b-ef4af51e56b9
md"""
Given the fact that an orthogonal matrix does not change the 2-norm of a vector, we can write

$||A\mathbf{x}-\mathbf{b}||_2^2=||Q^T(A\mathbf{x}-\mathbf{b})||_2^2=||R\mathbf{x} - \begin{bmatrix}Q_1^T & Q_2^T\end{bmatrix}\mathbf{b}||_2^2$

Since $R_1\mathbf{x}\in\text{rank}(A)$, $Q_1^T\mathbf{b}\in\text{rank}(A)$ and $Q_2^T\mathbf{b}\in\text{ker}(A)$ we can write

$||A\mathbf{x}-\mathbf{b}||_2^2=||R_1\mathbf{x} - Q_1^T\mathbf{b}||_2^2+||Q_2^T\mathbf{b}||_2^2=||R_1\mathbf{x} - \mathbf{z}||_2^2 + |\rho|^2$

Then, the solution of the least square problem $\min{||A\mathbf{x}-\mathbf{b}||_2}$ is $\mathbf{x}^*=R_1^{-1}\mathbf{z}$ and thus 

$||A\mathbf{x}^*-\mathbf{b}||_2=|\rho|$
"""

# ╔═╡ a99d636a-7180-46b1-8cbd-4fd01d96e9c6
md"""
## Problem 3
"""

# ╔═╡ 6864243d-bb5a-487a-8176-f5ce34bc9d29
md"""
### Task 1
"""

# ╔═╡ fa630a70-8b28-467e-afd5-30f2f4f8a9ef
md"""
If $A\in\mathbb{R}^{m\times n}$ is a matrix of rank $n$ with singular value decomposition $A=U\Sigma V^T$ then $\Sigma$ can be written as

$$\Sigma =
\begin{bmatrix}
\Sigma_r \\
0
\end{bmatrix}
\qquad\text{with}\qquad
\Sigma_r = 
\begin{bmatrix}
\sigma_1 &        &          \\
         & \ddots &          \\
         &        & \sigma_n
\end{bmatrix}$$

with $\sigma_1,\dots,\sigma_n$ singular values. This allows to write the following expressions in terms of the singular factors and vectors:
"""

# ╔═╡ fbad5eff-71e3-4364-8ada-5e9409dca5e4
md"""
1. $(A^T A)^{-1}$
Since $U$ is orthogonal, 
$A^T A = V\Sigma^T U^T \, U\Sigma V^T=V\Sigma^T\Sigma V^T$. 
We can further simplify the expression by rewriting the product $\Sigma^T\Sigma$ as

$\Sigma^T\Sigma \equiv \Sigma_r^2 = 
\begin{bmatrix}
\sigma_1^2 &        &            \\
           & \ddots &            \\
           &        & \sigma_n^2
\end{bmatrix}$

Then the singular value decomposition of $(A^T A)^{-1}$ is

$(A^T A)^{-1} = (V \Sigma_r^2 V^T)^{-1} = V (\Sigma_r^2)^{-1} V^T$

with

$(\Sigma_r^2)^{-1} = 
\begin{bmatrix}
1/\sigma_1^2 &        &             \\
             & \ddots &             \\
             &        & 1/\sigma_n^2
\end{bmatrix}\equiv\Xi$


The singular values are $1/\sigma_n^2, \dots, 1/\sigma_1^2$
"""

# ╔═╡ ef6a16e1-c896-40ca-87e0-878bfb5bccfa
md"""
2. $(A^T A)^{-1} A^T$
Starting from the previous results, we write

$(A^T A)^{-1} A^T = V\Xi V^T \, V \Sigma^T U^T = V\,\Xi \Sigma^T\, U^T$

The matrix $\Xi \Sigma^T\in\mathbb{R}^{n\times m}$ is

$\begin{bmatrix}
1/\sigma_1 &        &             &   \\
           & \ddots &             & 0 \\
           &        & 1/\sigma_n  &
\end{bmatrix}$


Then the singular values are $1/\sigma_n, \dots, 1/\sigma_1$
"""

# ╔═╡ ef2a1c8d-bfef-449d-a10e-c35c778d5aee
md"""
3. $A(A^T A)^{-1}$
Starting from the previous results, we write

$A(A^T A)^{-1} = U \Sigma V^T V\Xi V^T = U\, \Sigma \Xi \,V^T$

The matrix $\Sigma\Xi\in\mathbb{R}^{m\times n}$ is

$\begin{bmatrix}
1/\sigma_1 &        &             \\
           & \ddots &             \\
           &        & 1/\sigma_n  \\
           & 0      &
\end{bmatrix}$


Then the singular values are $1/\sigma_n, \dots, 1/\sigma_1$
"""

# ╔═╡ 99fcdeed-a3f4-4527-830c-c280ec662429
md"""
4. $A(A^T A)^{-1}A^T$
Starting from the previous results, we write:

$A(A^T A)^{-1}A^T = U \Sigma V^T V\Xi V^T V \Sigma^T U^T= U\, \Sigma \Xi \Sigma^T \,U^T$

The matrix $\Sigma \Xi \Sigma^T\in\mathbb{R}^{m\times m}$ is

$\begin{bmatrix}
I & 0 \\
0 & 0\end{bmatrix}$

with the identity block of size $n$. The singular values are then all equal to 1.
"""

# ╔═╡ 4ea3aae8-a54b-4d76-b920-610188b8f776
md"""
### Task 2
"""

# ╔═╡ 4a7b6aea-780b-4169-a67d-efd10eb98d76
md"""
To compute the singular values of the matrix

$A=
\begin{bmatrix}
1 & 2 \\
0 & 2
\end{bmatrix}$

we compute the eigenvalues of $AA^T$:

$\det (AA^T - \lambda I)=
\det\begin{bmatrix} 5 - \lambda & 4 \\ 4 & 4 - \lambda \end{bmatrix}=
\lambda^2 - 9\lambda + 4 = 0$

The solutions, expressed in base 10 with 4 digits precision, are 8.531 and 0.4689. The singular values of $A$ are the roots of the eigenvalues of $AA^T$, thus

$\sigma_1 = 2.921 \qquad \sigma_2 = 0.6847$

then the condition number is $\sigma_1/\sigma_2=4.266$. We can cross check our results using the functions of the Julia standard library:
"""

# ╔═╡ ced41e59-8e54-4d91-9c26-ef69853d8ca3
svdvals([1 2; 0 2])

# ╔═╡ 8bd80629-8a01-4ca1-b86b-af2e7084d47d
cond([1 2; 0 2])

# ╔═╡ 1f6382f4-0ef1-4180-b62f-af261fc5b005
md"""
Here we plot in the 2d plane the effect of the matrix $A$ applied to the unit ball. The projection is an ellypse where the principal axes are approximately 2.9 and 0.68 times the diameter of the unit ball.

This makes sense when we consider the singular value decomposition of $A=U\Sigma V^T$.
When applied to a $\mathbb{R}^2$ vector, the orthogonal matrix $V^T$ is a rotation, then the diagonal matrix $\Sigma$ scales the rotated vector and $U$ applies another rotation.
"""

# ╔═╡ d7d6bbb6-f8ea-4e71-b24d-ec6c75230f99
begin
	θ = 2 * π * rand(1000)
	ball = [cos.(θ) sin.(θ)]'
	proj = [1 2; 0 2] * ball
	projplot = scatter(ball[1,:], ball[2,:], label = "unit ball")
	scatter!(projplot, proj[1,:], proj[2,:], label = "projection")
	plot(projplot, aspect_ratio=:equal)
end

# ╔═╡ abedbe7b-9ec8-44fb-9303-a4e9f862533a
md"""
Given the fact that $V^T$ has no graphical effect since it is applied to a circle, the tilt of the ellypse is given only by $U$. We can use this information to plot the principal axes on the plot and verify that our intuitive explanation is in fact correct.
"""

# ╔═╡ 3dd200b9-8c0c-40fc-b871-0137b5cf315e
begin
	U, Σ, _ = svd([1 2; 0 2])
	pa1 = Σ[1] * U * [1 -1; 0 0]
	pa2 = Σ[2] * U * [0 0; 1 -1]
	paplot = scatter(proj[1,:], proj[2,:], label="data")
	plot!(paplot, pa1[1,:], pa1[2,:], label="principal axis σ 1") 
	plot!(paplot, pa2[1,:], pa2[2,:], label="principal axis σ 2") 
	plot(paplot, aspect_ratio=:equal)
end

# ╔═╡ 9da832c7-0522-48e2-9129-f6e1cbd2f3fa
md"""
### Task 3
"""

# ╔═╡ 6d3581b1-6499-4010-8019-ee796643b81e
md"""
The minimum norm solution $\hat{\mathbf{x}}$ for the problem $||\mathbf{b} - A \mathbf{x}||_2 = \min$, where

$A = \begin{bmatrix} 1 & 1 & 0 \\ 0 & 1 & 1 \end{bmatrix}
\qquad \mathbf{b} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$

is given by 

$\hat{\mathbf{x}}=A^+\mathbf{b}$

where $A^+$ is the Moore--Penrose pseudoinverse. Given the SVD of $A=U\Sigma V^T$, the Moore--Penrose pseudoinverse is given by $V\Sigma^+ U^T$, where $\Sigma^+$ is given by replacing every nonzero diagonal element of $\Sigma$ by its reciprocal.

Thus we can use the Julia standard library functions to find the minimum norm solution of our problem:
"""

# ╔═╡ 30d17722-7de3-4c20-bc0a-e35d76e4f180
begin
	# FROM THE svd() FUNCTION DOCUMENTATION:
	#
	# U, S, V and Vt can be obtained from the factorization
	# F with F.U, F.S, F.V and F.Vt, such that A = U * Diagonal(S) * Vt.
	#
	# If full = false (default), a "thin" SVD is returned.
	# For an M \times N matrix A, in the full factorization
	# U is M \times M and V is N \times N, while in
    # the thin factorization U is M \times K and V
	# is N \times K, where K = \min(M,N) is the number of singular values.
	D = svd([1 1 0; 0 1 1])
	x̂ =  D.V * Diagonal(1 ./ D.S) * D.U' * [1, 2]
end

# ╔═╡ 403e386e-37ac-4577-a129-643d0af87ebf
md"""
### Task 4
"""

# ╔═╡ 3f60adf8-bcf9-4b8b-afb3-99bae402745e
md"""
Using the features of the Julia `LinearAlgebra` standard library, we compute the SVD decomposition, the condition number, the rank and the Moore--Penrose pseudoinverse of the following matrix:
"""

# ╔═╡ 09679dad-aad2-4d5b-9df9-bbf90e6238d7
B = [
	-2    -4   -2   -4;
	2     -2    2    1;
	-800 200 -800 -401
]

# ╔═╡ d793814f-110e-4148-b56b-4e8e6a946e98
md"""
The SVD decomposition is:
"""

# ╔═╡ 3239de2d-54ef-4798-8387-da3345bca6c0
begin
	F = svd(B)
	svd(B, full=true)
end

# ╔═╡ 93c94d3a-9791-4d0f-a6f1-2af12f20fd22
md"""
There are three nonzero singular values, hence the matrix is of rank 3, as we can cross-check:
"""

# ╔═╡ ddfb314c-0c57-4a41-bb22-80d3cd36b827
rank(B) == 3

# ╔═╡ 57411892-53dd-4245-bdfc-ee0ad2f6fda2
md"""
The Moore--Penrose pseudoinverse:
"""

# ╔═╡ df5c4753-b803-4fd1-a8a4-003a3a5b7840
B⁺ = F.V * Diagonal(1 ./ F.S) * F.U'

# ╔═╡ 5e2c2fc5-559d-4b8e-9040-f9a3abb639b5
md"""
As we can cross-check:
"""

# ╔═╡ 21cde6be-16f3-4e77-aa07-9cb50cf646a5
pinv(B) ≈ B⁺

# ╔═╡ 4a699e21-0666-49a1-b6b8-ccb73142e3d0
md"""
The condition number, defined as the ratio of the greatest and smallest singular value, is:
"""

# ╔═╡ 99b6d0f6-becc-4de1-acc5-29f0629da5be
κ_B = F.S[1] / F.S[end]

# ╔═╡ 81fd82fa-8a10-4638-8486-3a8087d44a2b
md"""
As we can cross check:
"""

# ╔═╡ ae183c96-0c59-44a9-b3af-d1f1576a0b4c
cond(B) ≈ κ_B

# ╔═╡ 7bc9df74-f24f-4d96-a407-07b3eac37c35
md"""
### Task 5
"""

# ╔═╡ b260470f-d366-405b-b71f-e9b3044ffc5b
md"""
We want to find the best rank 1 and rank 2 approximation of `B`. To do so, we use the singular value decomposition. In fact, we know that if $U\Sigma V^T$ is the SVD decomposition of a matrix $B$, then $U \Sigma_k V^T$ (with $\Sigma_k$ being the rectangular matrix were only the first k singular values are taken) is the best rank $k$ approximation of $B$.

1. La sintassi può essere migliorata-chatty
"""

# ╔═╡ d7712c3f-81c9-4151-8c25-298eaade303a
md"""
Thus, the best rank 1 approximation is:
"""

# ╔═╡ d01412c9-7671-4049-b727-e8239b8aaafa
B_1 = F.U * Diagonal([F.S[1], 0, 0]) * F.Vt

# ╔═╡ ec66c587-f119-4817-8fc5-8f824af40bfe
md"""
The best rank 2 approximation is:
"""

# ╔═╡ f45f4d67-2191-4f61-945e-b7a6684d5b01
B_2 = F.U * Diagonal([F.S[1], F.S[2],0]) * F.Vt

# ╔═╡ 7ea6f6ee-0e8f-44bf-92ae-eb79f9469de2
md"""
With condition number
"""

# ╔═╡ ccd178a2-0366-4d27-82e0-5957789c7ab5
F.S[1]/F.S[2]

# ╔═╡ ebd1265e-5f11-404f-a04a-65aadaf6c800
md"""
As expected, the rank 2 approximation has a better condition number than the full rank decomposition, which is one of the reasons to approximate the matrix in the first place.
"""

# ╔═╡ 07c6c80e-c2bc-4cd4-89c7-f810bfbd3f46
md"""
### Task 6
"""

# ╔═╡ c706f0c9-581b-457c-9a9e-7f093b76f66d
md"""
First, we write the function to generate the upper triangular matrix $R=(r_{ij})$ with $r_{ii}=1$ and $r_{ij}=-1$ for $j>i$:
"""

# ╔═╡ ec7f87fb-e4aa-4d5d-a390-a580ea5e950e
function generate_R(n)
	R = [j > i ? -1 : 0 for i in 1:n, j in 1:n]
	for i in 1:n
		R[i,i] = 1
	end
	return R
end

# ╔═╡ 45a4a394-c180-4657-9dd4-6cef68df8af5
md"""
For example, the matrix $R$ of size 8 is:
"""

# ╔═╡ 08bde72f-9348-40a3-8304-d535ea348d06
generate_R(8)

# ╔═╡ e5f2fc03-ae3e-4888-b2ab-5f09e93966a6
md"""
The columns of $R$ are linear independent, henche the matrix has full rank.
"""

# ╔═╡ 071d0d94-0731-4434-aa43-1ead2b50dd96
md"""
Next, we evaluate the singular values of the matrices R of size 10, 20, 50 and 100:
"""

# ╔═╡ 2e9dfa69-d085-48de-97b8-d1ab214d2702
begin
	Ns = [10,20,50,100]
	Ss = []
	for n in Ns
		push!(Ss, svdvals(generate_R(n)))
	end
end

# ╔═╡ f63971a5-d483-4255-aa8a-74560e84eb9a
md"""
Here we plot the results:
"""

# ╔═╡ 6571e534-2de3-4bbf-8b6a-e8642a64b4a1
begin
	scp = scatter(Ss[1], [Ns[1] for _ in Ss[1]])
	for (s,n) in zip(Ss[2:end], Ns[2:end])
		scatter!(scp, s, [n for _ in s])
	end
	plot(scp, legend=false, xlabel="singular values", ylabel="N")
end
	

# ╔═╡ 32a2b126-45d8-4dde-86fd-f34401bdb646
md"""
We see that the greatest singular value depends on the dimension of the matrix. Here we plot the condition number (defined as the ratio of the greatest and smallest non-zero singular value) of $R_n$ with $n$ from 2 to 60, to check its dependency on the matrix size as well:
"""

# ╔═╡ be3777f1-300c-414f-8085-c5102e91f31f
begin
	κs = []
	for n in 1:60
		S = svdvals(generate_R(n))
		S = S[S.>0]
		append!(κs, S[1]/S[end])
	end
	plot(1:60, κs, yscale=:log10, legend=false, xlabel="matrix sixe", ylabel="condition number", yminorticks=true)
end

# ╔═╡ 63a2e02e-9056-4cef-b65c-23c4649da6af
md"""
We see that dependency is roughly exponential. This means that, above a certain dimension, the condition number will be greater than the inverse of the machine precision, hence the matrix will be not numerically full rank.
"""

# ╔═╡ 4cfd6dbb-d87b-4b08-b2c5-d482ba7ae805
md"""
Here we define a funcion `ϵ_rank()` to compute the numerical rank of a matrix by counting how many singular values are equal or greater than the machine epsilon times the greatest singular value: 
"""

# ╔═╡ 9fc8ef56-1b9d-4581-a6cf-67cfe1103272
function ϵ_rank(A)
	S = svdvals(A)
	T = typeof(S[1])
	ϵ = eps(T)
	rank_threshold = ϵ * S[1]
	return count( x -> x >= rank_threshold, S)
end

# ╔═╡ 625cb4b2-9b40-49c9-bdba-66ea8075eefa
md"""
And then we can verify our statements on the rumerical rank of $R_n$
"""

# ╔═╡ 340ca0b7-114e-471e-b9a1-3966a219caea
begin
	num_ranks = []
	matrices = 2:60
	for n in matrices
		num_rank = ϵ_rank(generate_R(n))
		if num_rank < n
			@show n, num_rank
		end
		append!(num_ranks, num_rank)
	end
	plot(matrices, num_ranks, legend=false, xlabel="matrix sixe", ylabel="numerical rank")
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
Polynomials = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
Plots = "~1.39.0"
Polynomials = "~4.0.6"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.0"
manifest_format = "2.0"
project_hash = "42cb3ce5ac7912aeebc56acdaca2cf1bf7db7dec"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BitFlags]]
git-tree-sha1 = "2dc09997850d68179b69dafb58ae806167a32b1b"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.8"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "cd67fc487743b2f0fd4380d4cbd3a24660d0eec8"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.3"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "67c1f244b991cad9b0aa4b7540fb758c2488b129"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.24.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.ColorVectorSpace.weakdeps]
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Compat]]
deps = ["UUIDs"]
git-tree-sha1 = "886826d76ea9e72b35fcd000e535588f7b60f21d"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.10.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+1"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "8cfa272e8bdedfa88b6aefbbca7c19f1befac519"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.3.0"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c53fc348ca4d40d7b371e71fd52251839080cbc9"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.4"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3dbd312d370723b6bb43ba9d02fc36abade4518d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.15"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8e9441ee83492030ace98f9789a654a6d0b1f643"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+0"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "e90caa41f5a86296e014e148ee061bd6c3edec96"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.9"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4558ab818dcceaab612d1bb8c19cee87eda2b83c"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.5.0+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "466d45dc38e15794ec7d5d63ec03d776a9aff36e"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.4+1"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d8db6a5a2fe1381c1ea4ef2cab7c69c2de7f9ea0"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.1+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "27442171f28c952804dede8ff72828a96f2bfc1f"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.72.10"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "025d171a2847f616becc0f84c8dc62fe18f0f6dd"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.72.10+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "e94c92c7bf4819685eb80186d51c43e71d4afa17"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.76.5+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "abbbb9ec3afd783a7cbd82ef01dcd088ea051398"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.1"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "a53ebe394b71470c7f97c2e7e170d51df21b17af"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.7"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "60b1194df0a3298f460063de985eae7b01bc011a"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.0.1+0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f689897ccbe049adb19a065c495e75f372ecd42b"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.4+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "f428ae552340899a935973270b8d98e5a31c49fe"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.1"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "2da088d113af58221c52828a80378e16be7d037a"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.5.1+1"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "7d6dd4e9212aebaeed356de34ccf262a3cd415aa"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.26"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "c1dd6d7978c12545b4179fb6153b9250c96b0075"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.3"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "9ee1618cbf5240e6d4e0371d6f24065083f60c48"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.11"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+2"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "51901a49222b09e3743c65b8847687ae5fc78eb2"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cc6e1927ac521b659af340e0ca45828a3ffc748f"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.12+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "a935806434c9d4c506ba941871b327b96d41f2bf"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.0"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "64779bc4c9784fee475689a1752ef4d5747c5e87"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.42.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "f92e1315dadf8c46561fb9396e525f7200cdc227"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.5"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Preferences", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "ccee59c6e48e6f2edf8a5b64dc817b6729f99eb5"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.39.0"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "RecipesBase", "Setfield", "SparseArrays"]
git-tree-sha1 = "a9c7a523d5ed375be3983db190f6a5874ae9286d"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "4.0.6"

    [deps.Polynomials.extensions]
    PolynomialsChainRulesCoreExt = "ChainRulesCore"
    PolynomialsFFTWExt = "FFTW"
    PolynomialsMakieCoreExt = "MakieCore"
    PolynomialsMutableArithmeticsExt = "MutableArithmetics"

    [deps.Polynomials.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    MakieCore = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
    MutableArithmetics = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00805cd429dcb4870060ff49ef443486c262e38e"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "37b7bb7aabf9a085e0044307e1717436117f2b3b"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.5.3+1"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "5165dfb9fd131cf0c6957a3a7605dede376e7b63"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.0"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "1d77abd07f617c4868c33d4f5b9e1dbb2643c9cf"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.2"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
git-tree-sha1 = "1fbeaaca45801b4ba17c251dd8603ef24801dd84"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.10.2"
weakdeps = ["Random", "Test"]

    [deps.TranscodingStreams.extensions]
    TestExt = ["Test", "Random"]

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "3c793be6df9dd77a0cf49d80984ef9ff996948fa"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.19.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "e2d817cc500e960fdbafcf988ac8436ba3208bfd"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.3"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "7558e29847e99bc3f04d6569e82d0f5c54460703"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+1"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "801cbe47eae69adc50f36c3caec4758d2650741b"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.12.2+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "522b8414d40c4cbbab8dee346ac3a09f9768f25d"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.4.5+0"

[[deps.Xorg_libICE_jll]]
deps = ["Libdl", "Pkg"]
git-tree-sha1 = "e5becd4411063bdcac16be8b66fc2f9f6f1e8fe5"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.0.10+1"

[[deps.Xorg_libSM_jll]]
deps = ["Libdl", "Pkg", "Xorg_libICE_jll"]
git-tree-sha1 = "4a9d9e4c180e1e8119b5ffc224a7b59d3a7f7e18"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.3+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "b4bfde5d5b652e22b9c790ad00af08b6d042b97d"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.15.0+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "730eeca102434283c50ccf7d1ecdadf521a765a4"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "04341cb870f29dcd5e39055f895c39d016e18ccd"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.4+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "330f955bc41bb8f5270a369c473fc4a5a4e4d3cb"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "49ce682769cd5de6c72dcf1b94ed7790cd08974c"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.5+0"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "gperf_jll"]
git-tree-sha1 = "431b678a28ebb559d224c0b6b6d01afce87c51ba"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.9+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a68c9655fbe6dfcab3d972808f1aafec151ce3f8"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.43.0+0"

[[deps.gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3516a5630f741c9eecb3720b1ec9d8edc3ecc033"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.1.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "141fe65dc3efabb0b1d5ba74e91f6ad26f84cc22"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "ad50e5b90f222cfe78aa3d5183a20a12de1322ce"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.18.0+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "93284c28274d9e75218a416c65ec49d0e0fcdf3d"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.40+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "814e154bdb7be91d78b6802843f76b6ece642f11"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.6+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9c304562909ab2bab0262639bd4f444d7bc2be37"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+1"
"""

# ╔═╡ Cell order:
# ╠═6476e2c3-4f65-4df3-84e3-a3f1aa2864f5
# ╟─72d1aade-524a-45f0-bdca-c5cd2456dfa5
# ╟─965e4ec5-d54c-4f2d-999b-4cee64672069
# ╟─4352df2d-210e-436d-bc00-528767d8afb2
# ╟─9b48e10a-c7c6-4e8d-968f-d6774416a942
# ╟─c378495c-3030-49f9-8031-486b4b4b4417
# ╟─b18a53e4-64ad-4edd-8f9b-83f92c79e35f
# ╟─aaa78df6-9ebe-4867-b8fc-778d33e84606
# ╟─b847ef7a-171f-404a-8761-3eaaa0e5f697
# ╟─034e39df-dcbf-42ec-b84e-1e5dc3f43aae
# ╟─5a7b400d-145b-4cd0-a1cd-067f4d95d072
# ╠═d024eff2-fbc0-403c-b950-fb9e9d7ed97d
# ╟─04c64c37-febc-414f-8b15-eb25ffdf459b
# ╟─c9e70e07-eb26-4184-ac78-442966ea4744
# ╠═29fe2bda-d242-49ad-bcec-fdc6747fb25d
# ╠═3c55be81-ff87-4a83-9807-0fd999d6da40
# ╟─d8ad5b17-8a8e-4cff-9c50-da4f98afb62f
# ╠═10e65287-1069-42d0-859a-6e6605226985
# ╟─9a9368c6-e7cb-46f7-82c5-26a08bd8a3c4
# ╠═91b1bef0-faff-448a-bac6-dafa9dfc51ae
# ╟─6591340b-c688-41c1-8d82-7c370fad3b4c
# ╠═d234abdf-a14a-449f-80d9-bf53f36264ff
# ╟─0d2c2305-757c-49a8-ba6a-050462409f00
# ╠═5bfc6d26-0da9-4e50-8f8d-0b2b42df72ba
# ╟─8176c4f4-07bb-457b-9514-58227cf4221e
# ╠═70be3242-51db-4566-a440-bed44ff4f101
# ╟─ce2da4cf-5418-4043-887a-9dd5eebc388a
# ╠═37453bfc-099d-4541-a76d-9d971231681c
# ╟─3e1b226d-cde2-4a16-b9fd-8b73e2086108
# ╟─839634e0-268a-41b2-afe1-290876cb4af5
# ╟─a26fd903-7541-40d4-9e95-443a57fe5a0c
# ╟─40d89559-f9a0-46c2-8caf-6f3339929620
# ╟─88a78727-afc2-4de9-8395-5f3deb16cfe2
# ╟─224fa724-812c-4c67-8eed-88ab20b880cd
# ╟─c12e7827-637c-42fe-a03e-965c65daf3ef
# ╟─08286a34-cf70-4ae5-862c-20cde4d03ea5
# ╟─afb8c3f3-b05c-4c2b-a916-8cb94c302649
# ╟─19f13630-6fe8-4201-97ee-0b41d733a934
# ╟─1cc06e5b-94bd-4faf-b082-6ce72c309803
# ╟─400e00db-717f-48d1-817b-ef4af51e56b9
# ╟─a99d636a-7180-46b1-8cbd-4fd01d96e9c6
# ╟─6864243d-bb5a-487a-8176-f5ce34bc9d29
# ╟─fa630a70-8b28-467e-afd5-30f2f4f8a9ef
# ╟─fbad5eff-71e3-4364-8ada-5e9409dca5e4
# ╟─ef6a16e1-c896-40ca-87e0-878bfb5bccfa
# ╟─ef2a1c8d-bfef-449d-a10e-c35c778d5aee
# ╟─99fcdeed-a3f4-4527-830c-c280ec662429
# ╟─4ea3aae8-a54b-4d76-b920-610188b8f776
# ╟─4a7b6aea-780b-4169-a67d-efd10eb98d76
# ╠═ced41e59-8e54-4d91-9c26-ef69853d8ca3
# ╠═8bd80629-8a01-4ca1-b86b-af2e7084d47d
# ╟─1f6382f4-0ef1-4180-b62f-af261fc5b005
# ╠═d7d6bbb6-f8ea-4e71-b24d-ec6c75230f99
# ╟─abedbe7b-9ec8-44fb-9303-a4e9f862533a
# ╠═3dd200b9-8c0c-40fc-b871-0137b5cf315e
# ╟─9da832c7-0522-48e2-9129-f6e1cbd2f3fa
# ╟─6d3581b1-6499-4010-8019-ee796643b81e
# ╠═30d17722-7de3-4c20-bc0a-e35d76e4f180
# ╟─403e386e-37ac-4577-a129-643d0af87ebf
# ╟─3f60adf8-bcf9-4b8b-afb3-99bae402745e
# ╟─09679dad-aad2-4d5b-9df9-bbf90e6238d7
# ╟─d793814f-110e-4148-b56b-4e8e6a946e98
# ╠═3239de2d-54ef-4798-8387-da3345bca6c0
# ╟─93c94d3a-9791-4d0f-a6f1-2af12f20fd22
# ╠═ddfb314c-0c57-4a41-bb22-80d3cd36b827
# ╟─57411892-53dd-4245-bdfc-ee0ad2f6fda2
# ╠═df5c4753-b803-4fd1-a8a4-003a3a5b7840
# ╟─5e2c2fc5-559d-4b8e-9040-f9a3abb639b5
# ╠═21cde6be-16f3-4e77-aa07-9cb50cf646a5
# ╟─4a699e21-0666-49a1-b6b8-ccb73142e3d0
# ╠═99b6d0f6-becc-4de1-acc5-29f0629da5be
# ╟─81fd82fa-8a10-4638-8486-3a8087d44a2b
# ╠═ae183c96-0c59-44a9-b3af-d1f1576a0b4c
# ╟─7bc9df74-f24f-4d96-a407-07b3eac37c35
# ╟─b260470f-d366-405b-b71f-e9b3044ffc5b
# ╟─d7712c3f-81c9-4151-8c25-298eaade303a
# ╠═d01412c9-7671-4049-b727-e8239b8aaafa
# ╟─ec66c587-f119-4817-8fc5-8f824af40bfe
# ╠═f45f4d67-2191-4f61-945e-b7a6684d5b01
# ╟─7ea6f6ee-0e8f-44bf-92ae-eb79f9469de2
# ╠═ccd178a2-0366-4d27-82e0-5957789c7ab5
# ╟─ebd1265e-5f11-404f-a04a-65aadaf6c800
# ╟─07c6c80e-c2bc-4cd4-89c7-f810bfbd3f46
# ╟─c706f0c9-581b-457c-9a9e-7f093b76f66d
# ╠═ec7f87fb-e4aa-4d5d-a390-a580ea5e950e
# ╟─45a4a394-c180-4657-9dd4-6cef68df8af5
# ╠═08bde72f-9348-40a3-8304-d535ea348d06
# ╟─e5f2fc03-ae3e-4888-b2ab-5f09e93966a6
# ╟─071d0d94-0731-4434-aa43-1ead2b50dd96
# ╠═2e9dfa69-d085-48de-97b8-d1ab214d2702
# ╟─f63971a5-d483-4255-aa8a-74560e84eb9a
# ╠═6571e534-2de3-4bbf-8b6a-e8642a64b4a1
# ╟─32a2b126-45d8-4dde-86fd-f34401bdb646
# ╠═be3777f1-300c-414f-8085-c5102e91f31f
# ╟─63a2e02e-9056-4cef-b65c-23c4649da6af
# ╟─4cfd6dbb-d87b-4b08-b2c5-d482ba7ae805
# ╠═9fc8ef56-1b9d-4581-a6cf-67cfe1103272
# ╟─625cb4b2-9b40-49c9-bdba-66ea8075eefa
# ╠═340ca0b7-114e-471e-b9a1-3966a219caea
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
