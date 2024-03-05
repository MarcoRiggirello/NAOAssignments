### A Pluto.jl notebook ###
# v0.19.36

using Markdown
using InteractiveUtils

# ╔═╡ 6e15ed80-1938-4ea9-82f8-a7d0b8dfd6ce
using LinearAlgebra, StaticArrays, Symbolics, Plots

# ╔═╡ 45ad341c-765c-40e1-a726-8a575dc3ab9a
@variables x₁, x₂

# ╔═╡ 61498356-cf52-11ee-3c16-038f22b043c9
md"""
# Numerical Analysis and Optimization Project 4
"""

# ╔═╡ 7435d8e2-dd5c-49bd-981e-46ee8ece3313
md"""
In the following, we define the structs and the functions for the assignment. Specifically, we define the following functions:

- `optimize_newton()`, which performs the Newton's optimization. Optionally, a linesearch function can be passed to determine the optimal steplength;
- `optimize_bfgs()`, which performs the BFGS quasi-Newton optimization (optional linesearch is available here as well);
- `optimize_trustregion()`, which performs the optimization via the Trust region algorithm.

As the linesearch algorithm, we implement the `backtracking()` algorithms as seen during the lectures.
"""

# ╔═╡ ed785d1d-54e9-49df-b45f-dffa161d978c
struct OptimizationResults
	algorithm::String
	converged::Bool
	min::Real
	argmin::Vector
	last_grad_norm::Real
	last_step_norm::Real
	n_iterations::Int
	steps::Matrix
	values::Vector
end

# ╔═╡ 52387950-56cc-414e-a434-8480757fdd45
md"""
## Newton
"""

# ╔═╡ 3f060cb3-3465-45ae-9a9a-4ec3f3628cf2
md"""
Note that we perform an additional check when doing the Cholesky decomposition of the Hessian to catch those cases where the matrix is not Positive defined. In this case the algorithm can converge to a saddle point and not to a minimum and so the function throws an error.
"""

# ╔═╡ fa77e47e-6876-42d5-ae9d-9c68e9afd174
function optimize_newton(f, g!, H!, x₀; tol=1e-5, maxitr=100, linesearch=Nothing)
	n = length(x₀)
	xₖ = zeros(n)
	gₖ = zeros(n)
	Hₖ = zeros(n,n)
	xₖ .= x₀
	steps = copy(xₖ)
	values = [f(xₖ)]
	step_norm = 0.
	grad_norm = 0.
	k = 0
	converged = false
	while k ≤ maxitr && !converged
		k += 1
		g!(gₖ, xₖ)
		H!(Hₖ, xₖ)
		Cₖ = 
			try 
				cholesky(Symmetric(Hₖ))
			catch e
				if e isa(PosDefException)
					throw(
						DomainError("Hessian is not positive definite at $xₖ.")
					)
				else
					throw(e)
				end
			end
		pₖ = -(Cₖ \ gₖ)
		αₖ = linesearch ≠ Nothing ? linesearch(f, xₖ, gₖ, pₖ) : 1.
		xₖ .+= αₖ * pₖ
		step_norm = norm(αₖ * pₖ)
		grad_norm = norm(gₖ)
		steps = [steps xₖ]
		push!(values, f(xₖ))
		converged = grad_norm ≤ tol && step_norm ≤ tol * (1 + norm(xₖ))
	end
	ls_name = linesearch ≠ Nothing ? " - $linesearch" : ""
	return OptimizationResults(
		"Newton" * ls_name,
		converged,
		f(xₖ),
		xₖ,
		grad_norm,
		step_norm,
		k,
		steps,
		values
	)
end

# ╔═╡ 27a151a4-1028-4baf-b4b4-138fa5ce9040
md"""
## BFGS
"""

# ╔═╡ fca61f4d-948a-4a05-a387-16ebe84ab56d
md"""
Here the BFGS update is carried out on the approzimation of the hessian inverse, as seen during lectures.
We used a slightly different version of the update formula to avoid computation of intermediary matrices. Starting from the form

$G_{k+1} = etc...$

by expanding the matrix product we can write

$etc$

By proper collecting the terms in the above formula, we note that we can perform only matrix--vector products and scalar products. We used this in the code implementation of the BFGS.
"""

# ╔═╡ 41e06b52-3b8c-47be-9a75-fdf3d708def4
function optimize_bfgs(f, g!, G₀, x₀; tol=1e-5, maxitr=100, linesearch=Nothing, comment=Nothing)
	n = length(x₀)
	xₖ   = zeros(n)
	gₖ   = zeros(n)
	gₖ₋₁ = zeros(n)
	Gₖ   = zeros(n,n)
	xₖ .= x₀
	g!(gₖ, x₀)
	Gₖ .= G₀
	steps = copy(xₖ)
	values = [f(xₖ)]
	step_norm = 0.
	grad_norm = 0.
	k = 0
	converged = false
	while k ≤ maxitr && !converged
		k += 1
		pₖ = -(Gₖ * gₖ)
		αₖ = linesearch ≠ Nothing ? linesearch(f, xₖ, gₖ, pₖ) : 1.
		xₖ .+= αₖ * pₖ
		step_norm = norm(αₖ * pₖ)
		grad_norm = norm(gₖ)
		steps = [steps xₖ]
		push!(values, f(xₖ))
		converged = grad_norm ≤ tol && step_norm ≤ tol * (1 + norm(xₖ))
		# gradient and secant update
		sₖ = αₖ * pₖ
		gₖ₋₁ .= gₖ
		g!(gₖ, xₖ)
		yₖ = gₖ - gₖ₋₁
		# intermediary variables
		yₖᵀsₖ   = yₖ'   * sₖ
		Gₖyₖ    = Gₖ    * yₖ
		yₖᵀGₖ   = yₖ'   * Gₖ
		yₖᵀGₖyₖ = yₖᵀGₖ * yₖ
		# Gₖ update
		if yₖᵀsₖ ≤ 0
			throw(DomainError("yₖᵀsₖ value is $yₖᵀsₖ at $xₖ"))
		end
		Gₖ .-= (Gₖyₖ .* sₖ' .+ sₖ .* yₖᵀGₖ) ./ yₖᵀsₖ
		Gₖ .+= ((yₖᵀsₖ + yₖᵀGₖyₖ)/(yₖᵀsₖ)^2) .* sₖ .* sₖ'
	end
	ls_name = linesearch ≠ Nothing ? " - $linesearch" : ""
	comment = comment ≠ Nothing ? " (" * comment * ")" : ""
	return OptimizationResults(
		"BFGS" * ls_name * comment,
		converged,
		f(xₖ),
		xₖ,
		grad_norm,
		step_norm,
		k,
		steps,
		values
	)
end

# ╔═╡ 6698003b-240b-4f1b-b1e4-8e70107c4150
md"""
## Backtracking
"""

# ╔═╡ b4c6e46a-f47e-4ce2-8cd9-c6229be6fdf4
function backtracking(f, xₖ, gₖ, pₖ; c=.5, ρ=.5, α₀=1.)
	if !(0 < c < 1) || !(0 < ρ < 1)
		throw(DomainError("Parameters c and/or ρ outside proper bound."))
	end
	α = α₀
	fₖ = f(xₖ)
	pₖᵀ∇f = pₖ' * gₖ
	fₜ = f(xₖ + α * pₖ)
	while fₜ > f(xₖ) + c * α * pₖᵀ∇f
		α *= ρ
		fₜ = f(xₖ + α * pₖ)
	end
	return α
end

# ╔═╡ 35f73a6e-ada1-412a-a770-175427d1e443
md"""
## Trust region
"""

# ╔═╡ 3220a80c-bd63-4683-99db-5a53901c9b0c
md"""
Here the trust region subproblem, defined as

$\arg \min_{||\mathbf{p}|| < \Delta} \mathbf{p}^T \mathbf{g} + \frac12 \mathbf{p}^T H \mathbf{p}$

is solved with penalty method by finding a $\mu$ such that 

$\mathbf{p} = (H + \mu I)^{-1} \mathbf{g}$

is within the trust region. As we proved during lectures, $||\mathbf{p}||$ is monothonically decrescent function of $μ$, so our alorithm proceed iteratively in two phases:
- First, starting from a initial value $\mu_0$, we double at each step the value of $μ_k$ until we get $||\mathbf{p}||_{\mu_k} < \Delta$;
- Then, we perform a binary search in the window $[\mu_{k-1}, \mu_k]$ to be as close as possible to the threshold $||\mathbf{p}||_{\mu_\text{thr}} = \Delta$ (we don't want to flatten the hessian information over the identity).
"""

# ╔═╡ e5d9ac0a-7cfc-44ab-b29f-fb8d2f1f69c7
function solve_penalty_problem(g, H, Δ)
	n = length(g)
	steps = 10
	μₗ = max(0., -eigmin(H)) # ensure positive definiteness
	μᵤ = μₗ ≠ 0 ? 2 * μₗ : 1.
	while steps > 0
		Cⱼ = cholesky(H + μᵤ * I(n))
		pⱼ = Cⱼ \ g
		if norm(pⱼ) > Δ
			μₗ = μᵤ
			μᵤ *= 2
			steps += 1
		else
			μⱼ = (μᵤ + μₗ) / 2
			Cⱼ = cholesky(H + μⱼ * I(n))
			pⱼ = Cⱼ \ g
			if norm(pⱼ) > Δ
				μₗ = μⱼ
			else
				μᵤ = μⱼ
			end
			steps -= 1
		end
	end
	μ = μᵤ
	C = cholesky(H + μ * I(n))
	p = C \ g
	return -p
end	

# ╔═╡ 54500345-520f-479c-88ce-f7d1a373389f
function optimize_trustregion(f, g!, H!, x₀; Δ₀=.1, η=0.020, tol=1e-5, maxitr=100)
	n = length(x₀)
	xₖ = zeros(n)
	gₖ = zeros(n)
	Hₖ = zeros(n,n)
	xₖ .= x₀
	Δₖ = Δ₀
	steps = copy(xₖ)
	values = [f(xₖ)]
	step_norm = 0.
	grad_norm = 0.
	k = 0
	converged = false
	while k ≤ maxitr && !converged
		k += 1
		g!(gₖ, xₖ)
		H!(Hₖ, xₖ)
		pₖ = solve_penalty_problem(gₖ, Hₖ, Δₖ)
		Δm = - gₖ' * pₖ - 1/2 * dot(pₖ, Hₖ, pₖ)
		Δf = f(xₖ) - f(xₖ + pₖ)
		ρₖ = Δf / Δm
		if 0 < ρₖ < 0.25
			Δₖ = norm(pₖ) / 4
		elseif ρₖ > 0.75
			Δₖ *= 2
		end
		if ρₖ > η
			xₖ .+= pₖ
		else
			pₖ = zeros(n)
		end
		step_norm = norm(pₖ)
		grad_norm = norm(gₖ)
		steps = [steps xₖ]
		push!(values, f(xₖ))
		converged = grad_norm ≤ tol && step_norm ≤ tol * (1 + norm(xₖ))
	end
	return OptimizationResults(
		"Trust Region",
		converged,
		f(xₖ),
		xₖ,
		grad_norm,
		step_norm,
		k,
		steps,
		values
	)
end

# ╔═╡ f8dc799a-d2ef-43ae-aa8b-5a43b094d7c3
md"""
## Graphics
"""

# ╔═╡ b8e94ea4-4d08-4e64-a134-e2562e7d007e
md"""
Here we define an helper function for plotting 2d optimization tasks.
"""

# ╔═╡ 2b253872-ca18-4f81-9a90-565d1503576d
function plot_optimization(f, truemin, opts...; xlim=(-10,10), ylim=(-10,10), gridsize=100)
	if any(j -> size(j.steps, 1) != 2, opts)
		throw(ArgumentError("Optimizations are not of size 2: impossible to plot."))
	end
	# Function landscape
	x = range(xlim..., gridsize)
	y = range(ylim..., gridsize)
	z = @. f(x', y)
	# Optimization steps array
	X = [o.steps[1,1] for o in opts]
	Y = [o.steps[2,1] for o in opts]
	# Black magic to animate optimization
	for (i,o) in enumerate(opts)
		for c in eachcol(o.steps)
			X = [X X[:, end]]
			Y = [Y Y[:, end]]
			X[i, end] = c[1]
			Y[i, end] = c[2]
		end
	end
	# Dealing with the contour plot and the scatter plot
	# (we don't want to update them in the animation)
	X = [NaN; NaN; X]
	Y = [NaN; NaN; Y]
	# Plotting! (Finally)
	plt = contour(
		x, y, z,
		xlim=xlim,
		ylim=ylim,
		xlabel="x₁",
		ylabel="x₂",
		legend=:outertop,
		legend_column = 2,
		dpi=180,
	)
	scatter!(
		plt,
		truemin,
		marker=:circle,
		label="True min"
	)
	for o in opts
		plot!(
			plt,
			1,
			label=o.algorithm
		)
	end
	# Animation of minimization steps
	@gif for (i, j) in zip(eachcol(X), eachcol(Y))
		push!(plt, i, j)
	end
end	
	

# ╔═╡ 96e08046-c44d-421c-9784-72a97ff8fbf8
md"""
## Optimization a
"""

# ╔═╡ 81742d93-d761-4ec1-ab3a-571711130fc6
md"""
We plot below the shape of the function.
"""

# ╔═╡ 57c33dc0-1e46-4c33-a432-550307367839
f_a(x₁, x₂) = (x₁ - 2)^2 + (x₁  - 2)^2 * x₂^2 + (x₂ + 1)^2

# ╔═╡ 85a9fbca-7866-4c7f-899e-d6738a96596a
begin
	x_a = range(1, 3, 100)
	y_a = range(-1.5, 1., 100)
	z_a = @. f_a(x_a', y_a)
	surface(x_a, y_a, z_a)
end

# ╔═╡ cdfc565c-c18b-4e8e-aa17-dfb9483e9eee
md"""
We use the power of [Symbolics.jl](https://symbolics.juliasymbolics.org/stable/) to derive the analytic form of both the gradient and the Hessian of the function.
"""

# ╔═╡ afc86c35-5c79-4f26-881d-45fcaea8dbb1
∇f_a = Symbolics.gradient(f_a(x₁, x₂), [x₁, x₂])

# ╔═╡ c156f0f0-1c1f-4c19-bb92-8cd3db60215d
md"""
We use [this trickery](https://symbolics.juliasymbolics.org/dev/getting_started/#Building-Functions) to make the symbolic function callable. The function `g_a!` takes as input a preallocated vector, which is more efficient in this scenario.
"""

# ╔═╡ d30f8a04-5fde-4b1c-933c-577229b8de77
g_a! = eval(build_function(∇f_a, [x₁, x₂])[2])

# ╔═╡ d73ae180-2d77-4e7f-a3fb-c9a0abaa9c51
md"""
The same is done for the Hessian. Everything is now ready to perform the opimization.
"""

# ╔═╡ b86e5b21-7b90-4111-b6da-eddbf7402bf2
∇²f_a = Symbolics.hessian(f_a(x₁, x₂), [x₁, x₂])

# ╔═╡ 7074f6d2-94c1-4440-9a44-7936db617bb3
H_a! = eval(build_function(∇²f_a, [x₁, x₂])[2])

# ╔═╡ f975f7e7-1790-47b3-a0ef-014df5f5a9fb
H_a = eval(build_function(∇²f_a, [x₁, x₂])[1])

# ╔═╡ c95a8234-9729-451b-9745-1abf35819000
md"""
### Runs
"""

# ╔═╡ d629443a-45a2-4abe-80fb-0b233863c960
md"""
In the first starting point, the Hessian is not SPD, so Newton fails:
"""

# ╔═╡ 5edc0250-8d8c-4c94-91ab-08a813845668
newt_a_1 = optimize_newton(x -> f_a(x[1], x[2]), g_a!, H_a!, [1,1])

# ╔═╡ d2b6d03b-45c4-4941-b18e-c7e3fc451889
md"""
However, BFGS is capable of converging to the minimum, even by using the Identity as the initial approximation of the Hessian inverse.
"""

# ╔═╡ c5a4de9b-e452-4428-83ae-6f9209e22cf9
bfgs_a_1 = optimize_bfgs(x -> f_a(x[1], x[2]), g_a!, [1 0; 0 1], [1,1], comment="G₀=Id")

# ╔═╡ cea16f18-a943-47c2-8803-dc877262ae8f
md"""
Convergence is also obtained through BFGS when starting from another (positive definite) approximation of the Hessian inverse.
"""

# ╔═╡ e446999e-490d-4afc-aaf7-9536d3ecd1f0
bfgs_a_2 = optimize_bfgs(x -> f_a(x[1], x[2]), g_a!, inv(H_a([1,1])+1e-4*I(2)), [1,1], comment="G₀=(∇²f(x₀)+1e-4Id)⁻¹")

# ╔═╡ 3f27c70f-8ad7-42bb-b93d-e7e7e586e095
md"""
If we use the backtracking algorithm to determine the steplength (instead of using  just $α_k = 1$ as above) with BFGS and the Identity as initial guess, we obtain a faster convergence than before. However, we observe that results for the argmin are slightly less accurate, while still within the required tolerance.
"""

# ╔═╡ 12d943f9-83b3-44e2-a0f5-f7becb7b614f
bfgs_a_3 = optimize_bfgs(x -> f_a(x[1], x[2]), g_a!, [1 0; 0 1], [1,1], linesearch=backtracking, comment="G₀=Id")

# ╔═╡ d6007def-8349-4678-aa6c-4b3c771b3c15
md"""
We repeat the backtracking with the different approximation for the Hessian inverse. The number of steps is again reduced.
"""

# ╔═╡ 29a1ba8e-e27a-45b4-a1ac-dcc2eefa05eb
bfgs_a_4 = optimize_bfgs(x -> f_a(x[1], x[2]), g_a!, inv(H_a([1,1])+1e-4*I(2)), [1,1], linesearch=backtracking, comment="G₀=(∇²f(x₀)+1e-4Id)⁻¹")

# ╔═╡ 51a11fce-5808-4479-b79f-fe8ae9310c3e
md"""
The trust region approach converges to accurate results with the least amount of steps.
"""

# ╔═╡ 5a78ea6c-b4ce-4d4f-9719-e09e9c1f36fd
trrg_a_1 = optimize_trustregion(x -> f_a(x[1], x[2]), g_a!, H_a!, [1,1])

# ╔═╡ 4fca73b7-be7c-427e-9b22-0c60f50279e4
md"""
We show below the animation for the different minimization routines. When BFGS is used without backtracking, the fixed step size results in a series of sharp steps across the domain.
"""

# ╔═╡ e11ee2fb-af2b-4daf-8c12-4e625f891f84
plot_optimization(f_a, (2,-1), bfgs_a_1, bfgs_a_2)

# ╔═╡ f3acf4de-592e-4709-a687-ad66e355360c
md"""
 Instead, the use of backtracking allows for smoother descent towards the minimum.
"""

# ╔═╡ 40df2ad0-0189-474c-bb71-3e00effae4b1
plot_optimization(f_a, (2,-1), bfgs_a_3, bfgs_a_4, trrg_a_1, xlim=(0, 3), ylim=(-2.5,2.5))

# ╔═╡ 95212305-bd5b-4c47-87dc-116231452373
md"""
The other starting point is exactly the minimum, so Newton converges in one step.
"""

# ╔═╡ a0d5ab72-f12d-4a75-93cd-40365b6bd4df
newt_a_2 = optimize_newton(x -> f_a(x[1], x[2]), g_a!, H_a!, [2,-1])

# ╔═╡ 23ab718b-5237-4cce-927e-ab29ec98deae
md"""
## Optimization b
"""

# ╔═╡ 0b6ed726-232f-484b-9e20-28bfdb812a67
md"""
here we don't need Symbolics.
"""

# ╔═╡ 015c5739-0e4a-426a-a6f9-376213da2424
b = [5.04, -59.4, 146.4, -96.6]

# ╔═╡ 10918712-2518-439e-9597-b4341e938ebd
H  = [
	 0.16 	 -1.2 	  2.4 	 -1.4;
	-1.2 	 12.0 	-27.0 	 16.8;
	 2.4 	-27.0 	 64.8 	-42.0;
	-1.4 	 16.8 	-42.0 	 28.0
]

# ╔═╡ 89e8b816-6a52-461c-b76d-c72b94c29974
f_b(x) = dot(b, x) + dot(x, H, x) / 2

# ╔═╡ 45081330-abf2-4ff3-bd32-9197446d3962
function g_b!(g, x)
	g .= b + H * x
end

# ╔═╡ 36afaf29-e07a-450d-a7e0-b9c057c2a698
function H_b!(h, x)
	h .= H
end

# ╔═╡ 376b44e1-7227-4e83-a558-55780235c568
md"""
### Runs
"""

# ╔═╡ 32fd9af2-16a6-44bd-8382-9f822395d3e6
newt_b_1 = optimize_newton(f_b, g_b!, H_b!, [-1.,3.,3.,0.])

# ╔═╡ 299e411b-6ccc-4f12-9d79-2b7275547e52
bfgs_b_1 = optimize_bfgs(f_b, g_b!, I(4), [-1.,3.,3.,0.], linesearch=backtracking)

# ╔═╡ 0c0114d3-ab4a-4631-b8a7-3fc0ec922859
bfgs_b_2 = optimize_bfgs(f_b, g_b!, inv(H), [-1.,3.,3.,0.])

# ╔═╡ 207e0b4c-3a6b-41b8-a760-0f02e048a6df
trrg_b_1 = optimize_trustregion(f_b, g_b!, H_b!, [-1.,3.,3.,0.])

# ╔═╡ b46d53a0-d52a-463c-951f-e5cb7a4e86fb
md"""
## Optimization c
"""

# ╔═╡ f59d3dac-21ad-45bc-9933-8182eebd5ce4
f_c(x₁, x₂) = (1.5 - x₁ * (x₂))^2 + (2.25 - x₁ * (1 - x₂^2))^2 + (2.625 - x₁ * (1 - x₂^3))^2

# ╔═╡ 925503d2-4262-4da0-b00a-c970ebde0318
begin
	x_c = range(0, 8, 100)
	y_c = range(0, 1, 100)
	z_c = @. f_c(x_c', y_c)
	surface(x_c, y_c, z_c)
end

# ╔═╡ bfffcc68-3719-40ef-87b7-ff944d9b6ae2
∇f_c = Symbolics.gradient(f_c(x₁, x₂), [x₁, x₂])

# ╔═╡ ba043839-e13d-475a-a93e-b170af23327d
g_c! = eval(build_function(∇f_c, [x₁, x₂])[2])

# ╔═╡ 851793c3-b9fd-44ee-b96f-0f0734fae4d9
∇²f_c = Symbolics.hessian(f_c(x₁, x₂), [x₁, x₂])

# ╔═╡ 8fcac8e1-7059-4547-beae-7d1d5ed44420
H_c! = eval(build_function(∇²f_c, [x₁, x₂])[2])

# ╔═╡ 3adc0e72-05e1-4c8b-8c7a-d2a176d18d49
H_c = eval(build_function(∇²f_c, [x₁, x₂])[1])

# ╔═╡ 1c6f05b3-a1c0-4294-9316-a1e0049550b2
md"""
### Runs
"""

# ╔═╡ 1f83f39f-2e67-4bc4-bc23-c5f5093fd7aa
md"""
	Che questo problema serva da esempio
"""

# ╔═╡ cc42fc4f-c0ab-461e-822f-07d9b36afc79
md"""
	si prova newton, non va
"""

# ╔═╡ 059f8000-ea4e-4338-b75e-89f136794c67
newt_c_1 = optimize_newton(x -> f_c(x[1], x[2]), g_c!, H_c!, [8.,0.2])

# ╔═╡ 94d79163-fa2a-4183-8802-213f605dcf59
md"""
	si prova bfgs con l'identità come G0, non va
"""

# ╔═╡ caaf83db-4701-45a1-9649-e34576470735
bfgs_c_1 = optimize_bfgs(x -> f_c(x[1], x[2]), g_c!, [1 0; 0 1], [8.,0.2], comment="G₀ = I")

# ╔═╡ f4e6494d-6c74-4a5f-ac8a-8ca7b0d2243d
md"""
	Si prova bfgs con l'inverso dell'hessiana, va
"""

# ╔═╡ 72ab53df-83e8-4ecf-a811-c1e8e848d249
bfgs_c_2 = optimize_bfgs(x -> f_c(x[1], x[2]), g_c!, inv(H_c([8,0.2])), [8.,0.2], comment="G₀=(∇²f(x₀))⁻¹")

# ╔═╡ 7b4cfc1c-17f6-44c1-a009-102e9f60d044
md"""
	si prova bfgs con bactrackin tornado a una stima spannometrica dell'hessiana, ora va
"""

# ╔═╡ 8853d92c-a774-40ee-a71f-4c6b7511686d
bfgs_c_1_bt = optimize_bfgs(x -> f_c(x[1], x[2]), g_c!, [1 0; 0 1], [8.,0.2], linesearch=backtracking, comment="G₀ = I")

# ╔═╡ 144103b9-24b2-4772-8d2e-119ff1a73c81
md"""
	si prova con trust region, va
"""

# ╔═╡ 91f47c4c-e334-4741-8105-1a6b14cf0e09
trrg_c_1 = optimize_trustregion(x -> f_c(x[1], x[2]), g_c!, H_c!, [8.,0.2])

# ╔═╡ 07fe8486-decd-43ea-bdbb-b97551edeb3a
md"""
	Si prova da un altro punto iniziale, Newton va
"""

# ╔═╡ 424974c8-a2e6-447b-b7f5-29e28572314d
newt_c_2 = optimize_newton(x -> f_c(x[1], x[2]), g_c!, H_c!, [8.,0.8])

# ╔═╡ 8f15f324-0cff-4737-ad0e-895173e8914e
md"""
	si prova bfgs spannometrico, non va manco da qui
"""

# ╔═╡ bdb8361d-e904-4857-bd96-16e609c4fe85
bfgs_c_3 = optimize_bfgs(x -> f_c(x[1], x[2]), g_c!, [1 0; 0 1], [8.,0.8], comment="G₀ = I")

# ╔═╡ e3a3d369-70ff-4f9b-9daa-5df76be0b071
md"""
	stima accurata va
"""

# ╔═╡ 7f792dcd-e1f6-40c5-bf76-55bc640ceb81
bfgs_c_4 = optimize_bfgs(x -> f_c(x[1], x[2]), g_c!, inv(H_c([8,0.8])), [8.,0.8], comment="G₀=(∇²f(x₀))⁻¹")

# ╔═╡ 6eafa5e6-5eb6-4c4c-889a-85093320ea26
md"""
	bactracking migliora
"""

# ╔═╡ 7b3cd1cb-e2b6-408d-b981-0ffbb3174822
bfgs_c_3_bt = optimize_bfgs(x -> f_c(x[1], x[2]), g_c!, [1 0; 0 1], [8.,0.8], linesearch=backtracking, comment="G₀ = I")

# ╔═╡ 2b727c28-62a5-4290-82e8-c486467b98b3
md"""
	questo inutile gia andava
"""

# ╔═╡ a02f8144-9b20-4caa-aadc-0fe00a8709d9
bfgs_c_6 = optimize_bfgs(x -> f_c(x[1], x[2]), g_c!, inv(H_c([8,0.8])), [8.,0.8], linesearch=backtracking, comment="G₀=(∇²f(x₀))⁻¹")

# ╔═╡ b0e20e6b-55c8-498c-ba4b-3393624e49ae
md"""
	trust region va anche sui sassi
"""

# ╔═╡ 7a00d2c7-6dc1-4795-8328-a7696a32cc56
trrg_c_2 = optimize_trustregion(x -> f_c(x[1], x[2]), g_c!, H_c!, [8.,0.8])

# ╔═╡ e12fdb33-d126-415b-b195-f75a2697f5e3
plot_optimization(f_c, (3,0.5), bfgs_c_2, bfgs_c_1_bt, trrg_c_1, newt_c_2,  bfgs_c_4, bfgs_c_3_bt, trrg_c_2, xlim=(2.5,8.5), ylim=(0,1))

# ╔═╡ 775a3f3b-94aa-4502-8b99-57824fa8a805
md"""
## Optimization d
"""

# ╔═╡ 3fbe3908-eee9-440b-993c-67e8a84d10e9
f_d(x₁, x₂) = x₁^4 + x₁ * x₂ + (1 + x₂)^2

# ╔═╡ b22fa713-2fe9-475c-97b8-688329d6969d
begin
	x_d = range(-2, 2, 100)
	y_d = range(-2, 2, 100)
	z_d = @. f_d(x_d', y_d)
	surface(x_d, y_d, z_d)
end

# ╔═╡ 9b835cfe-2206-4deb-b958-d3672e1a77fe
∇f_d = Symbolics.gradient(f_d(x₁, x₂), [x₁, x₂])

# ╔═╡ c7220592-fc91-4c1f-9c71-5dab5475c32d
g_d! = eval(build_function(∇f_d, [x₁, x₂])[2])

# ╔═╡ bef5fb16-a753-4375-90ef-e4bf70be23c9
∇²f_d = Symbolics.hessian(f_d(x₁, x₂), [x₁, x₂])

# ╔═╡ 09203146-bf72-491c-82f3-41c686a7e829
H_d! = eval(build_function(∇²f_d, [x₁, x₂])[2])

# ╔═╡ d51eba6e-809e-4c41-991d-5d90c65182b4
md"""
### Runs
"""

# ╔═╡ 76226106-ff98-41ba-94f8-63d02a563b48
newt_d_1 = optimize_newton(x -> f_d(x[1], x[2]), g_d!, H_d!, [0.75,-1.25])

# ╔═╡ 26b48bdf-8532-4e94-a2dc-8a4536fc45a2
md"""
	non cambia niente e ci piace, vuol dire che linesearch non interferisce con newton quando newton è forte
"""

# ╔═╡ 047ae524-591d-4e20-bcd5-28e92e9f560f
newt_d_2 = optimize_newton(x -> f_d(x[1], x[2]), g_d!, H_d!, [0.75,-1.25], linesearch=backtracking)

# ╔═╡ 765b22a5-2edd-4d16-81d6-a9665f1e5c1f
bfgs_d_1 = optimize_bfgs(x -> f_d(x[1], x[2]), g_d!, I(2), [0.75,-1.25], linesearch=backtracking)

# ╔═╡ 15fdb3ce-2d5b-4b94-a462-5c438ff94e8e
newt_d_3 = optimize_newton(x -> f_d(x[1], x[2]), g_d!, H_d!, [0.,0.])

# ╔═╡ c44fb844-61c8-44fb-aad0-dbbc9f1b554a
bfgs_d_2 = optimize_bfgs(x -> f_d(x[1], x[2]), g_d!, I(2), [0.,0.], linesearch=backtracking)

# ╔═╡ bc95da07-ceed-45d3-8826-42d29fe783ba
trrg_d_1 = optimize_trustregion(x -> f_d(x[1], x[2]), g_d!, H_d!, [0.,0.])

# ╔═╡ 65b9915c-dcc8-4ef7-bde3-423c709afc5e
plot_optimization(f_d, (0.695884386,−1.34794219), newt_d_1, bfgs_d_1,  xlim=(0.6,0.8), ylim=(-1.4,-1.2))

# ╔═╡ 6a2eb1cf-4e39-4cf0-807c-3080877cdc27
plot_optimization(f_d, (0.695884386,−1.34794219), bfgs_d_2, trrg_d_1, xlim=(-0.1,1), ylim=(-1.5,0.1))

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"

[compat]
Plots = "~1.40.1"
StaticArrays = "~1.9.2"
Symbolics = "~5.16.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.0"
manifest_format = "2.0"
project_hash = "e1882a21c78a2322bebbbb71b52a2ef8e7418157"

[[deps.ADTypes]]
git-tree-sha1 = "41c37aa88889c171f1300ceac1313c06e891d245"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "0.2.6"

[[deps.AbstractTrees]]
git-tree-sha1 = "faa260e4cb5aba097a73fab382dd4b5819d8ec8c"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.4"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "0fb305e0253fd4e833d486914367a2ee2c2e78d0"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.0.1"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra", "Requires", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "bbec08a37f8722786d87bedf84eae19c020c4efa"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.7.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Bijections]]
git-tree-sha1 = "c9b163bd832e023571e86d0b90d9de92a9879088"
uuid = "e2ed5e7c-b2de-5872-ae92-c73ca462fb04"
version = "0.1.6"

[[deps.BitFlags]]
git-tree-sha1 = "2dc09997850d68179b69dafb58ae806167a32b1b"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.8"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9e2a6b69137e6969bab0152632dcb3bc108c8bdd"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+1"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "ab79d1f9754a3988a7792caec43bfdc03996020f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.21.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "59939d8a997469ee05c4b4944560a820f9ba0d73"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.4"

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
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonSolve]]
git-tree-sha1 = "0eee5eb66b1cf62cd6ad1b460238e60e4b09400c"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.4"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "75bd5b6fc5089df449b5d35fa501c846c9b6549b"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.12.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+1"

[[deps.CompositeTypes]]
git-tree-sha1 = "02d2316b7ffceff992f3096ae48c7829a8aa0638"
uuid = "b152e2b5-7a66-4b01-a709-34e65c35f657"
version = "0.1.3"

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
weakdeps = ["IntervalSets", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "ac67408d9ddf207de5cfa9a97e114352430f01ed"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.16"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "7c302d7a5fec5214eb8a5a4c466dcf7a51fcf169"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.107"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.DomainSets]]
deps = ["CompositeTypes", "IntervalSets", "LinearAlgebra", "Random", "StaticArrays", "Statistics"]
git-tree-sha1 = "51b4b84d33ec5e0955b55ff4b748b99ce2c3faa9"
uuid = "5b8099bc-c8ec-5219-889f-1d9e522a28bf"
version = "0.6.7"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.DynamicPolynomials]]
deps = ["Future", "LinearAlgebra", "MultivariatePolynomials", "MutableArithmetics", "Pkg", "Reexport", "Test"]
git-tree-sha1 = "fea68c84ba262b121754539e6ea0546146515d4f"
uuid = "7c1d4256-1411-5781-91ec-d7bc3513ac07"
version = "0.5.3"

[[deps.EnumX]]
git-tree-sha1 = "bdb1942cd4c45e3c678fd11569d5cccd80976237"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.4"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8e9441ee83492030ace98f9789a654a6d0b1f643"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+0"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "dcb08a0d93ec0b1cdc4af184b26b591e9695423a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.10"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4558ab818dcceaab612d1bb8c19cee87eda2b83c"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.5.0+0"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

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

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random"]
git-tree-sha1 = "5b93957f6dcd33fc343044af3d48c215be2562f1"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.9.3"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

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

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "cf0fe81336da9fb90944683b8c41984b08793dad"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.36"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

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

[[deps.FunctionWrappers]]
git-tree-sha1 = "d62485945ce5ae9c0c48f124a84998d755bae00e"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.3"

[[deps.FunctionWrappersWrappers]]
deps = ["FunctionWrappers"]
git-tree-sha1 = "b104d487b34566608f8b4e1c39fb0b10aa279ff8"
uuid = "77dc65aa-8811-40c2-897b-53d922fa7daf"
version = "0.1.3"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "ff38ba61beff76b8f4acad8ab0c97ef73bb670cb"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.9+0"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "ec632f177c0d990e64d955ccc1b8c04c485a0950"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.6"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "3458564589be207fa6a77dbbf8b97674c9836aab"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.2"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "77f81da2964cc9fa7c0127f941e8bce37f7f1d70"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.2+0"

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

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "f218fe3736ddf977e0e772bc9a586b2383da2685"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.23"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IntervalSets]]
git-tree-sha1 = "581191b15bcb56a2aa257e9c160085d0f128a380"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.9"
weakdeps = ["Random", "Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsStatisticsExt = "Statistics"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

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
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d986ce2d884d49126836ea94ed5bfb0f12679713"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.7+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.LabelledArrays]]
deps = ["ArrayInterface", "ChainRulesCore", "ForwardDiff", "LinearAlgebra", "MacroTools", "PreallocationTools", "RecursiveArrayTools", "StaticArrays"]
git-tree-sha1 = "d1f981fba6eb3ec393eede4821bca3f2b7592cd4"
uuid = "2ee39098-c373-598a-b85f-a56591580800"
version = "1.15.1"

[[deps.LambertW]]
git-tree-sha1 = "c5ffc834de5d61d00d2b0e18c96267cffc21f648"
uuid = "984bce1d-4616-540c-a9ee-88d1112d94c9"
version = "0.4.6"

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

[[deps.Lazy]]
deps = ["MacroTools"]
git-tree-sha1 = "1370f8202dac30758f3c345f9909b97f53d87d3f"
uuid = "50d2b5c4-7a5e-59d5-8109-a42b560f39c0"
version = "0.15.1"

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
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

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

[[deps.MultivariatePolynomials]]
deps = ["ChainRulesCore", "DataStructures", "LinearAlgebra", "MutableArithmetics"]
git-tree-sha1 = "769c9175942d91ed9b83fa929eee4fe6a1d128ad"
uuid = "102ac46a-7ee4-5c85-9060-abc95bfdeaa3"
version = "0.5.4"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "806eea990fb41f9b36f1253e5697aa645bf6a9f8"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.4.0"

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
git-tree-sha1 = "60e3045590bd104a16fefb12836c00c0ef8c7f8c"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.13+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

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

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "949347156c25054de2db3b166c52ac4728cbad65"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.31"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

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
git-tree-sha1 = "862942baf5663da528f66d24996eb6da85218e76"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.0"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "c4fa93d7d66acad8f6f4ff439576da9d2e890ee0"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.1"

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

[[deps.PreallocationTools]]
deps = ["Adapt", "ArrayInterface", "ForwardDiff"]
git-tree-sha1 = "64bb68f76f789f5fe5930a80af310f19cdafeaed"
uuid = "d236fae5-4411-538c-8e31-a6e3d9e00b46"
version = "0.4.17"

    [deps.PreallocationTools.extensions]
    PreallocationToolsReverseDiffExt = "ReverseDiff"

    [deps.PreallocationTools.weakdeps]
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"

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

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9b23c31e76e333e6fb4c1595ae6afa74966a729e"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.9.4"

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

[[deps.RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterface", "DocStringExtensions", "GPUArraysCore", "IteratorInterfaceExtensions", "LinearAlgebra", "RecipesBase", "SparseArrays", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface", "Tables"]
git-tree-sha1 = "2bd309f5171a628efdf5309361cd8a779b9e63a9"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "3.8.0"

    [deps.RecursiveArrayTools.extensions]
    RecursiveArrayToolsFastBroadcastExt = "FastBroadcast"
    RecursiveArrayToolsForwardDiffExt = "ForwardDiff"
    RecursiveArrayToolsMeasurementsExt = "Measurements"
    RecursiveArrayToolsMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    RecursiveArrayToolsReverseDiffExt = ["ReverseDiff", "Zygote"]
    RecursiveArrayToolsTrackerExt = "Tracker"
    RecursiveArrayToolsZygoteExt = "Zygote"

    [deps.RecursiveArrayTools.weakdeps]
    FastBroadcast = "7034ab61-46d4-4ed7-9d0f-46aef9175898"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

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

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.RuntimeGeneratedFunctions]]
deps = ["ExprTools", "SHA", "Serialization"]
git-tree-sha1 = "6aacc5eefe8415f47b3e34214c1d79d2674a0ba2"
uuid = "7e49a35a-f44a-4d26-94aa-eba1b4ca6b47"
version = "0.5.12"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SciMLBase]]
deps = ["ADTypes", "ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "EnumX", "FillArrays", "FunctionWrappersWrappers", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "Markdown", "PrecompileTools", "Preferences", "Printf", "RecipesBase", "RecursiveArrayTools", "Reexport", "RuntimeGeneratedFunctions", "SciMLOperators", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface", "Tables", "TruncatedStacktraces"]
git-tree-sha1 = "75bae786dc8b07ec3c2159d578886691823bcb42"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "2.23.1"

    [deps.SciMLBase.extensions]
    SciMLBaseChainRulesCoreExt = "ChainRulesCore"
    SciMLBasePartialFunctionsExt = "PartialFunctions"
    SciMLBasePyCallExt = "PyCall"
    SciMLBasePythonCallExt = "PythonCall"
    SciMLBaseRCallExt = "RCall"
    SciMLBaseZygoteExt = "Zygote"

    [deps.SciMLBase.weakdeps]
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    PartialFunctions = "570af359-4316-4cb7-8c74-252c00c2016b"
    PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
    PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
    RCall = "6f49c342-dc21-5d91-9882-a32aef131414"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.SciMLOperators]]
deps = ["ArrayInterface", "DocStringExtensions", "Lazy", "LinearAlgebra", "Setfield", "SparseArrays", "StaticArraysCore", "Tricks"]
git-tree-sha1 = "51ae235ff058a64815e0a2c34b1db7578a06813d"
uuid = "c0aeaf25-5076-4817-a8d5-81caf7dfa961"
version = "0.3.7"

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
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e2cfc4012a19088254b3950b85c3c1d8882d864d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "7b0e9c14c624e435076d19aea1e5cbdec2b9ca37"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.2"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

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

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f625d686d5a88bcd2b15cd81f18f98186fdc0c9a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.0"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.SymbolicIndexingInterface]]
git-tree-sha1 = "b3103f4f50a3843e66297a2456921377c78f5e31"
uuid = "2efcf032-c050-4f8e-a9bb-153293bab1f5"
version = "0.3.5"

[[deps.SymbolicUtils]]
deps = ["AbstractTrees", "Bijections", "ChainRulesCore", "Combinatorics", "ConstructionBase", "DataStructures", "DocStringExtensions", "DynamicPolynomials", "IfElse", "LabelledArrays", "LinearAlgebra", "MultivariatePolynomials", "NaNMath", "Setfield", "SparseArrays", "SpecialFunctions", "StaticArrays", "SymbolicIndexingInterface", "TimerOutputs", "Unityper"]
git-tree-sha1 = "849b1dfb1680a9e9f2c6023f79a49b694fb6d0da"
uuid = "d1185830-fcd6-423d-90d6-eec64667417b"
version = "1.5.0"

[[deps.Symbolics]]
deps = ["ArrayInterface", "Bijections", "ConstructionBase", "DataStructures", "DiffRules", "Distributions", "DocStringExtensions", "DomainSets", "DynamicPolynomials", "IfElse", "LaTeXStrings", "LambertW", "Latexify", "Libdl", "LinearAlgebra", "LogExpFunctions", "MacroTools", "Markdown", "NaNMath", "PrecompileTools", "RecipesBase", "Reexport", "Requires", "RuntimeGeneratedFunctions", "SciMLBase", "Setfield", "SparseArrays", "SpecialFunctions", "StaticArrays", "SymbolicIndexingInterface", "SymbolicUtils"]
git-tree-sha1 = "ab1785cd8cbfa6cc26af3efa491fd241aa69855e"
uuid = "0c5d862f-8b57-4792-8d23-62f2024744c7"
version = "5.16.1"

    [deps.Symbolics.extensions]
    SymbolicsForwardDiffExt = "ForwardDiff"
    SymbolicsGroebnerExt = "Groebner"
    SymbolicsPreallocationToolsExt = ["ForwardDiff", "PreallocationTools"]
    SymbolicsSymPyExt = "SymPy"

    [deps.Symbolics.weakdeps]
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Groebner = "0b43b601-686d-58a3-8a1c-6623616c7cd4"
    PreallocationTools = "d236fae5-4411-538c-8e31-a6e3d9e00b46"
    SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "cb76cf677714c095e535e3501ac7954732aeea2d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.11.1"

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

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "f548a9e9c490030e545f72074a41edfd0e5bcdd7"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.23"

[[deps.TranscodingStreams]]
git-tree-sha1 = "54194d92959d8ebaa8e26227dbe3cdefcdcd594f"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.10.3"
weakdeps = ["Random", "Test"]

    [deps.TranscodingStreams.extensions]
    TestExt = ["Test", "Random"]

[[deps.Tricks]]
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

[[deps.TruncatedStacktraces]]
deps = ["InteractiveUtils", "MacroTools", "Preferences"]
git-tree-sha1 = "ea3e54c2bdde39062abf5a9758a23735558705e1"
uuid = "781d530d-4396-4725-bb49-402e4bee1e77"
version = "1.4.0"

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

[[deps.Unityper]]
deps = ["ConstructionBase"]
git-tree-sha1 = "25008b734a03736c41e2a7dc314ecb95bd6bbdb0"
uuid = "a7c27f48-0311-42f6-a7f8-2c11e75eb415"
version = "0.1.6"

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
git-tree-sha1 = "93f43ab61b16ddfb2fd3bb13b3ce241cafb0e6c9"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.31.0+0"

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
# ╠═6e15ed80-1938-4ea9-82f8-a7d0b8dfd6ce
# ╠═45ad341c-765c-40e1-a726-8a575dc3ab9a
# ╟─61498356-cf52-11ee-3c16-038f22b043c9
# ╟─7435d8e2-dd5c-49bd-981e-46ee8ece3313
# ╠═ed785d1d-54e9-49df-b45f-dffa161d978c
# ╟─52387950-56cc-414e-a434-8480757fdd45
# ╟─3f060cb3-3465-45ae-9a9a-4ec3f3628cf2
# ╠═fa77e47e-6876-42d5-ae9d-9c68e9afd174
# ╟─27a151a4-1028-4baf-b4b4-138fa5ce9040
# ╟─fca61f4d-948a-4a05-a387-16ebe84ab56d
# ╠═41e06b52-3b8c-47be-9a75-fdf3d708def4
# ╟─6698003b-240b-4f1b-b1e4-8e70107c4150
# ╠═b4c6e46a-f47e-4ce2-8cd9-c6229be6fdf4
# ╟─35f73a6e-ada1-412a-a770-175427d1e443
# ╟─3220a80c-bd63-4683-99db-5a53901c9b0c
# ╠═e5d9ac0a-7cfc-44ab-b29f-fb8d2f1f69c7
# ╠═54500345-520f-479c-88ce-f7d1a373389f
# ╟─f8dc799a-d2ef-43ae-aa8b-5a43b094d7c3
# ╟─b8e94ea4-4d08-4e64-a134-e2562e7d007e
# ╠═2b253872-ca18-4f81-9a90-565d1503576d
# ╟─96e08046-c44d-421c-9784-72a97ff8fbf8
# ╟─81742d93-d761-4ec1-ab3a-571711130fc6
# ╠═57c33dc0-1e46-4c33-a432-550307367839
# ╠═85a9fbca-7866-4c7f-899e-d6738a96596a
# ╟─cdfc565c-c18b-4e8e-aa17-dfb9483e9eee
# ╠═afc86c35-5c79-4f26-881d-45fcaea8dbb1
# ╟─c156f0f0-1c1f-4c19-bb92-8cd3db60215d
# ╠═d30f8a04-5fde-4b1c-933c-577229b8de77
# ╟─d73ae180-2d77-4e7f-a3fb-c9a0abaa9c51
# ╠═b86e5b21-7b90-4111-b6da-eddbf7402bf2
# ╠═7074f6d2-94c1-4440-9a44-7936db617bb3
# ╠═f975f7e7-1790-47b3-a0ef-014df5f5a9fb
# ╟─c95a8234-9729-451b-9745-1abf35819000
# ╟─d629443a-45a2-4abe-80fb-0b233863c960
# ╠═5edc0250-8d8c-4c94-91ab-08a813845668
# ╟─d2b6d03b-45c4-4941-b18e-c7e3fc451889
# ╠═c5a4de9b-e452-4428-83ae-6f9209e22cf9
# ╟─cea16f18-a943-47c2-8803-dc877262ae8f
# ╠═e446999e-490d-4afc-aaf7-9536d3ecd1f0
# ╟─3f27c70f-8ad7-42bb-b93d-e7e7e586e095
# ╠═12d943f9-83b3-44e2-a0f5-f7becb7b614f
# ╟─d6007def-8349-4678-aa6c-4b3c771b3c15
# ╠═29a1ba8e-e27a-45b4-a1ac-dcc2eefa05eb
# ╟─51a11fce-5808-4479-b79f-fe8ae9310c3e
# ╠═5a78ea6c-b4ce-4d4f-9719-e09e9c1f36fd
# ╟─4fca73b7-be7c-427e-9b22-0c60f50279e4
# ╠═e11ee2fb-af2b-4daf-8c12-4e625f891f84
# ╟─f3acf4de-592e-4709-a687-ad66e355360c
# ╠═40df2ad0-0189-474c-bb71-3e00effae4b1
# ╟─95212305-bd5b-4c47-87dc-116231452373
# ╠═a0d5ab72-f12d-4a75-93cd-40365b6bd4df
# ╟─23ab718b-5237-4cce-927e-ab29ec98deae
# ╟─0b6ed726-232f-484b-9e20-28bfdb812a67
# ╠═015c5739-0e4a-426a-a6f9-376213da2424
# ╠═10918712-2518-439e-9597-b4341e938ebd
# ╠═89e8b816-6a52-461c-b76d-c72b94c29974
# ╠═45081330-abf2-4ff3-bd32-9197446d3962
# ╠═36afaf29-e07a-450d-a7e0-b9c057c2a698
# ╟─376b44e1-7227-4e83-a558-55780235c568
# ╠═32fd9af2-16a6-44bd-8382-9f822395d3e6
# ╠═299e411b-6ccc-4f12-9d79-2b7275547e52
# ╠═0c0114d3-ab4a-4631-b8a7-3fc0ec922859
# ╠═207e0b4c-3a6b-41b8-a760-0f02e048a6df
# ╟─b46d53a0-d52a-463c-951f-e5cb7a4e86fb
# ╠═f59d3dac-21ad-45bc-9933-8182eebd5ce4
# ╠═925503d2-4262-4da0-b00a-c970ebde0318
# ╠═bfffcc68-3719-40ef-87b7-ff944d9b6ae2
# ╠═ba043839-e13d-475a-a93e-b170af23327d
# ╠═851793c3-b9fd-44ee-b96f-0f0734fae4d9
# ╠═8fcac8e1-7059-4547-beae-7d1d5ed44420
# ╠═3adc0e72-05e1-4c8b-8c7a-d2a176d18d49
# ╟─1c6f05b3-a1c0-4294-9316-a1e0049550b2
# ╟─1f83f39f-2e67-4bc4-bc23-c5f5093fd7aa
# ╟─cc42fc4f-c0ab-461e-822f-07d9b36afc79
# ╠═059f8000-ea4e-4338-b75e-89f136794c67
# ╟─94d79163-fa2a-4183-8802-213f605dcf59
# ╠═caaf83db-4701-45a1-9649-e34576470735
# ╟─f4e6494d-6c74-4a5f-ac8a-8ca7b0d2243d
# ╠═72ab53df-83e8-4ecf-a811-c1e8e848d249
# ╟─7b4cfc1c-17f6-44c1-a009-102e9f60d044
# ╠═8853d92c-a774-40ee-a71f-4c6b7511686d
# ╟─144103b9-24b2-4772-8d2e-119ff1a73c81
# ╠═91f47c4c-e334-4741-8105-1a6b14cf0e09
# ╟─07fe8486-decd-43ea-bdbb-b97551edeb3a
# ╠═424974c8-a2e6-447b-b7f5-29e28572314d
# ╟─8f15f324-0cff-4737-ad0e-895173e8914e
# ╠═bdb8361d-e904-4857-bd96-16e609c4fe85
# ╟─e3a3d369-70ff-4f9b-9daa-5df76be0b071
# ╠═7f792dcd-e1f6-40c5-bf76-55bc640ceb81
# ╟─6eafa5e6-5eb6-4c4c-889a-85093320ea26
# ╠═7b3cd1cb-e2b6-408d-b981-0ffbb3174822
# ╟─2b727c28-62a5-4290-82e8-c486467b98b3
# ╠═a02f8144-9b20-4caa-aadc-0fe00a8709d9
# ╟─b0e20e6b-55c8-498c-ba4b-3393624e49ae
# ╠═7a00d2c7-6dc1-4795-8328-a7696a32cc56
# ╠═e12fdb33-d126-415b-b195-f75a2697f5e3
# ╟─775a3f3b-94aa-4502-8b99-57824fa8a805
# ╠═3fbe3908-eee9-440b-993c-67e8a84d10e9
# ╠═b22fa713-2fe9-475c-97b8-688329d6969d
# ╠═9b835cfe-2206-4deb-b958-d3672e1a77fe
# ╠═c7220592-fc91-4c1f-9c71-5dab5475c32d
# ╠═bef5fb16-a753-4375-90ef-e4bf70be23c9
# ╠═09203146-bf72-491c-82f3-41c686a7e829
# ╟─d51eba6e-809e-4c41-991d-5d90c65182b4
# ╠═76226106-ff98-41ba-94f8-63d02a563b48
# ╟─26b48bdf-8532-4e94-a2dc-8a4536fc45a2
# ╠═047ae524-591d-4e20-bcd5-28e92e9f560f
# ╠═765b22a5-2edd-4d16-81d6-a9665f1e5c1f
# ╠═15fdb3ce-2d5b-4b94-a462-5c438ff94e8e
# ╠═c44fb844-61c8-44fb-aad0-dbbc9f1b554a
# ╠═bc95da07-ceed-45d3-8826-42d29fe783ba
# ╠═65b9915c-dcc8-4ef7-bde3-423c709afc5e
# ╠═6a2eb1cf-4e39-4cf0-807c-3080877cdc27
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
