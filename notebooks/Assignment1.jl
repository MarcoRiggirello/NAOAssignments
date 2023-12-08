### A Pluto.jl notebook ###
# v0.19.35

using Markdown
using InteractiveUtils

# ╔═╡ 71fb6b39-125d-481f-ad19-affa7c04d226
using LinearAlgebra

# ╔═╡ b8f31bda-95d7-11ee-29fe-9f2acb543f4a
md"""
# Numerical Analysis and Optimization Homework Project 1
"""

# ╔═╡ a5832cb4-5225-469c-96c6-52d08b68409a
md"""
This assignment is done using the Julia programming language. To easy the task, we load Julia's linear algebra standard library:
"""

# ╔═╡ 2d793d4f-6ca9-4551-b5ce-89d994bfc762
md"""
## Problem 1
"""

# ╔═╡ 7340ab2e-f583-48c4-a2d9-df1e0e3ec2c2
md"""
Here we define the function `lufact` that takes as input a squatre matrix and compute the non-pivoted LU factorization and its grow factor:
"""

# ╔═╡ 7010b1ae-7bb8-453a-ae37-718db11c74da
function lufact(A::AbstractMatrix)
	# checks that the given matrix is squared
	if size(A, 1) != size(A, 2)
		throw(ArgumentError("The matrix is not squared."))
	end
	n = size(A, 1)
	U = copy(A)
	L = UnitLowerTriangular(copy(A))
	γ = 0
	for k in 1:n-1
		for i in k+1:n
			L[i,k] = U[i,k]/U[i,i]
		end
		for j in k+1:n
			for i in k+1:n
				U[i,j] -= L[i,k]*U[k,j]
			end
		end
	end
	# lacks the gamma computation
	return L,U,γ
end

# ╔═╡ d486292d-5462-4e70-9f32-2e725c1f6d15
A = rand(8,8)

# ╔═╡ 02a5ce36-4273-47bc-bdb1-191596aa85d4
U,L,_=lufact(A)

# ╔═╡ 0c6a0565-fb81-4271-9d7a-5f2f87d0eca2
U*L - A

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.4"
manifest_format = "2.0"
project_hash = "ac1187e548c6ab173ac57d4e72da1620216bce54"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+0"
"""

# ╔═╡ Cell order:
# ╟─b8f31bda-95d7-11ee-29fe-9f2acb543f4a
# ╟─a5832cb4-5225-469c-96c6-52d08b68409a
# ╠═71fb6b39-125d-481f-ad19-affa7c04d226
# ╟─2d793d4f-6ca9-4551-b5ce-89d994bfc762
# ╟─7340ab2e-f583-48c4-a2d9-df1e0e3ec2c2
# ╠═7010b1ae-7bb8-453a-ae37-718db11c74da
# ╠═d486292d-5462-4e70-9f32-2e725c1f6d15
# ╠═02a5ce36-4273-47bc-bdb1-191596aa85d4
# ╠═0c6a0565-fb81-4271-9d7a-5f2f87d0eca2
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
