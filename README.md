# NAOAssignments

This code base is using the [Julia Language](https://julialang.org/) and
[DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> NAOAssignments

It is authored by Marco Riggirello, Francesco Vaselli.

To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box, including correctly finding local paths.

You may notice that most scripts start with the commands:
```julia
using DrWatson
@quickactivate "NAOAssignments"
```
which auto-activate the project and enable local path handling from DrWatson.

### Running the Pluto notebooks

To run Pluto notebooks, simply use these commands:
```julia
using DrWatson
@quickactivate "NAOAssignments"
using Pluto
Pluto.run("path/to/your/notebook.jl")
```
And then edit the blocks as usual. Remember to save the exported pdf in the `reports` directory.

