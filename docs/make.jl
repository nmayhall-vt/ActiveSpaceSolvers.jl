using ActiveSpaceSolvers
using Documenter

DocMeta.setdocmeta!(ActiveSpaceSolvers, :DocTestSetup, :(using ActiveSpaceSolvers); recursive=true)

makedocs(;
    modules=[ActiveSpaceSolvers],
    authors="Nick Mayhall <nmayhall@vt.edu> and contributors",
    repo="https://github.com/nmayhall-vt/ActiveSpaceSolvers.jl/blob/{commit}{path}#{line}",
    sitename="ActiveSpaceSolvers.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://nmayhall-vt.github.io/ActiveSpaceSolvers.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/nmayhall-vt/ActiveSpaceSolvers.jl",
    devbranch="main",
)
