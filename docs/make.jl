using LatentSDEBayes
using Documenter

DocMeta.setdocmeta!(LatentSDEBayes, :DocTestSetup, :(using LatentSDEBayes); recursive=true)

makedocs(;
    modules=[LatentSDEBayes],
    authors="Ron-27 <ronmaor@sas.upenn.edu> and contributors",
    sitename="LatentSDEBayes.jl",
    format=Documenter.HTML(;
        canonical="https://Ron-279.github.io/LatentSDEBayes.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Ron-279/LatentSDEBayes.jl",
    devbranch="main",
)
