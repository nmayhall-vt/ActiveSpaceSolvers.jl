using ActiveSpaceSolvers
using Test

@testset "ActiveSpaceSolvers.jl" begin
    include("test_FCI.jl")
    include("test_h4.jl")
end
