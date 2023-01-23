using ActiveSpaceSolvers
using Test

@testset "ActiveSpaceSolvers.jl" begin
    include("test_FCI.jl")
    include("test_h4.jl")
    include("test_ras_lu.jl")
    include("test_ras_rdms.jl")
end
