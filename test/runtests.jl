using ActiveSpaceSolvers
using Test

@testset "ActiveSpaceSolvers.jl" begin
    include("test_FCI.jl")
    include("test_h4.jl")
    include("RASCI/test_RASCI.jl")
    include("RASCI/test_ras_rdms.jl")
    include("RASCI/test_TDMS.jl")
end
