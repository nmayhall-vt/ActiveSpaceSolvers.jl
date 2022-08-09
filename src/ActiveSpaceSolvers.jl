module ActiveSpaceSolvers

abstract type Problem end

abstract type Solver end

include("oldCI/StringCI.jl");

import .StringCI.FCIProblem

end
