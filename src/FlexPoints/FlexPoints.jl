module FlexPoints

using Reexport

include("Types.jl")
@reexport using AKGECG.FlexPoints.Types

include("Data.jl")
@reexport using AKGECG.FlexPoints.Data

include("Approximation.jl")
@reexport using AKGECG.FlexPoints.Approximation

include("Derivatives.jl")
@reexport using AKGECG.FlexPoints.Derivatives

include("Filters.jl")
@reexport using AKGECG.FlexPoints.Filters

include("Algorithm.jl")
@reexport using AKGECG.FlexPoints.Algorithm

include("Measures.jl")
@reexport using AKGECG.FlexPoints.Measures

include("Benchmarks.jl")
@reexport using AKGECG.FlexPoints.Benchmarks

end
