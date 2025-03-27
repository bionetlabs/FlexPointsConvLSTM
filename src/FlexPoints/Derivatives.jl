module Derivatives

export DerivativesData, DerivativesSelector, topderivative, ∂, ∂zeros

using Parameters

using AKGECG.FlexPoints

@with_kw mutable struct DerivativesData
    ∂1data::Union{Vector{Float64},Nothing} = nothing
    ∂2data::Union{Vector{Float64},Nothing} = nothing
    ∂3data::Union{Vector{Float64},Nothing} = nothing
    ∂4data::Union{Vector{Float64},Nothing} = nothing
end

@with_kw mutable struct DerivativesSelector
    ∂1::Bool = false
    ∂2::Bool = false
    ∂3::Bool = true
    ∂4::Bool = false
end

function topderivative(selector::DerivativesSelector)
    if selector.∂4
        4
    elseif selector.∂3
        3
    elseif selector.∂2
        2
    elseif selector.∂1
        1
    else
        0
    end
end


"""
Finds derivative of the sampled function.
Zeros specifies how many values of derivative will be set to zero at the
beginning and at the end of the function.
Derivative is computed from function values before and after sample point
divided by twice the step size.
"""
function ∂(data::Points2D, boundingzeros::Integer)::Vector{Float64}
    lastindex = datalen = length(data)
    @assert datalen >= 3
    @assert 1 <= boundingzeros <= datalen ÷ 2 - 1

    derivatives = zeros(datalen)

    for i in 1:(datalen-2boundingzeros)
        offest = i + boundingzeros
        Δx = x(data, offest + 1) - x(data, offest - 1)
        Δy = y(data, offest + 1) - y(data, offest - 1)
        derivatives[offest] = Δy / Δx
    end

    if datalen >= boundingzeros + 1
        for i in 1:boundingzeros
            derivatives[i] = derivatives[boundingzeros+1]
        end
    end

    for i in (lastindex-boundingzeros):(datalen)
        derivatives[i] = derivatives[lastindex-boundingzeros]
    end

    derivatives
end

"""
Finds derivative of the sampled function.
Derivative is computed from function values before and after sample point
divided by twice the step size. So, `outputsize = inputsize - 2`.
"""
function ∂(data::Points2D)::Vector{Float64}
    @assert datalen >= 3

    derivatives = zeros(datalen - 2)

    for i in 2:(datalen-1)
        offest = i
        Δx = x(data, offest + 1) - x(data, offest - 1)
        Δy = y(data, offest + 1) - y(data, offest - 1)
        derivatives[offest-1] = Δy / Δx
    end

    derivatives
end

"""
Returns indices of x-intercepts
"""
function ∂zeros(derivatives::Vector{Float64})::Vector{Int}
    lastindex = derivativeslen = length(derivatives)
    mx = zeros(derivativeslen - 1)

    for i in 1:(lastindex-1)
        mx[i] = derivatives[i] * derivatives[i+1]
    end

    output = map(ix -> ix[1], filter(ix -> ix[2] <= 0.0, collect(enumerate(mx))))

    outputtail = if !isempty(output)
        if first(output) != 1
            newarray = [1]
            append!(newarray, output)
            newarray
        else
            output
        end
    else
        [1]
    end

    if last(outputtail) != lastindex
        push!(outputtail, lastindex)
    end

    outputtail
end

end