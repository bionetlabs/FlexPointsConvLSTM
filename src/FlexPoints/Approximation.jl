module Approximation

export linapprox, linregression, endpointsgradient,
    endpointsangle, midpointerror, endpoints2gradientintercept

using AKGECG.FlexPoints.Types

"Linear approximation"
function linapprox(data::Points2D, targetx::Real)
    @assert !isempty(data)
    @assert targetx >= data[1][1]
    @assert targetx <= data[end][1]

    for (i, point) in enumerate(data)
        x1, y1 = point
        if targetx == x1
            return y1
        elseif targetx < x1
            x2, y2 = data[i-1]
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            return slope * targetx + intercept
        end
    end
end

"""
Takes 2D data represented by x and y vectors 
and returns gradient m and y-intercept c
"""
function linregression(x, y)::Tuple{Number,Number}
    n = length(y)
    sx = sum(x)
    sy = sum(y)
    sx2 = x' * x
    sxy = x' * y
    m = (n * sxy - sx * sy) / (n * sx2 - sx^2)
    c = (sy * sx2 - sx * sxy) / (n * sx2 - sx^2)
    m, c
end

function endpointsgradient(data::Points2D)::Float64
    x1, y1 = first(data)
    x2, y2 = last(data)
    (y2 - y1) / (x2 - x1)
end

function endpoints2gradientintercept(data::Points2D)::Tuple{Float64,Float64}
    x1, y1 = first(data)
    x2, y2 = last(data)
    gradient = (y2 - y1) / (x2 - x1)
    intercept = y1 - gradient * x1
    gradient, intercept
end

function endpointsangle(data::Points2D)::Float64
    gradient = endpointsgradient(data)
    atan(gradient)
end

function midpointerror(data::Points2D, target::Tuple{Float64,Float64})::Float64
    gradient, intercept = endpoints2gradientintercept(data)
    reciprocalgradient = -1 / gradient
    reciprocalintercept = target[2] - reciprocalgradient * target[1]
    crosspointx = (reciprocalintercept - intercept) / (gradient - reciprocalgradient)
    crosspointy = gradient * crosspointx + intercept
    error = sqrt((target[1] - crosspointx)^2 + (target[2] - crosspointy)^2)
    error
end

end