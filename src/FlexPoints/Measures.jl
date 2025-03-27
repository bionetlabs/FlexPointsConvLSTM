module Measures

export cf, rmse, nrmse, minrmse, prd, nprd, qs, nqs

using Statistics

using AKGECG.FlexPoints

"Compression Factor (CF)"
function cf(inputsize::Integer, outputsize::Integer)::Float64
    @assert inputsize > 0
    @assert outputsize > 0
    convert(Float64, inputsize / outputsize)
end

"Compression Factor (CF)"
function cf(data::Vector{<:Number}, samples::Vector{<:Number})
    cf(length(data), length(samples))
end

"Root Mean Square Error (RMSE)"
function rmse(data::Vector{Float64}, samples::Vector{Float64})::Float64
    @assert !isempty(data)
    @assert !isempty(samples)

    errorsum = 0.0
    for (i, yi_approx) in enumerate(samples)
        yi = data[i]
        errorsum += (yi - yi_approx)^2
    end

    √(errorsum / length(data))
end

"Normalized Root Mean Square Error (NRMSE)"
function nrmse(data::Vector{Float64}, samples::Vector{Float64})::Float64
    @assert !isempty(data)
    @assert !isempty(samples)

    numerator = 0.0
    denominator = 0.0
    for (i, yi_approx) in enumerate(samples)
        yi = data[i]
        numerator += (yi - yi_approx)^2
        denominator += yi^2
    end

    √(numerator / denominator)
end

"Mean Independent Normalized Root Mean Square Error (MINRMSE)"
function minrmse(data::Vector{Float64}, samples::Vector{Float64})::Float64
    @assert !isempty(data)
    @assert !isempty(samples)

    ymean = mean(data)

    numerator = 0.0
    denominator = 0.0
    for (i, yi_approx) in enumerate(samples)
        yi = data[i]
        numerator += (yi - yi_approx)^2
        denominator += (yi - ymean)^2
    end

    √(numerator / denominator)
end

"Percentage Root mean square Difference (PRD)"
function prd(data::Vector{Float64}, samples::Vector{Float64})::Float64
    nrmse(data, samples) * 100.0
end

"Normalized Percentage Root mean square Difference (NPRD)"
function nprd(data::Vector{Float64}, samples::Vector{Float64})::Float64
    minrmse(data, samples) * 100.0
end

"Quality Score (QS)"
function qs(data::Vector{Float64}, samples::Vector{Float64})::Float64
    cf(data, samples) / prd(data, samples)
end

"Quality Score (QS)"
function qs(cf::Float64, prd::Float64)::Float64
    cf / prd
end

"Normalized Quality Score (NQS)"
function nqs(data::Vector{Float64}, samples::Vector{Float64})::Float64
    cf(data, samples) / nprd(data, samples)
end

"Normalized Quality Score (NQS)"
function nqs(cf::Float64, nprd::Float64)::Float64
    cf / nprd
end


end