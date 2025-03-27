module Filters

export MFilterParameters, mfilter, NoiseFilterParameters, noisefilter

using Parameters
using DataStructures
using Statistics
using LinearAlgebra

using AKGECG.FlexPoints

@with_kw mutable struct NoiseFilterParameters
    data::Bool = true
    derivatives::Bool = false
    filtersize::Unsigned = 7
end

@with_kw mutable struct MFilterParameters
    m1::Float64 = 1.5e-4
    m2::Float64 = 1.5e-4
    m3::Float64 = 0.0
end

function signal2noise(x::Vector, y::Vector, errflimit::Float64)
    samples = length(y)
    senergy = y' * y # find energy of the error signal after linear fit
    xmean = mean(x)
    M = [(x .- xmean) .* (x .- xmean) (x .- xmean) ones(samples)]
    A = pinv(M) * y
    ferr = y - M * A # find error after quadratic fit
    fenergy = ferr' * ferr # find energy of the error signal after quadratic fit
    errf = (fenergy / senergy)^(1 / 2) # find error reduction ratio figure
    # errf = (fenergy / senergy)^(2^(-(7 - samples) / 2))

    # use different constant depending on the window reduction policy
    # use 0.7 for 80% certainty, 0.5 for 95% certainy and 0.4 for 1% certainty
    if errf < errflimit
        reducewindow = true
        errvalue = ferr[Int(ceil(samples / 2))]
        reducewindow, errvalue
    else
        reducewindow = false
        errvalue = 0
        reducewindow, errvalue
    end
end

function checknoise(
    data::Vector{Float64},
    index::Unsigned,
    filtersize::Unsigned,
    errflimit::Float64
)
    winners = data[(index-filtersize):(index+filtersize)]
    x = collect((index-filtersize):(index+filtersize))
    m, c = linregression(x, winners)
    smean = m * index + c
    error = winners .- (m .* x .+ c)

    # set errf certainty limit %0.7 20%, 0.5 5%, 0.35 1%
    reducewindow, errvalue = signal2noise(x, error, errflimit)
    if reducewindow
        if filtersize > 2
            filtersize = filtersize - 1
            filtersize, smean = checknoise(data, index, filtersize, errflimit)
        else
            smean = data[index]
        end
    else
        smean = smean - errvalue
    end

    filtersize, smean
end

function noisefilter(
    data::Vector{Float64},
    filtersize::Unsigned
)::Vector{Float64}
    smean = zeros(length(data))
    errflimit = 0
    for kk = 1:3
        currentsize = filtersize
        for index in (filtersize+1):(length(data)-filtersize)
            if currentsize < filtersize
                currentsize += 1
            end
            if kk == 1
                errflimit = 0.3
            elseif kk == 2
                errflimit = 0.58
            elseif kk == 3
                errflimit = 0.7
            end
            currentsize, smeanlocal = checknoise(data, index, currentsize, errflimit)
            smean[index] = smeanlocal
            # return smean # TODO: remove
        end
        smean[1:filtersize] .= smean[filtersize+1]
        smean[length(data)-filtersize:length(data)] .= smean[length(data)-filtersize-1]
    end
    smean
end

function mfilter(
    derivatives::DerivativesData,
    selector::DerivativesSelector,
    parameters::MFilterParameters
)::Vector{Int}
    @unpack ∂1data, ∂2data, ∂3data, ∂4data = derivatives
    @unpack ∂1, ∂2, ∂3, ∂4 = selector
    @unpack m1, m2, m3 = parameters

    datalength = length(∂1data)

    ∂1zeros = ∂zeros(∂1data)
    ∂2zeros = ∂zeros(∂2data)
    ∂3zeros = ∂zeros(∂3data)
    ∂4zeros = ∂zeros(∂4data)

    validindices = SortedSet{Int}([1, datalength])

    if ∂1
        for index in ∂1zeros
            if index <= 1 && index >= datalength
                push!(validindices, index)
            end
            ∂2zero_min = ∂2zero_max = nothing
            if index < datalength / 2
                ∂2zero_min = findlast(x -> x < index, ∂2zeros)
                ∂2zero_max = findfirst(x -> x >= index, ∂2zeros)
            else
                ∂2zero_min = findfirst(x -> x < index, reverse(∂2zeros))
                ∂2zero_max = findlast(x -> x >= index, reverse(∂2zeros))
            end
            if !isnothing(∂2zero_min) && abs(∂1data[∂2zero_min]) >= m1
                push!(validindices, index)
            end
            if !isnothing(∂2zero_max) && abs(∂1data[∂2zero_max]) >= m1
                push!(validindices, index)
            end
        end
    end

    if ∂2
        for index in ∂2zeros
            if index <= 1 && index >= datalength
                push!(validindices, index)
            end
            if abs(∂1data[index]) >= m1
                ∂3zero_min = ∂3zero_max = nothing
                if index < datalength / 2
                    ∂3zero_min = findlast(x -> x < index, ∂3zeros)
                    ∂3zero_max = findfirst(x -> x >= index, ∂3zeros)
                else
                    ∂3zero_min = findfirst(x -> x < index, reverse(∂3zeros))
                    ∂3zero_max = findlast(x -> x >= index, reverse(∂3zeros))
                end
                if !isnothing(∂3zero_min) && abs(∂2data[∂3zero_min]) >= m2
                    push!(validindices, index)
                end
                if !isnothing(∂3zero_max) && abs(∂2data[∂3zero_max]) >= m2
                    push!(validindices, index)
                end
            end
        end
    end

    if ∂3
        for index in ∂3zeros
            if index <= 1 && index >= datalength
                push!(validindices, index)
            end
            if abs(∂2data[index]) >= m2
                ∂4zero_min = ∂4zero_max = nothing
                if index < datalength / 2
                    ∂4zero_min = findlast(x -> x < index, ∂4zeros)
                    ∂4zero_max = findfirst(x -> x >= index, ∂4zeros)
                else
                    ∂4zero_min = findfirst(x -> x < index, reverse(∂4zeros))
                    ∂4zero_max = findlast(x -> x >= index, reverse(∂4zeros))
                end
                if !isnothing(∂4zero_min) && abs(∂3data[∂4zero_min]) >= m3
                    push!(validindices, index)
                end
                if !isnothing(∂4zero_max) && abs(∂3data[∂4zero_max]) >= m3
                    push!(validindices, index)
                end
            end
        end
    end

    if ∂4
        for index in ∂4zeros
            if index <= 1 && index >= datalength
                push!(validindices, index)
            end
            if abs(∂3data[index]) >= m3
                push!(validindices, index)
            end
        end
    end

    collect(validindices)
end

end