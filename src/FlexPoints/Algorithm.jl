module Algorithm

export flexpoints, FlexPointsParameters

using Parameters
using Statistics
import Polynomials
using DataStructures

using AKGECG.FlexPoints

@with_kw mutable struct FlexPointsParameters
    dselector::DerivativesSelector = DerivativesSelector()
    noisefilter::NoiseFilterParameters = NoiseFilterParameters()
    mfilter::MFilterParameters = MFilterParameters()
    mspp::Unsigned = 5 # minimum samples per period
    frequency::Unsigned = 360 # number of samples of signal per second 
    devv::Float64 = 1.0 # statistical measure for outliers in terms of standard deviation
    removeoutliers::Bool = false
    yresolution::Float64 = 0.021 # values with smaller Δ are considered as one point 
    polyapprox::UInt = 6
    polyapprox_yresolutionratio::Float64 = 2.0
end

function Base.copy(params::FlexPointsParameters)
    FlexPointsParameters(
        dselector=params.dselector,
        noisefilter=params.noisefilter,
        mfilter=params.mfilter,
        mspp=params.mspp,
        frequency=params.frequency,
        devv=params.devv,
        removeoutliers=params.removeoutliers,
        yresolution=params.yresolution,
        polyapprox=params.polyapprox,
        polyapprox_yresolutionratio=params.polyapprox_yresolutionratio,
    )
end

function paramsapprox(yresolution::Float64)::FlexPointsParameters
    FlexPointsParameters(
        dselector=DerivativesSelector(false, false, true, false),
        noisefilter=NoiseFilterParameters(false, false, 1),
        mfilter=MFilterParameters(0.0, 0.0, 0.0),
        mspp=5,
        frequency=360,
        devv=1.0,
        removeoutliers=false,
        yresolution=yresolution,
        polyapprox=1,
        polyapprox_yresolutionratio=2.0
    )
end

function polyapprox(
    data::Vector{Float64},
    points::Vector{Int},
    params::FlexPointsParameters
)::Vector{Int}
    degree = params.polyapprox
    if degree > 1
        newpoints = SortedSet(points)
        localparams = paramsapprox(params.yresolution / params.polyapprox_yresolutionratio)
        for i in 2:length(points)
            segmentindices = points[i-1]:points[i]
            if length(segmentindices) >= 10 # datalen ÷ 2 - 1 >= 4
                segmentxs = Float64.(1:length(segmentindices))
                segmentys = data[segmentindices]
                polyfit = Polynomials.fit(segmentxs, segmentys, degree)
                segmentdata = collect(zip(segmentxs, polyfit.(segmentxs)))
                _datafiltered, validpoints = flexpoints(segmentdata, localparams)
                push!(newpoints, (validpoints .+ (segmentindices[1] - 1))...)
            end
        end
        collect(newpoints)
    else
        points
    end
end

function flexpointsremoval(
    data::Vector{Float64},
    points::Vector{Int},
    params::FlexPointsParameters
)::Vector{Int}
    if params.removeoutliers
        points = removeoutliers(data, points, params)
    end
    points = yresolution(data, points, params)
    points = removelinear(data, points, params)
    points = findsinusoid(data, points, params)
    points = removelinear(data, points, params)
    points
end

function removeoutliers(
    data::Vector{Float64},
    points::Vector{Int},
    params::FlexPointsParameters
)::Vector{Int}
    @unpack mspp, frequency, devv = params
    blank = frequency / mspp # max size of the blank space - space on x axis without samples
    winmean = mean(data)
    winstd = std(data)
    highoutlier = winmean + devv * winstd
    lowoutlier = winmean - devv * winstd
    toremove = []

    current = 0 # current counts in which windows outliers are specified
    for i in 2:(length(points)-1)
        if ceil(points[i] / frequency) > current && 0 <= length(data) - (current + 1) * frequency
            currentdata = data[UInt(current * frequency + 1):UInt(current + 1 * frequency)]
            winmean = mean(currentdata)
            winstd = std(currentdata)
            highoutlier = winmean + devv * winstd
            lowoutlier = winmean - devv * winstd
            current = ceil(points[i] / frequency)
        end
        if abs(data[points[i]]) > highoutlier && (data[points[i]] - data[points[i]-1]) * (data[points[i]] - data[points[i]+1]) < 0
            push!(toremove, i)
        elseif abs(data[points[i]]) < lowoutlier && (data[points[i]] - data[points[i]-1]) * (data[points[i]] - data[points[i]+1]) < 0
            push!(toremove, i)
        end
    end

    filter(p -> !(p in toremove), points) |> collect
end

function yresolution(
    data::Vector{Float64},
    points::Vector{Int},
    params::FlexPointsParameters
)::Vector{Int}
    @unpack yresolution = params
    toremove = []

    valuebefore = data[first(points)]
    for point in points[2:end-1]
        value = data[point]
        if abs(value - valuebefore) < yresolution
            push!(toremove, point)
        else
            valuebefore = value
        end
    end

    filter(p -> !(p in toremove), points) |> collect
end

function removelinear(
    data::Vector{Float64},
    points::Vector{Int},
    params::FlexPointsParameters
)::Vector{Int}
    if length(points) <= 2
        return points
    end
    @unpack yresolution = params
    toremove = []

    firstpoint, middlepoint, lastpoint = points[1:3]
    for lastpoint in points[3:end]
        line = [
            (Float64(firstpoint), data[firstpoint]),
            (Float64(lastpoint), data[lastpoint])
        ]
        # prediction = Float64(linapprox(line, middlepoint))
        # error = abs(data[middlepoint] - prediction)
        error = midpointerror(line, (Float64(middlepoint), data[middlepoint]))
        if error < yresolution
            push!(toremove, middlepoint)
        else
            firstpoint = middlepoint
        end
        middlepoint = lastpoint
    end

    filter(p -> !(p in toremove), points) |> collect
end

function findsinusoid(
    data::Vector{Float64},
    points::Vector{Int},
    params::FlexPointsParameters
)::Vector{Int}
    @unpack yresolution = params
    toadd = []

    for (i, lastpoint) in enumerate(points[2:end])
        firstpoint = points[i]
        line = [
            (Float64(firstpoint), data[firstpoint]),
            (Float64(lastpoint), data[lastpoint])
        ]
        toperror = 0
        topindex = nothing
        for li in firstpoint:lastpoint
            # prediction = Float64(linapprox(line, li))
            # error = abs(data[li] - prediction)
            error = midpointerror(line, (Float64(li), data[li]))
            if error > toperror
                topindex = li
                toperror = error
            end
        end
        if !isnothing(topindex) && toperror > yresolution
            push!(toadd, topindex)
        end
    end

    sort(union(points, toadd))
end


# Variable derivatives indicates which derivatives should be use
# e.g. if `derivatives = (true, false, true, false)` 
# then the first and the third derivative will be used.
function flexpoints(
    data::Points2D,
    params::FlexPointsParameters
)::Tuple{Vector{Float64},Vector{Int}}
    @assert !isempty(data)
    requiredlen = length(data) ÷ 2 - 1
    maxderivative = topderivative(params.dselector)
    maxderivative == 0 && error("at least one derivative should be used")
    @assert requiredlen >= maxderivative

    datax = map(point -> x(point), data)
    datay = map(point -> y(point), data)
    maxvalue = maximum(datay)
    minvalue = minimum(datay)
    params = copy(params)
    params.yresolution = 2.0params.yresolution + 0.075 * (params.yresolution * (Float64(maxvalue) - Float64(minvalue)))

    datafiltered = if params.noisefilter.data
        noisefilter(datay, params.noisefilter.filtersize)
    else
        datay
    end

    ∂1data = maxderivative >= 1 ? ∂(collect(zip(datax, datafiltered)), 1) : nothing
    ∂2data = maxderivative >= 1 ? ∂(collect(zip(datax, ∂1data)), 2) : nothing
    ∂3data = maxderivative >= 2 ? ∂(collect(zip(datax, ∂2data)), 3) : nothing
    ∂4data = maxderivative >= 3 ? ∂(collect(zip(datax, ∂3data)), 4) : nothing

    if params.noisefilter.derivatives
        if !isnothing(∂1data) && !isempty(∂1data)
            ∂1data = noisefilter(∂1data, UInt(max(params.noisefilter.filtersize, 2)))
        end
        if !isnothing(∂2data) && !isempty(∂2data)
            ∂2data = noisefilter(∂2data, UInt(max(params.noisefilter.filtersize ÷ 2, 2)))
        end
        if !isnothing(∂3data) && !isempty(∂3data)
            ∂3data = noisefilter(∂3data, UInt(max(params.noisefilter.filtersize ÷ 3, 2)))
        end
        if !isnothing(∂4data) && !isempty(∂4data)
            ∂4data = noisefilter(∂4data, UInt(max(params.noisefilter.filtersize ÷ 4, 2)))
        end
    end

    derivatives = DerivativesData(∂1data, ∂2data, ∂3data, ∂4data)

    validpoints = mfilter(derivatives, params.dselector, params.mfilter)

    validpoints = flexpointsremoval(datafiltered, validpoints, params)

    validpoints = polyapprox(datafiltered, validpoints, params)

    # validpoints = removelinear(datafiltered, validpoints, params)

    datafiltered, validpoints
end

end