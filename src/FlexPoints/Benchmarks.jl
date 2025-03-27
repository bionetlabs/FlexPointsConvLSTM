module Benchmarks

export benchmark, benchmarkthreads, MIT_BIH_ARRHYTHMIA_2K, MIT_BIH_ARRHYTHMIA_5K, MIT_BIH_ARRHYTHMIA_FULL,
    geneticsearch, gridsearch

using DataFrames
using Statistics
using Evolutionary

using AKGECG.FlexPoints

const MIT_BIH_ARRHYTHMIA_2K = "data/mit_bih_arrhythmia_2k.csv"
const MIT_BIH_ARRHYTHMIA_5K = "data/mit_bih_arrhythmia_5k.csv"
const MIT_BIH_ARRHYTHMIA_FULL = "data/mit_bih_arrhythmia_full.csv"

function benchmark(
    datafile::String=MIT_BIH_ARRHYTHMIA_2K;
    parameters=FlexPointsParameters(),
    filteredreference::Bool=false,
    verbose=true
)::DataFrame
    println("loading $datafile, threadid $(Threads.threadid())")
    datadf = csv2df(datafile)
    cfs = []
    rmses = []
    nrmses = []
    minrmses = []
    prds = []
    nprds = []
    qss = []
    nqss = []
    seriesnames = []

    for seriesname in names(datadf)
        verbose && println("benchmarking $seriesname, threadid $(Threads.threadid())")
        ys = datadf[!, seriesname]
        datalen = length(ys)
        xs = LinRange(0, (datalen - 1) / SAMPES_PER_MILLISECOND, datalen)
        data = collect(zip(xs, ys))
        datafiltered, points = flexpoints(data, parameters)
        if filteredreference
            ys = datafiltered
        end
        points2d = map(i -> (Float64(i), ys[i]), points)
        reconstruction = map(1:datalen) do x
            Float64(linapprox(points2d, x))
        end

        push!(seriesnames, seriesname)
        cf_ = cf(ys, points)
        push!(cfs, cf_)
        push!(rmses, rmse(ys, reconstruction))
        push!(nrmses, nrmse(ys, reconstruction))
        push!(minrmses, minrmse(ys, reconstruction))
        prd_ = prd(ys, reconstruction)
        push!(prds, prd_)
        nprd_ = nprd(ys, reconstruction)
        push!(nprds, nprd_)
        push!(qss, qs(cf_, prd_))
        push!(nqss, nqs(cf_, nprd_))
    end

    push!(seriesnames, "mean")
    push!(cfs, mean(cfs))
    push!(rmses, mean(rmses))
    push!(nrmses, mean(nrmses))
    push!(minrmses, mean(minrmses))
    push!(prds, mean(prds))
    push!(nprds, mean(nprds))
    push!(qss, mean(qss))
    push!(nqss, mean(nqss))

    DataFrame(
        :lead => seriesnames,
        :cf => cfs,
        :rmse => rmses,
        :nrmse => nrmses,
        :minrmse => minrmses,
        :prd => prds,
        :nprd => nprds,
        :qs => qss,
        :nqs => nqss,
    )
end

function benchmarkthreads(
    datafile::String=MIT_BIH_ARRHYTHMIA_2K;
    parameters=FlexPointsParameters(),
    filteredreference::Bool=false,
    verbose=true
)::DataFrame
    println("loading $datafile, threadid $(Threads.threadid())")
    datadf = csv2df(datafile)
    cfs = []
    rmses = []
    nrmses = []
    minrmses = []
    prds = []
    nprds = []
    qss = []
    nqss = []
    seriesnames = []

    mutex = ReentrantLock()
    Threads.@threads for seriesname in names(datadf)
        verbose && println("benchmarking $seriesname, threadid $(Threads.threadid())")
        ys = datadf[!, seriesname]
        datalen = length(ys)
        xs = LinRange(0, (datalen - 1) / SAMPES_PER_MILLISECOND, datalen)
        data = collect(zip(xs, ys))
        datafiltered, points = flexpoints(data, parameters)
        if filteredreference
            ys = datafiltered
        end
        points2d = map(i -> (Float64(i), ys[i]), points)
        reconstruction = map(1:datalen) do x
            Float64(linapprox(points2d, x))
        end

        lock(mutex)
        try
            push!(seriesnames, seriesname)
            cf_ = cf(ys, points)
            push!(cfs, cf_)
            push!(rmses, rmse(ys, reconstruction))
            push!(nrmses, nrmse(ys, reconstruction))
            push!(minrmses, minrmse(ys, reconstruction))
            prd_ = prd(ys, reconstruction)
            push!(prds, prd_)
            nprd_ = nprd(ys, reconstruction)
            push!(nprds, nprd_)
            push!(qss, qs(cf_, prd_))
            push!(nqss, nqs(cf_, nprd_))
        finally
            unlock(mutex)
        end
    end

    push!(seriesnames, "mean")
    println(cfs, typeof(cfs))
    push!(cfs, mean(cfs))
    push!(rmses, mean(rmses))
    push!(nrmses, mean(nrmses))
    push!(minrmses, mean(minrmses))
    push!(prds, mean(prds))
    push!(nprds, mean(nprds))
    push!(qss, mean(qss))
    push!(nqss, mean(nqss))

    DataFrame(
        :lead => seriesnames,
        :cf => cfs,
        :rmse => rmses,
        :nrmse => nrmses,
        :minrmse => minrmses,
        :prd => prds,
        :nprd => nprds,
        :qs => qss,
        :nqs => nqss,
    )
end

function gridsearch(
    datafile::String=MIT_BIH_ARRHYTHMIA_2K;
    filteredreference::Bool=false
)
    mutex = ReentrantLock()
    bestqs = 0.0
    bestparameters = FlexPointsParameters()
    finished = 0
    println("finished jobs: $finished")
    Threads.@threads for yresolution in 0.021:0.002:0.021
        for polyapprox in 1:10
            for filtersize in 7:7
                parameters = FlexPointsParameters()
                parameters.yresolution = yresolution
                parameters.polyapprox = polyapprox
                parameters.noisefilter.filtersize = filtersize
                df = benchmark(datafile; parameters=parameters, filteredreference=filteredreference, verbose=false)
                qs = df[df.lead.=="mean", :qs][1]
                lock(mutex)
                try
                    if qs > bestqs
                        bestqs = qs
                        bestparameters = parameters
                    end
                    finished += 1
                    println("finished jobs: $finished")
                finally
                    unlock(mutex)
                end
            end
        end
    end
    (bestqs, bestparameters)
end

function geneticsearch(
    datafile::String=MIT_BIH_ARRHYTHMIA_2K;
    filteredreference::Bool=false,
    iterations=10,
    populationsize=10
)
    function f(x)
        parameters = FlexPointsParameters()
        parameters.yresolution = x[1]
        parameters.noisefilter.filtersize = round(UInt, x[2])
        df = benchmark(datafile; parameters=parameters, filteredreference=filteredreference, verbose=false)
        qs = df[df.lead.=="mean", :qs][1]
        -qs # due to minimization problem
    end

    ga = GA(
        populationSize=populationsize,
        selection=uniformranking(3),
        mutation=gaussian(),
        crossover=uniformbin()
    )

    defaultparams = FlexPointsParameters()
    lower = [1e-3, 1]
    upper = [1e-1, 10]
    constraints = BoxConstraints(lower, upper)
    x0 = [defaultparams.yresolution, defaultparams.noisefilter.filtersize]

    Evolutionary.optimize(
        f,
        constraints,
        x0,
        ga,
        Evolutionary.Options(iterations=iterations, parallelization=:thread, show_trace=true)
    )
end

end