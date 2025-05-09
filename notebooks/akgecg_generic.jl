### A Pluto.jl notebook ###
# v0.20.6

using Markdown
using InteractiveUtils

# ╔═╡ 57f5fc4e-bf83-4e83-95bb-dd4c44c4b52f
begin
    projectdir = dirname(Base.current_project())

    import Pkg
    Pkg.activate(projectdir)
    Pkg.instantiate()

    using AKGECG
    using AKGECG.FlexPoints

    using DataFrames
    using CSV
    using WGLMakie
    using CairoMakie
    using Dates
    using Statistics
    using CUDA
    using cuDNN
    using Latexify
    using OrderedCollections
    using JSON
    using ProgressLogging
    using MLJ
    using MLJFlux
    using Flux
    using FileIO
    using ImageMagick
    using Latexify
    using HypothesisTests
    using Random

    import GeoInterface as GI
    import GeometryOps as GO
    import GeometryBasics as GB
end

# ╔═╡ f923a8c6-f2ed-41b8-9bd6-f9830790faec
md"##### project setup"

# ╔═╡ 41372110-ff60-11ef-1068-e92f06552b64
html"""
<style>
	@media screen {
		main {
			margin: 0 auto;
			max-width: 2000px;
    		padding-left: max(283px, 10%);
    		padding-right: max(383px, 10%); 
            # 383px to accomodate TableOfContents(aside=true)
		}
	}
</style>
"""

# ╔═╡ c8f39317-2b0d-47e7-8dc8-52eebbf6193d
begin
    # CUDA.versioninfo()
    CUDA.device!(1)
    device = Flux.gpu_device()
end

# ╔═╡ 6cb60280-d193-4756-9e84-1e6bf427379a
md"##### data baseline"

# ╔═╡ 9b599684-619d-49a2-8844-e5fd18af52a2
begin
    datadir = joinpath(projectdir, "data", "mitdb_prepared")
    flexpointsdir = joinpath(projectdir, "data", "mitdb_flexpoints")

    data2k = csv2df(joinpath(datadir, "mit_bih_arrhythmia_2k.csv"))
    data5k = csv2df(joinpath(datadir, "mit_bih_arrhythmia_5k.csv"))
    data = csv2df(joinpath(datadir, "mit_bih_arrhythmia_full.csv"))
    annotations = open(joinpath(datadir, "mit_bih_annotations.json"), "r") do f
        JSON.parse(f)
    end

    datalen = nrow(data)
    datalen, size(data2k), size(data5k), size(data), length(annotations)
end

# ╔═╡ 32af21c7-f656-4770-87cc-7b084fd57580
annotations

# ╔═╡ 20056417-66fa-4de9-b35c-d235b0afcc0a
records = keys(annotations) |> collect |> sort

# ╔═╡ b53c8508-f45a-460b-8b70-bd8cb3d4778b
begin
    annotatedrecords = OrderedDict()
    labelsall = Set([">"])
    auxlabelsall = Set([">"])
    for record in records
        lead1_colname = string(record, "_I")
        lead2_colname = string(record, "_II")

        samples = annotations[record]["sample"]
        labels = fill("Q", 1:first(samples))

        for i in 1:length(samples)
            label_startindex = samples[i] + 1
            label_endindex = i < length(samples) ? samples[i+1] : datalen
            indexrange = label_startindex:label_endindex
            label = annotations[record]["symbol"][i]
            append!(labels, fill(label, length(indexrange)))
            push!(labelsall, label)
        end

        sample_lastindex = findfirst(x -> startswith(x, "("), annotations[record]["aux_note"])
        auxlabel_last = replace(annotations[record]["aux_note"][sample_lastindex], "\0" => "")
        sample_lastindex_value = annotations[record]["sample"][sample_lastindex] + 1
        auxlabels = fill("Q", 1:(sample_lastindex_value-1))
        for i in (sample_lastindex+1):length(samples)
            sample_index_value = samples[i] + 1
            auxlabel = replace(annotations[record]["aux_note"][i], "\0" => "")
            if (startswith(auxlabel, "(") && auxlabel_last != auxlabel)
                append!(auxlabels, fill(auxlabel_last, ((sample_index_value - 1) - sample_lastindex_value) + 1))
                push!(auxlabelsall, auxlabel_last)
                sample_lastindex_value = sample_index_value
                auxlabel_last = auxlabel
            end
            if i == length(samples)
                append!(auxlabels, fill(auxlabel_last, (length(data[!, lead1_colname]) - sample_lastindex_value) + 1))
                push!(auxlabelsall, auxlabel_last)
            end
        end

        annotatedrecords[record] = DataFrame(
            lead1_colname => data[!, lead1_colname],
            lead2_colname => data[!, lead2_colname],
            "label" => labels,
            "auxlabel" => auxlabels,
        )
    end

    labelsall = labelsall |> collect
    auxlabelsall = auxlabelsall |> collect

    annotatedrecords
end

# ╔═╡ 9f00fc06-f9b5-4fbb-a8da-be48c5c81e01
begin
    labelchunks = OrderedDict(map(x -> x => OrderedDict("I" => Vector{Float64}[], "II" => Vector{Float64}[]), labelsall))
    for (record, df) in annotatedrecords
        n = nrow(df)
        label_lastindex = 1
        label_last = df.label[label_lastindex]
        for i in 2:n
            label = df.label[i]
            if label_last != label || i == n
                push!(labelchunks[label_last]["I"], df[label_lastindex:(i-1), string(record, "_I")])
                push!(labelchunks[label_last]["II"], df[label_lastindex:(i-1), string(record, "_II")])
                label_lastindex = i
                label_last = label
            end
        end
    end
    labelchunks
end

# ╔═╡ df73ebe6-b9d7-4aa2-b8f8-d5b51ea32480
begin
    auxlabelchunks = OrderedDict(map(x -> x => OrderedDict("I" => Vector{Float64}[], "II" => Vector{Float64}[]), auxlabelsall))
    for (record, df) in annotatedrecords
        n = nrow(df)
        auxlabel_lastindex = findfirst(x -> startswith(x, "("), df.auxlabel)
        auxlabel_last = df.auxlabel[auxlabel_lastindex]
        for i in (auxlabel_lastindex+1):n
            auxlabel = df.auxlabel[i]
            if (startswith(auxlabel, "(") && auxlabel_last != auxlabel) || i == n
                push!(auxlabelchunks[auxlabel_last]["I"], df[auxlabel_lastindex:(i-1), string(record, "_I")])
                push!(auxlabelchunks[auxlabel_last]["II"], df[auxlabel_lastindex:(i-1), string(record, "_II")])
                auxlabel_lastindex = i
                auxlabel_last = auxlabel
            end
        end
    end
    auxlabelchunks
end

# ╔═╡ cbfa601b-a8f9-499d-af05-8d5f9ff6ce1b
begin
    auxfrequencies_raw = OrderedDict(map(x -> x => 0, keys(auxlabelchunks) |> collect))
    for (record, df) in annotatedrecords
        for label in keys(auxlabelchunks)
            auxfrequencies_raw[label] += sum(df.auxlabel .== label)
        end
    end
    auxfrequencies_raw
end

# ╔═╡ 2d68804a-c210-448f-a99d-bfde5670ada0
begin
    auxfrequencies = OrderedDict(map(x -> x => 0, keys(auxlabelchunks) |> collect))
    for (label, leads) in auxlabelchunks
        for (lead, chunks) in leads
            for chunk in chunks
                auxfrequencies[label] += length(chunk)
            end
        end
    end
    auxfrequencies
end

# ╔═╡ d572be5a-1c16-4a1f-b733-44367fa05b8a
begin
    wfdblabels = """
    	 label_store symbol                                    description
    	0             0                              Not an actual annotation
    	1             1      N                                    Normal beat
    	2             2      L                  Left bundle branch block beat
    	3             3      R                 Right bundle branch block beat
    	4             4      a                Aberrated atrial premature beat
    	5             5      V              Premature ventricular contraction
    	6             6      F          Fusion of ventricular and normal beat
    	7             7      J              Nodal (junctional) premature beat
    	8             8      A                   Atrial premature contraction
    	9             9      S     Premature or ectopic supraventricular beat
    	10           10      E                        Ventricular escape beat
    	11           11      j                 Nodal (junctional) escape beat
    	12           12      /                                     Paced beat
    	13           13      Q                            Unclassifiable beat
    	14           14      ~                          Signal quality change
    	16           16      |                     Isolated QRS-like artifact
    	18           18      s                                      ST change
    	19           19      T                                  T-wave change
    	20           20      *                                        Systole
    	21           21      D                                       Diastole
    	22           22      "                             Comment annotation
    	23           23      =                         Measurement annotation
    	24           24      p                                    P-wave peak
    	25           25      B              Left or right bundle branch block
    	26           26      ^                      Non-conducted pacer spike
    	27           27      t                                    T-wave peak
    	28           28      +                                  Rhythm change
    	29           29      u                                    U-wave peak
    	30           30      ?                                       Learning
    	31           31      !                       Ventricular flutter wave
    	32           32      [      Start of ventricular flutter/fibrillation
    	33           33      ]        End of ventricular flutter/fibrillation
    	34           34      e                             Atrial escape beat
    	35           35      n                   Supraventricular escape beat
    	36           36      @  Link to external data (aux_note contains URL)
    	37           37      x             Non-conducted P-wave (blocked APB)
    	38           38      f                Fusion of paced and normal beat
    	39           39      (                                 Waveform onset
    	40           40      )                                   Waveform end
    	41           41      r       R-on-T premature ventricular contraction
    """
    md"wfdb labels"
end

# ╔═╡ 8b654cdd-8d27-40c8-932e-5752d394540f
begin
    physionetlabels = """
    	Symbol	Meaning
    	· or N	Normal beat
    	L	Left bundle branch block beat
    	R	Right bundle branch block beat
    	A	Atrial premature beat
    	a	Aberrated atrial premature beat
    	J	Nodal (junctional) premature beat
    	S	Supraventricular premature beat
    	V	Premature ventricular contraction
    	F	Fusion of ventricular and normal beat
    	[	Start of ventricular flutter/fibrillation
    	!	Ventricular flutter wave
    	]	End of ventricular flutter/fibrillation
    	e	Atrial escape beat
    	j	Nodal (junctional) escape beat
    	E	Ventricular escape beat
    	/	Paced beat
    	f	Fusion of paced and normal beat
    	x	Non-conducted P-wave (blocked APB)
    	Q	Unclassifiable beat
    	|	Isolated QRS-like artifact
    	Rhythm annotations appear below the level used for beat annotations:
    	(AB	Atrial bigeminy
    	(AFIB	Atrial fibrillation
    	(AFL	Atrial flutter
    	(B	Ventricular bigeminy
    	(BII	2° heart block
    	(IVR	Idioventricular rhythm
    	(N	Normal sinus rhythm
    	(NOD	Nodal (A-V junctional) rhythm
    	(P	Paced rhythm
    	(PREX	Pre-excitation (WPW)
    	(SBR	Sinus bradycardia
    	(SVTA	Supraventricular tachyarrhythmia
    	(T	Ventricular trigeminy
    	(VFL	Ventricular flutter
    	(VT	Ventricular tachycardia
    	Signal quality and comment annotations appear above the level used for beat annotations:
    	qq	Signal quality change: the first character (`c' or `n') indicates the quality of the upper signal (clean or noisy), and the second character indicates the quality of the lower signal
    	U	Extreme noise or signal loss in both signals: ECG is unreadable
    	M (or MISSB)	Missed beat
    	P (or PSE)	Pause
    	T (or TS)	Tape slippage
    """
    md"pythsionet labes"
end

# ╔═╡ 2b1b21a3-97e5-4475-9963-e5794489f0c8
labelsdescribed = OrderedDict(
    ">" => "Initial unclassified beat",
    "|" => "Isolated QRS-like artifact",
    "Q" => "Unclassifiable beat",
    "f" => "Fusion of paced and normal beat",
    "e" => "Atrial escape beat",
    "!" => "Ventricular flutter wave",
    "V" => "Premature ventricular contraction",
    "x" => "Non-conducted P-wave (blocked APB)",
    "L" => "Left bundle branch block beat",
    "a" => "Aberrated atrial premature beat",
    "N" => "Normal beat",
    "/" => "Paced beat",
    "[" => "Start of ventricular flutter/fibrillation",
    "]" => "End of ventricular flutter/fibrillation",
    "A" => "Atrial premature contraction",
    "j" => "Nodal (junctional) escape beat",
    "E" => "Ventricular escape beat",
    "J" => "Nodal (junctional) premature beat",
    "S" => "Premature or ectopic supraventricular beat",
    "~" => "Signal quality change",
    "+" => "Rhythm change",
    "R" => "Right bundle branch block beat",
    "\\" => "Not an actual annotation",
    "F" => "Fusion of ventricular and normal beat",
)

# ╔═╡ bb20f631-298e-4072-bcd0-0a8ed1aa954c
auxlabeldescribed = OrderedDict(
    ">" => "Initial unclassified rhythm",
    "(N" => "Normal sinus rhythm",
    "(AB" => "Atrial bigeminy",
    "(SVTA" => "Supraventricular tachyarrhythmia",
    "(T" => "Ventricular trigeminy",
    "(AFL" => "Atrial flutter",
    "(VFL" => "Ventricular flutter",
    "(B" => "Ventricular bigeminy",
    "(PREX" => "Pre-excitation (WPW)",
    "PSE" => "Pause",
    "(BII" => "2° heart block",
    "(NOD" => "Nodal (A-V junctional) rhythm",
    "TS" => "Tape slippage",
    "(P" => "Paced rhythm",
    "(IVR" => "Idioventricular rhythm",
    "MISSB" => "Missed beat",
    "(VT" => "Ventricular tachycardia",
    "" => "Not an actual annotation",
    "(SBR" => "Sinus bradycardia",
    "(AFIB" => "Atrial fibrillation",
)

# ╔═╡ 05de5742-b453-4b20-a156-42c53373974d
begin
    rhythmdf = DataFrame(
        :label => collect(keys(auxfrequencies_raw)),
        :description => map(x -> auxlabeldescribed[x], collect(keys(auxfrequencies_raw))),
        :annotations => collect(values(auxfrequencies_raw)),
        :duration => collect(round.(values(auxfrequencies_raw) ./ (360 * 60); digits=2)),
    )
    latexify(rhythmdf; env=:table, booktabs=false, latex=false) |> print
end

# ╔═╡ 46ab0c02-74ac-434c-a525-7f01312e32ea
md"##### flexpoints"

# ╔═╡ f9baee8e-8c0a-4e6f-b9a2-027fbdc04519
parameters = FlexPointsParameters(
    dselector=DerivativesSelector(
        ∂1=true,
        ∂2=true,
        ∂3=true,
        ∂4=true,
    ),
    noisefilter=NoiseFilterParameters(
        data=true,
        derivatives=true,
        filtersize=9,
    ),
    mfilter=MFilterParameters(
        m1=1.5e-4,
        m2=1.5e-4,
        m3=0,
    ),
    mspp=5, # minimum samples per period
    frequency=360, # number of samples of signal per second 
    devv=1.0, # statistical measure for outliers in terms of standard deviation
    removeoutliers=false,
    yresolution=0.0085, # 0.0075, # values with smaller Δ are considered as one point 
    polyapprox=1,
    polyapprox_yresolutionratio=2.0,
)

# ╔═╡ ec9fb06f-1c90-4463-bd58-6bd1d8177a8f
@kwdef struct ChunkData
    data::Vector{Float64}
    datafiltered::Vector{Float64}
    flexpoints::Vector{Tuple{Float64,Float64}}
    reconstruction::Vector{Float64}
    cf::Float64
    prd::Float64
    qs::Float64
end

# ╔═╡ bfa6bd3e-93e0-491b-8e3a-1bc009e32858
scores(lead::ChunkData) = OrderedDict(:cf => lead.cf, :prd => lead.prd, :qs => lead.qs)

# ╔═╡ 1309496c-f270-44d2-bf9a-d775ab177076
function find_flexpoints(ys, parameters, filteredreference)::ChunkData
    datalen = length(ys)
    xs = 1:datalen .|> Float64 |> collect
    data = collect(zip(xs, ys))
    datafiltered, points = flexpoints(data, parameters)

    points2d = map(points) do i
        windowsize = parameters.noisefilter.filtersize + 3
        if i < windowsize || i > (length(ys) - (windowsize - 1))
            if filteredreference
                (Float64(i), datafiltered[i])
            else
                (Float64(i), ys[i])
            end
        else
            if filteredreference
                (Float64(i), datafiltered[i])
            else
                (Float64(i), ys[i])
            end
        end
    end

    reconstruction = map(1:datalen) do x
        Float64(linapprox(points2d, x))
    end

    cfscore = cf(ys, points)
    prdscore = prd(ys, reconstruction)
    ChunkData(
        ys,
        datafiltered,
        points2d,
        reconstruction,
        cfscore,
        prdscore,
        qs(cfscore, prdscore),
    )
end

# ╔═╡ 71b193d8-8481-491a-847c-cbf49041c3d3
function find_rdp(ys; epsilon=0.03)::ChunkData
    datalen = length(ys)
    xs = 1:datalen .|> Float64 |> collect
    data = GI.LineString([GB.Point2(x, ys[Int(x)]) for x in xs])

    datafiltered = ys

    points2d = GO.simplify(data; tol=epsilon).geom
    points = Int.(first.(points2d))


    reconstruction = map(1:datalen) do x
        Float64(linapprox(points2d, x))
    end

    cfscore = cf(ys, points)
    prdscore = prd(ys, reconstruction)
    ChunkData(
        ys,
        datafiltered,
        points2d,
        reconstruction,
        cfscore,
        prdscore,
        qs(cfscore, prdscore),
    )
end

# ╔═╡ 299034a7-cea1-4edb-a88f-43003bc34ca7
fp2k = let
    filteredreference = true
    datadf = data2k # data2k, data5k, datafull
    fpdict = OrderedDict()
    for seriesname in names(datadf)
        fpdict[seriesname] = find_flexpoints(datadf[!, seriesname], parameters, true)
    end
    fpdict
end

# ╔═╡ a54fd49b-40db-43d0-816b-9dd0a38a805c
rdp2k = let
    filteredreference = true
    datadf = data2k # data2k, data5k, datafull
    rdpdict = OrderedDict()
    for seriesname in names(datadf)
        rdpdict[seriesname] = find_rdp(datadf[!, seriesname])
    end
    rdpdict
end

# ╔═╡ 81573a74-2b76-4365-a219-db7ed4da982a
let
    WGLMakie.activate!()
    set_theme!(theme_dark())

    fig1 = Figure(size=(1322, 500))
    ax1 = Axis(fig1[1, 1])

    lead = "100_I"

    lines!(ax1, fp2k[lead].data, colormap=:seaborn_colorblind)
    lines!(ax1, fp2k[lead].datafiltered, colormap=:seaborn_colorblind)
    scatter!(ax1, fp2k[lead].flexpoints, color=:orange)
    lines!(ax1, fp2k[lead].reconstruction, colormap=:seaborn_colorblind, linestyle=:dash)

    println(scores(fp2k[lead]))

    fig1
end

# ╔═╡ 9afe58ed-8a5c-4660-a262-b21335022339
let
    WGLMakie.activate!()
    set_theme!(theme_dark())

    fig1 = Figure(size=(1322, 500))
    ax1 = Axis(fig1[1, 1])

    lead = "100_I"

    lines!(ax1, rdp2k[lead].data, colormap=:seaborn_colorblind)
    lines!(ax1, rdp2k[lead].datafiltered, colormap=:seaborn_colorblind)
    scatter!(ax1, rdp2k[lead].flexpoints, color=:orange)
    lines!(ax1, rdp2k[lead].reconstruction, colormap=:seaborn_colorblind, linestyle=:dash)

    println(scores(rdp2k[lead]))

    fig1
end

# ╔═╡ 56ed382b-7f26-45a6-b0e3-ec6f19f0f890
function flexpointschunks(
    selectedauxlabels::Vector{String},
    chunks::OrderedDict,
    limit::Union{Missing,Int}=missing;
    seqencelength::Int=5025,
    shuffle::Bool=false,
    rng::Int=58,
)
    filtered_auxlabechunks = filter(((k, v),) -> k in selectedauxlabels, chunks)
    fpdict = OrderedDict(OrderedDict(map(x -> x => OrderedDict("I" => ChunkData[], "II" => ChunkData[]), selectedauxlabels)))
    maxcounter = 0

    @withprogress name = "auxlabel flexpoints chunks" begin
        for (i, (auxlabel, leads)) in enumerate(filtered_auxlabechunks)
            for (j, (lead, chunks)) in enumerate(leads)
                allchunks = copy(chunks)
                if shuffle
                    shuffle!(MersenneTwister(rng), allchunks)
                end
                counter = 0

                @progress name = "$(auxlabel) -> $(lead) -> $(length(chunks))" for chunk in allchunks
                    nchunk = length(chunk)
                    validportions = nchunk ÷ seqencelength
                    for portionindex in 1:validportions
                        startindex = max(1, (portionindex - 1) * seqencelength + 1)
                        endindex = portionindex * seqencelength

                        achunk = chunk[startindex:endindex]
                        fpres = find_flexpoints(achunk, parameters, true)
                        push!(fpdict[auxlabel][lead], fpres)
                        counter += length(achunk)
                        if !ismissing(limit) && counter >= limit
                            @logprogress 1.0
                            break
                        end
                        if shuffle && portionindex >= 5
                            break
                        end
                    end
                    if !ismissing(limit) && counter >= limit
                        @logprogress 1.0
                        break
                    end
                end

                maxcounter = max(maxcounter, counter)
                @logprogress (2 * (i - 1) + j) / 2length(chunks)
            end
        end
    end

    fpdict, maxcounter
end

# ╔═╡ efee10e4-8272-4717-b4ea-f4f53cd889ca
function rdpchunks(
    selectedauxlabels::Vector{String},
    chunks::OrderedDict,
    limit::Union{Missing,Int}=missing;
    seqencelength::Int=5025,
    shuffle::Bool=false,
    rng::Int=58,
)
    filtered_auxlabechunks = filter(((k, v),) -> k in selectedauxlabels, chunks)
    rdpdict = OrderedDict(OrderedDict(map(x -> x => OrderedDict("I" => ChunkData[], "II" => ChunkData[]), selectedauxlabels)))
    maxcounter = 0

    @withprogress name = "auxlabel flexpoints chunks" begin
        for (i, (auxlabel, leads)) in enumerate(filtered_auxlabechunks)
            for (j, (lead, chunks)) in enumerate(leads)
                allchunks = copy(chunks)
                if shuffle
                    shuffle!(MersenneTwister(rng), allchunks)
                end
                counter = 0

                @progress name = "$(auxlabel) -> $(lead) -> $(length(chunks))" for chunk in allchunks
                    nchunk = length(chunk)
                    validportions = nchunk ÷ seqencelength
                    for portionindex in 1:validportions
                        startindex = max(1, (portionindex - 1) * seqencelength + 1)
                        endindex = portionindex * seqencelength

                        achunk = chunk[startindex:endindex]
                        rdpres = find_rdp(achunk)
                        push!(rdpdict[auxlabel][lead], rdpres)
                        counter += length(achunk)
                        if !ismissing(limit) && counter >= limit
                            @logprogress 1.0
                            break
                        end
                        if shuffle && portionindex >= 5
                            break
                        end
                    end
                    if !ismissing(limit) && counter >= limit
                        @logprogress 1.0
                        break
                    end
                end

                maxcounter = max(maxcounter, counter)
                @logprogress (2 * (i - 1) + j) / 2length(chunks)
            end
        end
    end

    rdpdict, maxcounter
end

# ╔═╡ deeb8466-d89d-45d5-bd7d-ffef89c497e4
selectedpathology = "(T"

# ╔═╡ 73302630-7c63-4a95-9196-82d15f3c3905
vtfp, vt_targetcounter, vt_normalcounter = let
    targetauxlabel = selectedpathology
    normalauxlabel = "(N"
    selectedauxlabels = [normalauxlabel, targetauxlabel]

    target_fpdict, targetcounter = flexpointschunks([targetauxlabel], auxlabelchunks)
    normal_fpdict, normalcounter = flexpointschunks([normalauxlabel], auxlabelchunks, targetcounter; shuffle=true)

    fpdict = OrderedDict(target_fpdict..., normal_fpdict...)
    fpdict, targetcounter, normalcounter
end

# ╔═╡ 2571d757-0562-4bc6-aaa5-1b2160e9f1ce
vtrdp, vtrdp_targetcounter, vtrdp_normalcounter = let
    targetauxlabel = selectedpathology
    normalauxlabel = "(N"
    selectedauxlabels = [normalauxlabel, targetauxlabel]

    target_rdpdict, targetcounter = rdpchunks([targetauxlabel], auxlabelchunks)
    normal_rdpdict, normalcounter = rdpchunks([normalauxlabel], auxlabelchunks, targetcounter; shuffle=true)

    rdpdict = OrderedDict(target_rdpdict..., normal_rdpdict...)
    rdpdict, targetcounter, normalcounter
end

# ╔═╡ 76f98311-86ad-4a15-891f-1cedb5d82713
begin
    lseg = map(x -> length(x), values(vtfp[selectedpathology])) |> sum
    nseg = map(x -> length(x), values(vtfp["(N"])) |> sum
    lseg_rdp = map(x -> length(x), values(vtrdp[selectedpathology])) |> sum
    nseg_rdp = map(x -> length(x), values(vtrdp["(N"])) |> sum
    lseg, nseg, lseg_rdp, nseg_rdp
end

# ╔═╡ 3da44940-b063-4acf-a4eb-1ef94d0a3cbe
begin
    lcf = map(x -> mean(map(y -> y.cf, x) |> collect), values(vtfp[selectedpathology])) |> mean
    lprd = map(x -> mean(map(y -> y.prd, x) |> collect), values(vtfp[selectedpathology])) |> mean
    lqs = map(x -> mean(map(y -> y.qs, x) |> collect), values(vtfp[selectedpathology])) |> mean

    ncf = map(x -> mean(map(y -> y.cf, x) |> collect), values(vtfp["(N"])) |> mean
    nprd = map(x -> mean(map(y -> y.prd, x) |> collect), values(vtfp["(N"])) |> mean
    nqs = map(x -> mean(map(y -> y.qs, x) |> collect), values(vtfp["(N"])) |> mean

    lcf, lprd, lqs, ncf, nprd, nqs
end

# ╔═╡ 2c79044d-6c38-4d4f-bcd5-febd40993831
begin
    lcf_rdp = map(x -> mean(map(y -> y.cf, x) |> collect), values(vtrdp[selectedpathology])) |> mean
    lprd_rdp = map(x -> mean(map(y -> y.prd, x) |> collect), values(vtrdp[selectedpathology])) |> mean
    lqs_rdp = map(x -> mean(map(y -> y.qs, x) |> collect), values(vtrdp[selectedpathology])) |> mean

    ncf_rdp = map(x -> mean(map(y -> y.cf, x) |> collect), values(vtrdp["(N"])) |> mean
    nprd_rdp = map(x -> mean(map(y -> y.prd, x) |> collect), values(vtrdp["(N"])) |> mean
    nqs_rdp = map(x -> mean(map(y -> y.qs, x) |> collect), values(vtrdp["(N"])) |> mean

    lcf_rdp, lprd_rdp, lqs_rdp, ncf_rdp, nprd_rdp, nqs_rdp
end

# ╔═╡ e7e4d8fc-f344-4495-b023-83a2ca66d885
begin
    vtfp_json = JSON.json(vtfp)
    open(joinpath(flexpointsdir, "vtfp.json"), "w") do f
        write(f, vtfp_json)
    end
end

# ╔═╡ 1a67db99-ffc8-456d-a6c3-fd0da8755ace
# ╠═╡ disabled = true
#=╠═╡
begin
	vtfp = OrderedDict()
	vtfp_raw = JSON.parsefile(joinpath(flexpointsdir, "vtfp.json"))
	for (key, value) in vtfp_raw
		println("key $key: ", vtfp_raw[key]["I"] |> first |> keys)
		vtfp[key] = OrderedDict("I" => ChunkData[], "II" => ChunkData[])
		for lead in ["I", "II"]
			for chunk in vtfp_raw[key][lead]
				println(chunk)
				chunkdata = ChunkData(
					data = Float64.(chunk["data"]),
					datafiltered = Float64.(chunk["datafiltered"]),
					flexpoints = Tuple{Float64, Float64}.(chunk["flexpoints"]),
					reconstruction = Float64.(chunk["reconstruction"]),
					cf = Float64(chunk["cf"]),
					prd = Float64(chunk["prd"]),
					qs = Float64(chunk["qs"]),
				)
				push!(vtfp[key][lead], chunkdata)
			end
		end
	end
	vtfp
end
  ╠═╡ =#

# ╔═╡ 13260533-5a77-4479-9378-44e7a1c4a861
let
    WGLMakie.activate!()
    set_theme!(theme_light())

    fontsize = 12 * 2
    fig1 = Figure(size=(7.5 * 300, 2.0 * 250), fontsize=fontsize, fonts=(; regular="Times New Roman"), textcolor=:black)

    startpoint = 10
    endpoint = 1990
    ax = Axis(
        fig1[1, 1],
        titlealign=:left,
        title=string("B) ", auxlabeldescribed[selectedpathology]),
        titlefont="Times New Roman",
        titlecolor=:gray25,
        titlesize=fontsize,
        yautolimitmargin=(0.05, 0.05),
        xautolimitmargin=(0.05, 0.05),
        xticks=(1:360:(endpoint-startpoint), collect(map(x -> string(Int(round((x - startpoint) / 360; digits=0))), collect(startpoint:360:endpoint)))),
        xlabel="time (s)",
        xlabelsize=fontsize * 1.2,
        ylabel="signal amplitude (mV)",
        ylabelsize=fontsize * 1.2,
    )

    signal = vtfp[selectedpathology]["I"][1]

    fpmin = findfirst(x -> x[1] >= startpoint, signal.flexpoints)
    fpmax = findlast(x -> x[1] <= endpoint, signal.flexpoints)

    vtpoints = signal.flexpoints[fpmin:fpmax]
    vtpoints = map(x -> (x[1] - (startpoint - 1), x[2]), vtpoints) |> collect
    lines!(ax, signal.data[startpoint:endpoint], color=:gray53, linewidth=0.7)
    scatter!(ax, vtpoints, color=:darkorange2)
    lines!(ax, signal.reconstruction[startpoint:endpoint], color=:deepskyblue3, linestyle=:dashdot, linewidth=1.5)

    save(joinpath(projectdir, "img", "Fig1_$selectedpathology.png"), fig1, px_per_unit=1)

    fig1
end

# ╔═╡ 94cfd77d-92d8-4c19-b3c1-897a5aa0dcbb
let
    WGLMakie.activate!()
    set_theme!(theme_light())

    fontsize = 12 * 2
    fig1 = Figure(size=(3.25 * 300, 2.25 * 300), fontsize=fontsize, fonts=(; regular="Times New Roman"), textcolor=:black)

    startpoint = 780
    endpoint = 1250
    ax = Axis(
        fig1[1, 1],
        titlealign=:left,
        # title=string("A) ", auxlabeldescribed[selectedpathology]),
        titlefont="Times New Roman",
        titlecolor=:gray25,
        titlesize=fontsize,
        yautolimitmargin=(0.05, 0.05),
        xautolimitmargin=(0.05, 0.05),
        xticks=(1:360:(endpoint-startpoint), collect(map(x -> string(Int(round((x - startpoint) / 360; digits=0))), collect(startpoint:360:endpoint)))),
        xlabel="time (s)",
        xlabelsize=fontsize * 1.2,
        ylabel="signal amplitude (mV)",
        ylabelsize=fontsize * 1.2,
    )

    signal = vtfp[selectedpathology]["I"][1]

    fpmin = findfirst(x -> x[1] >= startpoint, signal.flexpoints)
    fpmax = findlast(x -> x[1] <= endpoint, signal.flexpoints)

    vtpoints = signal.flexpoints[fpmin:fpmax]
    vtpoints = map(x -> (x[1] - (startpoint - 1), x[2]), vtpoints) |> collect
    lines!(ax, signal.data[startpoint:endpoint], color=:gray53, linewidth=0.7)
    scatter!(ax, vtpoints, color=:darkorange2)
    lines!(ax, signal.reconstruction[startpoint:endpoint], color=:deepskyblue3, linestyle=:dashdot, linewidth=1.5)

    save(joinpath(projectdir, "img", "Fig4.png"), fig1, px_per_unit=1)

    fig1
end

# ╔═╡ 103fe3b0-1010-401e-97ea-ea98d4a67303
md"##### data organization"

# ╔═╡ 946022e6-382b-4b75-a76c-75c1625110a0
let
    vtlengths = map(vtfp[selectedpathology]["I"]) do chunk
        chunk.data |> length
    end
    normallengths = map(vtfp["(N"]["I"]) do chunk
        chunk.data |> length
    end

    vtdf = DataFrame(:vtlength => vtlengths)
    normaldf = DataFrame(:normallengths => normallengths)

    describe(vtdf), describe(normaldf)
end

# ╔═╡ 5c315695-291b-42a9-92c1-9db0323ffd4e
begin
    seqencelength = 1000
    xraw = Array{Float32}(undef, (0, seqencelength))
    xfiltered = Array{Float32}(undef, (0, seqencelength))
    xreconstructed = Array{Float32}(undef, (0, seqencelength))
    xreconstructed_rdp = Array{Float32}(undef, (0, seqencelength))
    ylabels = []

    validportions_vec = []
    for label in [selectedpathology, "(N"]
        validportions_total = 0
        for chunk in vtfp[label]["I"]
            nchunk = length(chunk.data)
            validportions = nchunk ÷ seqencelength
            validportions_total += validportions
        end
        push!(validportions_vec, validportions_total)
    end
    validportions_min = minimum(validportions_vec)

    for label in [selectedpathology, "(N"]
        validportions_total = 0
        for chunk in vtfp[label]["I"]
            nchunk = length(chunk.data)
            validportions = nchunk ÷ seqencelength
            for portionindex in 1:validportions
                startindex = max(1, (portionindex - 1) * seqencelength + 1)
                endindex = portionindex * seqencelength
                global xraw = vcat(xraw, reshape(chunk.data[startindex:endindex], (1, seqencelength)))
                global xfiltered = vcat(xfiltered, reshape(chunk.datafiltered[startindex:endindex], (1, seqencelength)))
                global xreconstructed = vcat(xreconstructed, reshape(chunk.reconstruction[startindex:endindex], (1, seqencelength)))
                push!(ylabels, label)
                validportions_total += 1
                if validportions_total >= validportions_min
                    @goto skiplabel
                end
            end
        end
        @label skiplabel
        println("$label validportions_total: $validportions_total")
    end

    for label in [selectedpathology, "(N"]
        validportions_total = 0
        for chunk in vtrdp[label]["I"]
            nchunk = length(chunk.data)
            validportions = nchunk ÷ seqencelength
            for portionindex in 1:validportions
                startindex = max(1, (portionindex - 1) * seqencelength + 1)
                endindex = portionindex * seqencelength
                global xreconstructed_rdp = vcat(xreconstructed_rdp, reshape(chunk.reconstruction[startindex:endindex], (1, seqencelength)))
                validportions_total += 1
                if validportions_total >= validportions_min
                    @goto skiplabel_rdp
                end
            end
        end
        @label skiplabel_rdp
        println("$label validportions_total: $validportions_total")
    end
    size(xraw), size(xfiltered), size(xreconstructed), size(xreconstructed_rdp), size(ylabels)
end

# ╔═╡ 86beeb4b-c375-4a51-9c72-5c2754e764d4
begin
    nofeatures = 1
    rngseed = 58
    local builder = MLJFlux.@builder begin
        Flux.Chain(
            # x -> (println("before reshape: ", size(x)); x),
            x -> reshape(x, (seqencelength, nofeatures, :)),

            #(signallen, 1, batch) -> (L1, 16, batch)
            # x -> (println("before conv1: ", size(x)); x),
            Conv((20,), nofeatures => 3, relu, stride=1, pad=19),

            # x -> (println("before maxpool1: ", size(x)); x),
            MaxPool((2,), stride=2),

            # x -> (println("before conv2: ", size(x)); x),
            Conv((10,), 3 => 6, relu, stride=1, pad=9),

            # x -> (println("before maxpool2: ", size(x)); x),
            MaxPool((2,), stride=2),

            # x -> (println("before conv3: ", size(x)); x),
            Conv((5,), 6 => 6, relu, stride=1, pad=4),

            # x -> (println("before maxpool3: ", size(x)); x),
            MaxPool((2,), stride=2),

            # LSTM expects input in the shape (features, sequence_length, batch)
            # x -> (println("before perumte: ", size(x)); x),
            x -> permutedims(x, (2, 1, 3)),

            # x -> (println("before lstm1: ", size(x)); x),
            Flux.LSTM(6 => 20),
            # x -> (println("before lstm1 output: ", size(x)); x),
            x -> x[:, end, :],
            # x -> (println("before lstm1 dropout: ", size(x)); x),
            Dropout(0.2),

            # x -> (println("before dense1: ", size(x)); x),
            Dense(20 => 10),
            # x -> (println("before dense1 dropout: ", size(x)); x),
            Dropout(0.2),

            # x -> (println("before dense2: ", size(x)); x),
            Dense(10 => 2),
            # x -> (println("before dense2 dropout: ", size(x)); x),
            Dropout(0.2),

            # x -> (println("before softmax: ", size(x)); x),
            # softmax,

            # x -> (println("out: ", size(x)); x),
        )
    end
    convlstmmodel1 = @load(NeuralNetworkClassifier, pkg = "MLJFlux", verbosity = 0)(
        builder=builder, epochs=500, batch_size=128, acceleration=CUDALibs(), optimiser=Flux.Optimisers.Adam(0.001), rng=rngseed
    )
end

# ╔═╡ e521d3cd-1939-49b9-a3ef-918ef605ad98
begin
    local builder = MLJFlux.@builder begin
        Flux.Chain(
            # x -> (println("before reshape: ", size(x)); x),
            x -> reshape(x, (seqencelength, nofeatures, :)),

            #(signallen, 1, batch) -> (L1, 16, batch)
            # x -> (println("before conv1: ", size(x)); x),
            Conv((20,), nofeatures => 3, relu, stride=1, pad=19),

            # x -> (println("before maxpool1: ", size(x)); x),
            MaxPool((2,), stride=2),

            # x -> (println("before conv2: ", size(x)); x),
            Conv((10,), 3 => 6, relu, stride=1, pad=9),

            # x -> (println("before maxpool2: ", size(x)); x),
            MaxPool((2,), stride=2),

            # x -> (println("before conv3: ", size(x)); x),
            Conv((5,), 6 => 6, relu, stride=1, pad=4),

            # x -> (println("before maxpool3: ", size(x)); x),
            MaxPool((2,), stride=2),

            # x -> (println("before flatten: ", size(x)); x),
            Flux.flatten,

            # x -> (println("before dense1: ", size(x)); x),
            Dense(786 => 10),
            # x -> (println("before dense1 dropout: ", size(x)); x),
            Dropout(0.2),

            # x -> (println("before dense2: ", size(x)); x),
            Dense(10 => 2),
            # x -> (println("before dense2 dropout: ", size(x)); x),
            Dropout(0.2),

            # x -> (println("before softmax: ", size(x)); x),
            # softmax,

            # x -> (println("out: ", size(x)); x),
        )
    end
    convmodel1 = @load(NeuralNetworkClassifier, pkg = "MLJFlux", verbosity = 0)(
        builder=builder, epochs=500, batch_size=64, acceleration=CUDALibs(), optimiser=Flux.Optimisers.Adam(0.001), rng=rngseed
    )
end

# ╔═╡ c166fbe0-3cf5-4af5-ba79-881347b336ee
begin
    xraw_mlj = coerce(xraw, Continuous)
    xfiltered_mlj = coerce(xfiltered, Continuous)
    xreconstructed_mlj = coerce(xreconstructed, Continuous)
    xreconstructed_rdp_mlj = coerce(xreconstructed_rdp, Continuous)
    ylabels_mlj = coerce(ylabels, Multiclass)
end

# ╔═╡ 2d79b5bd-34d4-4493-b7c3-f75ed577b7c1
begin
    trainindices, testindices = partition(eachindex(ylabels_mlj), 0.5, stratify=ylabels_mlj, rng=rngseed)

    x_train, y_train = xraw_mlj[trainindices, :], ylabels_mlj[trainindices]
    x_test, y_test = xraw_mlj[testindices, :], ylabels_mlj[testindices]

    x_train_fold, y_train_fold = xraw_mlj[testindices, :], ylabels_mlj[testindices]
    x_test_fold, y_test_fold = xraw_mlj[trainindices, :], ylabels_mlj[trainindices]

    xr_train, yr_train = xreconstructed_mlj[trainindices, :], ylabels_mlj[trainindices]
    xr_test, yr_test = xreconstructed_mlj[testindices, :], ylabels_mlj[testindices]

    xr_train_fold, yr_train_fold = xreconstructed_mlj[testindices, :], ylabels_mlj[testindices]
    xr_test_fold, yr_test_fold = xreconstructed_mlj[trainindices, :], ylabels_mlj[trainindices]

    xrdp_train, yrdp_train = xreconstructed_rdp_mlj[trainindices, :], ylabels_mlj[trainindices]
    xrdp_test, yrdp_test = xreconstructed_rdp_mlj[testindices, :], ylabels_mlj[testindices]

    xrdp_train_fold, yrdp_train_fold = xreconstructed_rdp_mlj[testindices, :], ylabels_mlj[testindices]
    xrdp_test_fold, yrdp_test_fold = xreconstructed_rdp_mlj[trainindices, :], ylabels_mlj[trainindices]
end

# ╔═╡ 48f84c8c-00a1-43db-a1a8-83c3edb5cd7a
function classificationscore(y_test, y_hat, truelabel)
    labelsref = y_test
    labelshat = y_hat

    accuracy = sum(labelsref .== labelshat) / length(labelsref)

    tp = sum(labelshat .== truelabel .&& labelsref .== truelabel)
    tn = sum(labelshat .!= truelabel .&& labelsref .!= truelabel)
    fp = sum(labelshat .== truelabel .&& labelsref .!= truelabel)
    fn = sum(labelshat .!= truelabel .&& labelsref .== truelabel)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1 = 2tp / (2tp + fp + fn)

    accuracy, precision, recall, specificity, f1
end

# ╔═╡ 6ae73a2d-b6f8-4b5f-b417-5e631fef5030
@kwdef struct Performance
    name::String
    accuracy::Float64
    precision::Float64
    recall::Float64
    specificity::Float64
    f1::Float64
end

# ╔═╡ 57a3aa36-d932-49c9-9b94-71900acc0691
function singlepipeline_rdp(model)::Vector{Performance}
    rdpmach1 = machine(model, xrdp_train, yrdp_train) |> fit!
    rdpmach1_fold = machine(model, xrdp_train_fold, yrdp_train_fold) |> fit!

    yhat_rdp = MLJ.predict_mode(rdpmach1, xrdp_test)
    accuracy_rdp, precision_rdp, recall_rdp, specificity_rdp, f1_rdp = classificationscore(yrdp_test, yhat_rdp, selectedpathology)
    yhat_rdp_fold = MLJ.predict_mode(rdpmach1_fold, xrdp_test_fold)
    accuracy_rdp_fold, precision_rdp_fold, recall_rdp_fold, specificity_rdp_fold, f1_rdp_fold = classificationscore(
        yrdp_test_fold, yhat_rdp_fold, selectedpathology
    )

    accuracy_rdp_cv = mean([accuracy_rdp, accuracy_rdp_fold])
    precision_rdp_cv = mean([precision_rdp, precision_rdp_fold])
    recall_rdp_cv = mean([recall_rdp, recall_rdp_fold])
    specificity_rdp_cv = mean([specificity_rdp, specificity_rdp_fold])
    f1_rdp_cv = mean([f1_rdp, f1_rdp_fold])

    [
        Performance("rdp", accuracy_rdp, precision_rdp, recall_rdp, specificity_rdp, f1_rdp),
        Performance("rdp_fold", accuracy_rdp_fold, precision_rdp_fold, recall_rdp_fold, specificity_rdp_fold, f1_rdp_fold),
        Performance("rdp_cv", accuracy_rdp_cv, precision_rdp_cv, recall_rdp_cv, specificity_rdp_cv, f1_rdp_cv)
    ]
end

# ╔═╡ 8f61c85b-ac2c-458a-a41a-14b6f02b65ce
function singlepipeline_raw(model)::Vector{Performance}
    rawmach1 = machine(model, x_train, y_train) |> fit!
    rawmach1_fold = machine(model, x_train_fold, y_train_fold) |> fit!

    yhat_raw = MLJ.predict_mode(rawmach1, x_test)
    accuracy_raw, precision_raw, recall_raw, specificity_raw, f1_raw = classificationscore(y_test, yhat_raw, selectedpathology)
    yhat_raw_fold = MLJ.predict_mode(rawmach1_fold, x_test_fold)
    accuracy_raw_fold, precision_raw_fold, recall_raw_fold, specificity_raw_fold, f1_raw_fold = classificationscore(y_test_fold, yhat_raw_fold, selectedpathology)

    accuracy_raw_cv = mean([accuracy_raw, accuracy_raw_fold])
    precision_raw_cv = mean([precision_raw, precision_raw_fold])
    recall_raw_cv = mean([recall_raw, recall_raw_fold])
    specificity_raw_cv = mean([specificity_raw, specificity_raw_fold])
    f1_raw_cv = mean([f1_raw, f1_raw_fold])

    [
        Performance("raw", accuracy_raw, precision_raw, recall_raw, specificity_raw, f1_raw),
        Performance("raw_fold", accuracy_raw_fold, precision_raw_fold, recall_raw_fold, specificity_raw_fold, f1_raw_fold),
        Performance("raw_cv", accuracy_raw_cv, precision_raw_cv, recall_raw_cv, specificity_raw_cv, f1_raw_cv),
    ]
end

# ╔═╡ 1d270e26-a542-4ddc-a8e2-ae7242190bdf
function singlepipeline_fp(model)::Vector{Performance}
    fpmach1 = machine(model, xr_train, yr_train) |> fit!
    fpmach1_fold = machine(model, xr_train_fold, yr_train_fold) |> fit!

    yhat_fp = MLJ.predict_mode(fpmach1, xr_test)
    accuracy_fp, precision_fp, recall_fp, specificity_fp, f1_fp = classificationscore(yr_test, yhat_fp, selectedpathology)
    yhat_fp_fold = MLJ.predict_mode(fpmach1_fold, xr_test_fold)
    accuracy_fp_fold, precision_fp_fold, recall_fp_fold, specificity_fp_fold, f1_fp_fold = classificationscore(
        yr_test_fold, yhat_fp_fold, selectedpathology
    )

    accuracy_fp_cv = mean([accuracy_fp, accuracy_fp_fold])
    precision_fp_cv = mean([precision_fp, precision_fp_fold])
    recall_fp_cv = mean([recall_fp, recall_fp_fold])
    specificity_fp_cv = mean([specificity_fp, specificity_fp_fold])
    f1_fp_cv = mean([f1_fp, f1_fp_fold])

    [
        Performance("fp", accuracy_fp, precision_fp, recall_fp, specificity_fp, f1_fp),
        Performance("fp_fold", accuracy_fp_fold, precision_fp_fold, recall_fp_fold, specificity_fp_fold, f1_fp_fold),
        Performance("fp_cv", accuracy_fp_cv, precision_fp_cv, recall_fp_cv, specificity_fp_cv, f1_fp_cv),
    ]
end

# ╔═╡ 2c305721-2e11-4e21-992a-a30eea7f128f
function singlepipeline(model)::Vector{Performance}
    rawmach1 = machine(model, x_train, y_train) |> fit!
    rawmach1_fold = machine(model, x_train_fold, y_train_fold) |> fit!

    yhat_raw = MLJ.predict_mode(rawmach1, x_test)
    accuracy_raw, precision_raw, recall_raw, specificity_raw, f1_raw = classificationscore(y_test, yhat_raw, selectedpathology)
    yhat_raw_fold = MLJ.predict_mode(rawmach1_fold, x_test_fold)
    accuracy_raw_fold, precision_raw_fold, recall_raw_fold, specificity_raw_fold, f1_raw_fold = classificationscore(y_test_fold, yhat_raw_fold, selectedpathology)

    accuracy_raw_cv = mean([accuracy_raw, accuracy_raw_fold])
    precision_raw_cv = mean([precision_raw, precision_raw_fold])
    recall_raw_cv = mean([recall_raw, recall_raw_fold])
    specificity_raw_cv = mean([specificity_raw, specificity_raw_fold])
    f1_raw_cv = mean([f1_raw, f1_raw_fold])

    ###

    fpmach1 = machine(model, xr_train, yr_train) |> fit!
    fpmach1_fold = machine(model, xr_train_fold, yr_train_fold) |> fit!

    yhat_fp = MLJ.predict_mode(fpmach1, xr_test)
    accuracy_fp, precision_fp, recall_fp, specificity_fp, f1_fp = classificationscore(yr_test, yhat_fp, selectedpathology)
    yhat_fp_fold = MLJ.predict_mode(fpmach1_fold, xr_test_fold)
    accuracy_fp_fold, precision_fp_fold, recall_fp_fold, specificity_fp_fold, f1_fp_fold = classificationscore(
        yr_test_fold, yhat_fp_fold, selectedpathology
    )

    accuracy_fp_cv = mean([accuracy_fp, accuracy_fp_fold])
    precision_fp_cv = mean([precision_fp, precision_fp_fold])
    recall_fp_cv = mean([recall_fp, recall_fp_fold])
    specificity_fp_cv = mean([specificity_fp, specificity_fp_fold])
    f1_fp_cv = mean([f1_fp, f1_fp_fold])

    ###

    rdpmach1 = machine(model, xrdp_train, yrdp_train) |> fit!
    rdpmach1_fold = machine(model, xrdp_train_fold, yrdp_train_fold) |> fit!

    yhat_rdp = MLJ.predict_mode(rdpmach1, xrdp_test)
    accuracy_rdp, precision_rdp, recall_rdp, specificity_rdp, f1_rdp = classificationscore(yrdp_test, yhat_rdp, selectedpathology)
    yhat_rdp_fold = MLJ.predict_mode(rdpmach1_fold, xrdp_test_fold)
    accuracy_rdp_fold, precision_rdp_fold, recall_rdp_fold, specificity_rdp_fold, f1_rdp_fold = classificationscore(
        yrdp_test_fold, yhat_rdp_fold, selectedpathology
    )

    accuracy_rdp_cv = mean([accuracy_rdp, accuracy_rdp_fold])
    precision_rdp_cv = mean([precision_rdp, precision_rdp_fold])
    recall_rdp_cv = mean([recall_rdp, recall_rdp_fold])
    specificity_rdp_cv = mean([specificity_rdp, specificity_rdp_fold])
    f1_rdp_cv = mean([f1_rdp, f1_rdp_fold])

    ###

    [
        Performance("raw", accuracy_raw, precision_raw, recall_raw, specificity_raw, f1_raw),
        Performance("raw_fold", accuracy_raw_fold, precision_raw_fold, recall_raw_fold, specificity_raw_fold, f1_raw_fold),
        Performance("raw_cv", accuracy_raw_cv, precision_raw_cv, recall_raw_cv, specificity_raw_cv, f1_raw_cv),
        Performance("fp", accuracy_fp, precision_fp, recall_fp, specificity_fp, f1_fp),
        Performance("fp_fold", accuracy_fp_fold, precision_fp_fold, recall_fp_fold, specificity_fp_fold, f1_fp_fold),
        Performance("fp_cv", accuracy_fp_cv, precision_fp_cv, recall_fp_cv, specificity_fp_cv, f1_fp_cv),
        Performance("rdp", accuracy_rdp, precision_rdp, recall_rdp, specificity_rdp, f1_rdp),
        Performance("rdp_fold", accuracy_rdp_fold, precision_rdp_fold, recall_rdp_fold, specificity_rdp_fold, f1_rdp_fold),
        Performance("rdp_cv", accuracy_rdp_cv, precision_rdp_cv, recall_rdp_cv, specificity_rdp_cv, f1_rdp_cv)
    ]
end

# ╔═╡ 26b88169-076f-433d-a760-64cd7743d393
function wholepipeline(model; repeat::Int=10)
    allresults = []
    for i in 1:repeat
        performance = singlepipeline(model)
        push!(allresults, performance)
        @info "[$i]: $performance"
    end

    totalperfornamce = OrderedDict(map(x -> x.name => OrderedDict(), first(allresults)))
    for key in keys(totalperfornamce)
        for field in fieldnames(Performance)
            if field != :name
                totalperfornamce[key][field] = []
            end
        end
    end
    for result in allresults
        for performance in result
            for field in fieldnames(Performance)
                if field != :name
                    push!(totalperfornamce[performance.name][field], getproperty(performance, field))
                end
            end
        end
    end

    meanperfornamce = OrderedDict(
        map(first(allresults)) do x
            x.name => Performance(
                x.name,
                mean(totalperfornamce[x.name][:accuracy]),
                mean(totalperfornamce[x.name][:precision]),
                mean(totalperfornamce[x.name][:recall]),
                mean(totalperfornamce[x.name][:specificity]),
                mean(totalperfornamce[x.name][:f1]),
            )
        end
    )
    stdperfornamce = OrderedDict(
        map(first(allresults)) do x
            x.name => Performance(
                x.name,
                std(totalperfornamce[x.name][:accuracy]),
                std(totalperfornamce[x.name][:precision]),
                std(totalperfornamce[x.name][:recall]),
                std(totalperfornamce[x.name][:specificity]),
                std(totalperfornamce[x.name][:f1]),
            )
        end
    )

    meanperfornamce, stdperfornamce
end

# ╔═╡ 6a70f7ec-7d99-4aa8-aa7c-65fb20d39fc8
function multipipeline_rdp(model; repeat::Int=10)
    allresults = []
    for i in 1:repeat
        performance = singlepipeline_rdp(model)
        push!(allresults, performance)
        @info "[$i]: $performance"
    end

    allresults
end

# ╔═╡ 72d2765b-0ada-4e1f-9853-2641b5a01e49
function multipipeline_raw(model; repeat::Int=10)
    allresults = []
    for i in 1:repeat
        performance = singlepipeline_raw(model)
        push!(allresults, performance)
        @info "[$i]: $performance"
    end

    allresults
end

# ╔═╡ 2c130ab8-f4a0-41a8-bf71-ed09d889d435
function multipipeline_fp(model; repeat::Int=10)
    allresults = []
    for i in 1:repeat
        performance = singlepipeline_fp(model)
        push!(allresults, performance)
        @info "[$i]: $performance"
    end

    allresults
end

# ╔═╡ be00e8ad-4f14-4478-b8d8-3c1f8db23471
# ╠═╡ show_logs = false
# ╠═╡ disabled = true
#=╠═╡
begin
    convlstmmodel1_meanperfornamce, convlstmmodel1_stdperfornamce = wholepipeline(convlstmmodel1; repeat=5)
end
  ╠═╡ =#

# ╔═╡ 17976815-c487-4762-bf28-32ad1edcd433
# ╠═╡ disabled = true
#=╠═╡
begin
    open(joinpath("..", "results", "$(lowercase(selectedpathology[2:end]))_convlstmmodel1_meanperfornamce.json"), "w") do f
        write(f, JSON.json(convlstmmodel1_meanperfornamce))
    end
    open(joinpath("..", "results", "$(lowercase(selectedpathology[2:end]))_convlstmmodel1_stdperfornamce.json"), "w") do f
        write(f, JSON.json(convlstmmodel1_stdperfornamce))
    end
end
  ╠═╡ =#

# ╔═╡ da001aeb-bd36-4a24-80b4-fa783db09912
# ╠═╡ show_logs = false
# ╠═╡ disabled = true
#=╠═╡
begin
    convmodel1_meanperfornamce, convmodel1_stdperfornamce = wholepipeline(convmodel1; repeat=5)
end
  ╠═╡ =#

# ╔═╡ 0e87e353-a79c-4ed0-9ac6-83d00b7ab93d
# ╠═╡ disabled = true
#=╠═╡
begin
    open(joinpath("..", "results", "$(lowercase(selectedpathology[2:end]))_convmodel1_meanperfornamce.json"), "w") do f
        write(f, JSON.json(convmodel1_meanperfornamce))
    end
    open(joinpath("..", "results", "$(lowercase(selectedpathology[2:end]))_convmodel1_stdperfornamce.json"), "w") do f
        write(f, JSON.json(convmodel1_stdperfornamce))
    end
end
  ╠═╡ =#

# ╔═╡ 4dcd8a90-248a-4488-96f6-99bdefce0775
# multipipeline_raw(convlstmmodel1; repeat=5)
# multipipeline_fp(convlstmmodel1; repeat=5)
# multipipeline_rdp(convlstmmodel1; repeat=5)

# ╔═╡ 0581b49a-6e30-4886-87ba-12e4f307e121
# multipipeline_raw(convmodel1; repeat=25)

# ╔═╡ ef818394-1f8f-4af5-a5ba-7b99fd85ea00
# multipipeline_fp(convmodel1; repeat=25)

# ╔═╡ 835baa43-ae96-4f68-a372-b2420eed7e98
# multipipeline_rdp(convmodel1; repeat=25)

# ╔═╡ Cell order:
# ╟─f923a8c6-f2ed-41b8-9bd6-f9830790faec
# ╟─41372110-ff60-11ef-1068-e92f06552b64
# ╠═57f5fc4e-bf83-4e83-95bb-dd4c44c4b52f
# ╠═c8f39317-2b0d-47e7-8dc8-52eebbf6193d
# ╟─6cb60280-d193-4756-9e84-1e6bf427379a
# ╠═9b599684-619d-49a2-8844-e5fd18af52a2
# ╠═32af21c7-f656-4770-87cc-7b084fd57580
# ╠═20056417-66fa-4de9-b35c-d235b0afcc0a
# ╠═b53c8508-f45a-460b-8b70-bd8cb3d4778b
# ╠═9f00fc06-f9b5-4fbb-a8da-be48c5c81e01
# ╠═df73ebe6-b9d7-4aa2-b8f8-d5b51ea32480
# ╠═cbfa601b-a8f9-499d-af05-8d5f9ff6ce1b
# ╠═2d68804a-c210-448f-a99d-bfde5670ada0
# ╟─d572be5a-1c16-4a1f-b733-44367fa05b8a
# ╟─8b654cdd-8d27-40c8-932e-5752d394540f
# ╟─2b1b21a3-97e5-4475-9963-e5794489f0c8
# ╠═bb20f631-298e-4072-bcd0-0a8ed1aa954c
# ╠═05de5742-b453-4b20-a156-42c53373974d
# ╟─46ab0c02-74ac-434c-a525-7f01312e32ea
# ╠═f9baee8e-8c0a-4e6f-b9a2-027fbdc04519
# ╠═ec9fb06f-1c90-4463-bd58-6bd1d8177a8f
# ╠═bfa6bd3e-93e0-491b-8e3a-1bc009e32858
# ╟─1309496c-f270-44d2-bf9a-d775ab177076
# ╠═71b193d8-8481-491a-847c-cbf49041c3d3
# ╠═299034a7-cea1-4edb-a88f-43003bc34ca7
# ╠═a54fd49b-40db-43d0-816b-9dd0a38a805c
# ╠═81573a74-2b76-4365-a219-db7ed4da982a
# ╠═9afe58ed-8a5c-4660-a262-b21335022339
# ╟─56ed382b-7f26-45a6-b0e3-ec6f19f0f890
# ╟─efee10e4-8272-4717-b4ea-f4f53cd889ca
# ╠═deeb8466-d89d-45d5-bd7d-ffef89c497e4
# ╠═73302630-7c63-4a95-9196-82d15f3c3905
# ╠═2571d757-0562-4bc6-aaa5-1b2160e9f1ce
# ╠═76f98311-86ad-4a15-891f-1cedb5d82713
# ╠═3da44940-b063-4acf-a4eb-1ef94d0a3cbe
# ╠═2c79044d-6c38-4d4f-bcd5-febd40993831
# ╟─e7e4d8fc-f344-4495-b023-83a2ca66d885
# ╟─1a67db99-ffc8-456d-a6c3-fd0da8755ace
# ╠═13260533-5a77-4479-9378-44e7a1c4a861
# ╠═94cfd77d-92d8-4c19-b3c1-897a5aa0dcbb
# ╟─103fe3b0-1010-401e-97ea-ea98d4a67303
# ╠═946022e6-382b-4b75-a76c-75c1625110a0
# ╠═5c315695-291b-42a9-92c1-9db0323ffd4e
# ╠═86beeb4b-c375-4a51-9c72-5c2754e764d4
# ╠═e521d3cd-1939-49b9-a3ef-918ef605ad98
# ╠═c166fbe0-3cf5-4af5-ba79-881347b336ee
# ╠═2d79b5bd-34d4-4493-b7c3-f75ed577b7c1
# ╠═48f84c8c-00a1-43db-a1a8-83c3edb5cd7a
# ╠═6ae73a2d-b6f8-4b5f-b417-5e631fef5030
# ╠═57a3aa36-d932-49c9-9b94-71900acc0691
# ╠═8f61c85b-ac2c-458a-a41a-14b6f02b65ce
# ╠═1d270e26-a542-4ddc-a8e2-ae7242190bdf
# ╟─2c305721-2e11-4e21-992a-a30eea7f128f
# ╟─26b88169-076f-433d-a760-64cd7743d393
# ╠═6a70f7ec-7d99-4aa8-aa7c-65fb20d39fc8
# ╠═72d2765b-0ada-4e1f-9853-2641b5a01e49
# ╠═2c130ab8-f4a0-41a8-bf71-ed09d889d435
# ╠═be00e8ad-4f14-4478-b8d8-3c1f8db23471
# ╟─17976815-c487-4762-bf28-32ad1edcd433
# ╠═da001aeb-bd36-4a24-80b4-fa783db09912
# ╟─0e87e353-a79c-4ed0-9ac6-83d00b7ab93d
# ╠═4dcd8a90-248a-4488-96f6-99bdefce0775
# ╠═0581b49a-6e30-4886-87ba-12e4f307e121
# ╠═ef818394-1f8f-4af5-a5ba-7b99fd85ea00
# ╠═835baa43-ae96-4f68-a372-b2420eed7e98
