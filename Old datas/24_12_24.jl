using DrWatson
#@quickactivate :intleaved25
using ITensors, HDF5, Plots, LinearAlgebra, Kronecker, Dates, ITensorMPS, LaTeXStrings, FFTW, ZChop, IteratorSampling, PolyChaos, Polynomials, QuadGK

ITensors.disable_warn_order()

let    #@
    include("function_intl.jl")
    #-------------------------------------------------------------------------------------------------------
    #All system parameters 
    dh = 2^8 # number of level(s) considered in each HO
    nt = Int(log(2, dh))
    nb = 2 #number of bath  
    ek = 100 #length of the bath chain  
    #------------------------------------------------------------------------------------------------------   
    # ITensor space 
    # Dot position
    sysp = Int(nt + 1)
    # Pbose
    bpose_en = 1
    bpose_st = Int(nt)
    tot_chain = Int((1 + 2 * nb * ek + nt))
    stbath = [(n == sysp) ? Index(2, "Fermion, dot") : ((n >= bpose_en) && (n <= bpose_st)) ? Index(2, "Boson") : Index(2, "Fermion, bath") for n = 1:tot_chain]
    #-------------------------------------------------------------------------------------------------------
    # q-dot 
    #Ec = 5
    ξ = 1 # energy of the qubit
    ϵ = ξ
    # phonon
    ω0 = 1
    λ = 2
    #η = 10          #__@__
    #chemical potentials 
    μ0 = 0
    eV = 1
    Δ = eV / 2
    μ1 = μ0 + Δ
    μ2 = μ0 - Δ
    #-------------------------------------------------------------------------------------------------------
    # State specifications 
    #ex_s = "Emp" # excitation of the fermion = initial occupation
    #ρ_en = [1 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 0]  ## Electron
    ρ_en = [1 0; 0 1]    ## Fermion
    ρ_en /= tr(ρ_en)
    ex_ph = 0
    #-------------------------------------------------------------------------------------------------------
    # Bath specifications 
    Γ0_list = [1. 1.]
    δ_list = [1 1]
    ω_bath_list = [μ1 μ2]
    μ_list = [μ1 μ2]
    #-------------------------------------------------------------------------------------------------------
    # Evolution parameters
    N_coeff = 160
    tau = 0.01             ## time step duration
    nst = 2000
    maxdim1 = 40
    maxdim2 = 30    #__@__
    cutoff1 = 1E-9
    #-------------------------------------------------------------------------------------------------------
    # Runge Kutta specifications    
    h = 0.01
    hstep = max(Int(div(tau, h)), Int(1))
    #-------------------------------------------------------------------------------------------------------
    # Truncation parameters
    # kotota truncate hobe?
    max_frac = 0.995
    # koto dur thek hobe?
    p_gap = 10
    # koto steep hobe?
    steepness = 2
    #= #------------------------------------------------------------------------------------------------------- 
    println("-----------------------------------------------------------------------------------------------------------------------------")
    println("N_chain = $ek, ξ = ϵ = μ0 = $ξ, eV = $eV, Γ0 = $(Γ0_list[1]), β = $(β_list[1]), δ = $(δ_list[1]), ω_bath = $(ω_bath_list[1]), τ = $tau, T_st = $nst")
    println("-----------------------------------------------------------------------------------------------------------------------------")
    #------------------------------------------------------------------------------------------------------- =#
    #state preparation 
    ITensors.op(::OpName"ρdot", ::SiteType"Fermion") = ρ_en
    ITensors.op(::OpName"0", ::SiteType"Fermion") = [1 0; 0 0]
    ITensors.op(::OpName"1", ::SiteType"Fermion") = [0 0; 0 1]
    ITensors.op(::OpName"0", ::SiteType"Boson", d::Int) = [1 0; 0 0]
    ITensors.op(::OpName"1", ::SiteType"Boson", d::Int) = [0 0; 0 1]
    #-------------------------------------------------------------------------------------------------------
    r = reverse(findr(ex_ph, dh))
    #states1 = [i == nt + 1 ? "ρdot" : (Int(r[i]) == 0 ? "0" : "1") for i = 1:nt+1]
    #states = [string("0")]
    states = [i == nt + 1 ? "ρdot" : (Int(r[i]) == 0 ? "0" : "1") for i = 1:nt+1]               #__@__
    for i in sysp+1:tot_chain
        append!(states, ["0"])
    end
    ρ0 = MPO(stbath, states) #initial state
    #-------------------------------------------------------------------------------------------------------
    #Threads.@threads for temps in [5]
    temps = 1.5
    β_list = [1/temps 1/temps]   #__@__
    #------------------------------------------------------------------------------------------------------- 
    println("-----------------------------------------------------------------------------------------------------------------------------")
    println("N_chain = $ek, ξ = ϵ = μ0 = $ξ, eV = $eV, Γ0 = $(Γ0_list[1]), β = $(β_list[1]), δ = $(δ_list[1]), ω_bath = $(ω_bath_list[1]), τ = $tau, T_st = $nst")
    println("-----------------------------------------------------------------------------------------------------------------------------")
    #-------------------------------------------------------------------------------------------------------
    # File names
    today1 = string(Dates.format(Dates.now(), "yyyy-mm-dd"))
    file_name_txt = string(split(split(@__FILE__, ".")[end-1], string('\\'))[end], "_$temps.txt")
    file_name_txt_1 = string(split(split(@__FILE__, ".")[end-1], string('\\'))[end], "_all_$temps.txt")
    #file_name_txt_2 = string(split(split(@__FILE__, ".")[end-1], string('\\'))[end], "_MPO_$temps")
    file_name_txt_2(x, y) = string(split(dirname(@__FILE__), '\\')[end], "/$y/", "$x", "_", "$today1", "_$(temps)_.png")
    #-------------------------------------------------------------------------------------------------------  
    # recurrance coefficients 
    recurr_list = call_recurr_in(ek, Γ0_list, β_list, μ_list, δ_list, ω_bath_list, N_coeff, dh, nb)
    #-------------------------------------------------------------------------------------------------------
    # gates
    chain_ham = unit_gates_fm_in(recurr_list, ek, dh, tau, nb, stbath)
    sys_ham = Hs_chain_gen_fm_in(stbath, ϵ, ω0, λ, dh)
    #---------------------------------------------------------------------------
    # observable
    pop2 = OpSum()
    for n in 1:nt # running for core system length     
        pop2 += (2)^(n - 1), "Adag", bpose_st - n + 1, "A", bpose_st - n + 1
    end
    pop2_op = MPO(pop2, stbath)
    #-------------------------------------------------------------------------------------------------------
    # EVOLUTION

    # initialization
    ρ = deepcopy(ρ0)                               #__@__
    pop_ph = []
    times = []
    times_plot = []
    t_list = []
    pop_plot = []
    pd = exp(loginner(pop2_op, ρ0))
    append!(pop_ph, pd)
    append!(pop_plot, pd)
    append!(times, [string(Dates.format(now(), "HH:MM"))])
    append!(t_list, 0)
    println("Time = ", 0)
    println("Phonon = ", pd)
    write_for_loop(file_name_txt, string(1), "cutoff = $cutoff1, maxdim = $maxdim1, dt = $tau, T = $temps", string(pop_ph))

    # evolution 
    for t = 1:nst
        #RKM
        for i in 1:hstep
            ρ = RKM_rho_in(ρ, sys_ham, h, cutoff1, maxdim1)
        end
        #-------------------------------------------------------------------------------------------------------
        # Unitary evolution
        ρ = apply(chain_ham, ρ; cutoff=cutoff1, maxdim=maxdim1, apply_dag=true)
        #-------------------------------------------------------------------------------------------------------
        # Hermiticity enforcement, truncation, orthogonalize
        if t % 10 == 0
            ρ = normalize(0.5 * add(swapprime(dag(ρ), 0 => 1), ρ; maxdim=200))
            ρ = dynamical_trunc_MP_in(ρ, maxdim1, maxdim2, steepness, max_frac, sysp, p_gap)     #__@__
        end
        orthogonalize!(ρ, Int(floor((bpose_st + bpose_en) / 2)))
        #-------------------------------------------------------------------------------------------------------
        if t % 20 == 0
            # Measurement
            pd = exp(loginner(pop2_op, ρ; cutoff=cutoff1))
            append!(pop_ph, pd)
            println("Time = ", t * tau)
            println("Phonon = ", pd)
            println("beta = ", temps)
            write_for_loop(file_name_txt_1, string(t + 1), string(t * tau), string(pd))
            #@show maxlinkdim(ρ)
        end
        #-------------------------------------------------------------------------------------------------------
        # Population plotting 
        if t % 100 == 0
            append!(t_list, t * tau)
            append!(pop_plot, pd)
            append!(times, [string(Dates.format(now(), "HH:MM"))])

            p1 = plot()
            title_string = L"\Gamma _{0} = %$(Γ0_list[1]), T = %$temps, m_{dim} = %$maxdim1, trunc = [%$(p_gap), \ %$(max_frac)]"
            p1 = plot(t_list, real(pop_plot), xticks=(t_list, string.(times)), title=title_string, xlabel=L"t", ylabel=L"n_{ph}", legend=false)
            display("image/png", p1)
            safesave(file_name_txt_2("ph_pop", "pop_plots_$temps"), p1)
#= 
            p3 = plot()
            title_string = L"\Gamma _{0} = %$(Γ0_list[1]), \beta = %$temps, m_{dim} = %$maxdim1, trunc = %$(p_gap), t = %$(t_list[end])"
            p3 = plot([bpose_en, sysp, sysp, bpose_en, bpose_en], [0, 0, maxdim1, maxdim1, 0], linewidth=0, color=:pink, seriestype=:shape, title=title_string, xlabel=L"sites", ylabel=L"\chi", legend=false)
            p3 = plot!(linkdims(ρ))

            display("image/png", p3)
            safesave(file_name_txt_2("linkdim_$t", "linkdim_plots_$temps"), p3) =#
        end
        #-------------------------------------------------------------------------------------------------------
        # MPO store 
        #= if t % 500 == 0                                             #__@__
            name_string = string("$file_name_txt_2", "_$t", ".h5")
            fw = h5open("$name_string", "w")
            [write(fw, "ρ$i", ρ[i]) for i = 1:eachindex(ρ)[end]]
            #fr = h5open("rho_29_11.h5","r");
            #readMPO = MPO(stbath);
            #[readMPO[i]=read(fr,"ρ$i",ITensor) for i=1:eachindex(readMPO)[end]];
        end =#
    end

    write_for_loop(file_name_txt, string(2), "cutoff = $cutoff1, T = $temps, maxdim = $maxdim1, dt = $tau", string(real(pop_ph)))

    #-------------------------------------------------------------------------------------------------------
    # Final plotting and phonon density matrix 
    p1 = plot()
    title_string = L"\Gamma _{0} = %$(Γ0_list[1]), T = %$temps, m_{dim} = %$maxdim1, trunc = %$(p_gap)"
    p1 = plot(collect(0:1:length(real(pop_ph))-1) * tau, real(pop_ph), title=title_string, xlabel=L"t", ylabel=L"n_{ph}", legend=false)
    display("image/png", p1)

    rho_red = findrho2_en_in(ρ, stbath, dh, ek, nb)
    ρ_mod = broadcast(abs, rho_red)
    filename = string(Dates.format(Dates.now(), "dduyy-HH:MM"), "_data_rho_l_beta_$temps.txt")
    write_for_loop(filename, "1", "cutoff = $cutoff1, maxdim = $maxdim1, dt = $tau", string(ρ_mod))
    p2 = plot()
    p2 = heatmap(ρ_mod, yflip=true, c=:acton10, clims=(0, 0.01))
    display("image/png", p2)

    safesave(file_name_txt_2("ph_pop_fin", "pop_plots_beta_$temps"), p1)
    safesave(file_name_txt_2("ph_rho_fin", "mat_plots_beta_$temps"), p2)
    #-------------------------------------------------------------------------------------------------------

    #end

end
