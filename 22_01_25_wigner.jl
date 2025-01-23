using DrWatson
using ITensors, HDF5, Plots, LinearAlgebra, Kronecker, Dates, ITensorMPS, LaTeXStrings, FFTW, ZChop, IteratorSampling, PolyChaos, Polynomials, QuadGK
using SpecialFunctions, SparseArrays
using QuantumToolbox
ITensors.disable_warn_order()
#-------------------------------------------------------------------------------------------------------
function wigner_tbox(state::Any, xvec::AbstractVector, yvec::AbstractVector; g::Real = √2,method::WignerSolver = WignerClenshaw())
	stateobj = Qobj((1.0 + 1.0*im)*state  |> dense_to_sparse)
	wig = wigner(stateobj,xvec,yvec;g = g, method=method)
	return wig 
end
#-------------------------------------------------------------------------------------------------------
let  
	include("function_intl.jl")
	include("wigner_function.jl")
	#-------------------------------------------------------------------------------------------------------
	# Phase-space for Wigner function
	xvec = range(-12, 12, length = 150)  # x-coordinates
	yvec = range(-12, 12, length = 150)  # y-coordinates
	#-------------------------------------------------------------------------------------------------------
	#All system parameters 
	dh = 2^7 # number of level(s) considered in each HO
	nt = Int(log(2, dh))
	nb = 2 #number of bath  
	ek = 80 #length of the bath chain  
	#------------------------------------------------------------------------------------------------------   
	# ITensor space 
	# Dot position
	sysp = Int(nt + 1)
	# Pbose
	bpose_en = 1
	bpose_st = Int(nt)
	tot_chain = Int((1 + 2 * nb * ek + nt))
	stbath = [(n == sysp) ? Index(2, "Fermion, dot") : ((n >= bpose_en) && (n <= bpose_st)) ? Index(2, "Boson") : Index(2, "Fermion, bath") for n ∈ 1:tot_chain]
	#-------------------------------------------------------------------------------------------------------
	# q-dot 
	#Ec = 5
	ξ = 1 # energy of the qubit
	ϵ = ξ
	# phonon
	ω0 = 1
	λ = 2
	#η = 10        
	#chemical potentials 
	μ0 = 0
	eV = 5
	Δ = eV / 2
	μ1 = μ0 + Δ
	μ2 = μ0 - Δ
	#-------------------------------------------------------------------------------------------------------
	# State specifications 
	#ex_s = "Emp" # excitation of the fermion = initial occupation
	#ρ_en = [1 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 0]  ## Electron
	ρ_en = [1 0; 0 1]    ## Fermion
	ρ_en /= tr(ρ_en)
	ex_ph = 80
	#-------------------------------------------------------------------------------------------------------
	# Bath specifications 
	Γ0_list = [1.0 1.0]
	δ_list = [1.0 1.0]
	ω_bath_list = [μ1 μ2]
	μ_list = [μ1 μ2]
	#-------------------------------------------------------------------------------------------------------
	# Evolution parameters
	N_coeff = 100               ##*********************##
	tau = 0.01             ## time step duration
	nst = 2500
	maxdim1 = 120
	maxdim2 = 20    #__@__
	cutoff1 = 1E-10
	#-------------------------------------------------------------------------------------------------------
	# Runge Kutta specifications    
	h = 0.002
	hstep = max(Int(div(tau, h)), Int(1))
	#-------------------------------------------------------------------------------------------------------
	# Truncation parameters
	# kotota truncate hobe?
	max_frac = 0.95
	# koto dur thek hobe?
	p_gap = 10
	# koto steep hobe?
	steepness = 2
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
	states = [i == nt + 1 ? "ρdot" : (Int(r[i]) == 0 ? "0" : "1") for i ∈ 1:nt+1]              
	for i in sysp+1:tot_chain
		append!(states, ["0"])
	end
	ρ0 = MPO(stbath, states) #initial state
	@show tr(ρ0)
	#-------------------------------------------------------------------------------------------------------
	#Threads.@threads for temps in [5]
	temps = 1.5
	β_list = [1/temps 1/temps]   
	#------------------------------------------------------------------------------------------------------- 
	println("-----------------------------------------------------------------------------------------------------------------------------")
	println("N_chain = $ek, ξ = ϵ = μ0 = $ξ, eV = $eV, Γ0 = $(Γ0_list[1]), β = $(β_list[1]), δ = $(δ_list[1]), ω_bath = $(ω_bath_list[1]), τ = $tau, T_st = $nst")
	println("-----------------------------------------------------------------------------------------------------------------------------")
	#-------------------------------------------------------------------------------------------------------
	# File names
		today1 = string(Dates.format(Dates.now(), "yyyy-mm-dd"))
		file_name_txt = string(split(split(@__FILE__, ".")[end-1], string('\\'))[end], "_popend_$temps.txt")
		file_name_txt_1 = string(split(split(@__FILE__, ".")[end-1], string('\\'))[end], "_allpop_$temps.txt")
		file_name_txt_2 = string(split(split(@__FILE__, ".")[end-1], string('\\'))[end], "_pop_$temps.png")
		file_name_txt_3 = string(split(split(@__FILE__, ".")[end-1], string('\\'))[end], "_wig_$temps.png")
		file_name_txt_fun(x, y) = string(split(dirname(@__FILE__), '\\')[end], "/$y/", "$x", "_", "$today1", "_$(temps)_.png")

	# Plot the Wigner function as a heatmap
		# Define custom colors with white for zero
		color_min = RGB(50 / 255, 200 / 255, 100 / 255)   # Bright green for negative values
		color_mid = RGB(1, 1, 1)                    # White for zero
		color_max = RGB(255 / 255, 100 / 255, 100 / 255)  # Bright pink for positive values

		# Create a custom color gradient with white at the midpoint
		custom_colormap_with_white = cgrad([color_min, color_mid, color_max], [0.0, 0.5, 1.0])

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
	ρ = deepcopy(ρ0)                             
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
	write_for_loop(file_name_txt_1, string(1), "cutoff = $cutoff1, maxdim = $maxdim1, dt = $tau, T = $temps", string(pop_ph))

	# evolution 
	for t ∈ 1:nst
		#RKM
		for i in 1:hstep
			ρ = RKM_rho_in(ρ, sys_ham, h, cutoff1, maxdim1)
		end
		#-------------------------------------------------------------------------------------------------------
		# Unitary evolution
		ρ = apply(chain_ham, ρ; cutoff = cutoff1, maxdim = maxdim1, apply_dag = true)
		#-------------------------------------------------------------------------------------------------------
		# Hermiticity enforcement, truncation, orthogonalize
		if t % 10 == 0
			ρ = 0.5 * add(swapprime(ITensors.dag(ρ), 0 => 1), ρ; maxdim = 200)
			ρ /= tr(ρ)
			#ρ = normalize(dynamical_trunc_MP_in(ρ, maxdim1, maxdim2, steepness, max_frac, sysp, p_gap))     
		end
		orthogonalize!(ρ, Int(floor((bpose_st + bpose_en) / 2)))
		#-------------------------------------------------------------------------------------------------------
		# Measurement
			pd = exp(loginner(pop2_op, ρ; cutoff = cutoff1))
			append!(pop_ph, pd)
			append!(times, [string(Dates.format(now(), "HH:MM"))])
			write_for_loop(file_name_txt_1, string(t + 1), string(t * tau), string(real(pd)))
		#-------------------------------------------------------------------------------------------------------
		if t % 150 == 1
			println("Time = ", t * tau)
			println("Phonon = ", pd)
			#println("beta = ", temps)
			list = linkdims(ρ)
			@show link_front = length(list) - findfirst(==(findmax(list)[1]), reverse(list))[1] + 1
			#-------------------------------------------------------------------------------------------------------
			# Calculate Wigner Function
			density_matrix = findrho2_en_in(ρ, stbath, dh)
			@show tr(density_matrix)
			# Compute the Wigner function
			W = wigner_clenshaw(density_matrix, xvec, yvec, sqrt(2), false)
			dx = step(xvec)  # Grid spacing in x-direction
            dy = step(yvec)  # Grid spacing in y-direction
			#W1 = wigner_tbox(density_matrix,xvec,yvec)
            @show sum(W) * dx * dy
			#-------------------------------------------------------------------------------------------------------
			# Plot Wigner function
			p = heatmap(
				xlim = (minimum(xvec), maximum(xvec)), ylim = (minimum(yvec), maximum(yvec)),
				xvec, yvec, W,
				color = custom_colormap_with_white,  # Custom colormap with white at zero
				xlabel = L"x", ylabel = L"p",
				title = L"W(x,p), d = %$(dh), t = %$(t*tau)",
				#colorbar_title="Value",           # Label for the color bar
				clim = (-0.2, 0.2),                # Adjust the range to highlight negative regions
				aspect_ratio = 1.0,               # Set the aspect ratio to 1:1
			)
			display("image/png", p) # heatmap(W))
			safesave(file_name_txt_3, p)
			#-------------------------------------------------------------------------------------------------------
			# Plot Population 
			p2 = plot()
			title_string = L"\Gamma _{0} = %$(Γ0_list[1]), T = %$temps, m_{dim} = %$maxdim1, trunc = %$(p_gap)"
			p2 = plot(times, real(pop_ph), title = title_string, xlabel = L"t", ylabel = L"n_{ph}", legend = false)
			display("image/png", p2)
			safesave(file_name_txt_2, p2)
		end
	end

	#write_for_loop(file_name_txt, string(2), "cutoff = $cutoff1, T = $temps, maxdim = $maxdim1, dt = $tau", string(real(pop_ph)))

	#-------------------------------------------------------------------------------------------------------
	# Final plotting and phonon density matrix 
		p1 = plot()
		title_string = L"\Gamma _{0} = %$(Γ0_list[1]), T = %$temps, m_{dim} = %$maxdim1, trunc = %$(p_gap)"
		p1 = plot(collect(0:1:length(real(pop_ph))-1) * tau, real(pop_ph), title = title_string, xlabel = L"t", ylabel = L"n_{ph}", legend = false)
		display("image/png", p1)

		rho_red = findrho2_en_in(ρ, stbath, dh)
		ρ_mod = broadcast(abs, rho_red)
		filename = string(Dates.format(Dates.now(), "dduyy-HH:MM"), "_data_rho_l_beta_$temps.txt")
		write_for_loop(filename, "1", "cutoff = $cutoff1, maxdim = $maxdim1, dt = $tau", string(round.(ρ_mod*10^7)*10^(-7)))
		p4 = plot()
		p4 = heatmap(ρ_mod, yflip = true, c = :acton10, clims = (0, 0.01))
		display("image/png", p4)

		safesave(file_name_txt_fun("ph_pop_fin", "pop_plots_beta_$temps"), p1)
		safesave(file_name_txt_fun("ph_rho_fin", "mat_plots_beta_$temps"), p4)
	#-------------------------------------------------------------------------------------------------------
end


