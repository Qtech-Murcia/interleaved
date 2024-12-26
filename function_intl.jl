######################################################################################
function gen_state2(alfa, dh)
    #-------------------------------------------------------------------------------------------------------         
    # alfa = the excitation of the phonon state 
    # dh = the dimension of the phonon mode
    #-------------------------------------------------------------------------------------------------------     
    #n = number of pseudobosons
    n = Int(log(2, dh))
    r = findr(alfa, dh)
    states = [(r[i] == 0) ? "0" : "1" for i = 1:n]
    return states
end
##########################################################################
function findr(alfa, dh)# alfa = the excitation of the phonon state; n= number of pseudobosons
    n = Int(log(2, dh))
    if alfa <= dh - 1
        r = digits(alfa, base=2, pad=n)
        return r
    else
        print("error")
    end
end
##########################################################################
function recur_coeff(w_fn, supp, N_coeff, Nquad)
    my_meas = Measure("my_meas", w_fn, supp, false, Dict())

    my_op = OrthoPoly("my_op", N_coeff - 1, my_meas; Nquad)

    return PolyChaos.coeffs(my_op)
end
##########################################################################
function alfa(A, d)
    y = 0
    for i in 1:d
        y = y + 2^(i - 1) * A[i]
    end
    return y
end
##########################################################################
function findinstance(N)
    rlist = Iterators.product(fill(0:1, N)...) |> collect |> vec

    rselect = []#Array{Int64}(undef, 0, 0)

    for i in 0:(2)^(N)-1 #N+1
        for j in 1:(2)^(N)
            if alfa(rlist[j], N) == i
                rselect = push!(rselect, rlist[j])
            else
                continue
            end
        end
    end
    return rselect
end
##########################################################################
function dynamical_trunc_MP_in(rho, maxdim1, maxdim2, steepness, max_frac, sysp, p_gap)
    tot_gap = 2 * p_gap
    rho = ITensors.truncate(rho; maxdim=maxdim1)
    list = linkdims(rho)
    link_front = length(list) - findfirst(==(findmax(list)[1]), reverse(list))[1] + 1

    if (list[Int(sysp + tot_gap)] == maxdim1)
        #--------------------------------------------------------------------------------------------
        for i in Int(sysp + p_gap):Int(length(list)[1] - 1)
            if list[i] - list[i+1] < steepness && i > Int(floor((4*link_front + sysp + p_gap) / 5))     #__@__
                rho = ITensors.truncate(rho; maxdim=max(Int(floor(list[i] * max_frac)), maxdim2), site_range=i:length(list))
            else
                continue
            end
            list = linkdims(rho)
        end
        #--------------------------------------------------------------------------------------------
        return rho
    else
        return rho
    end
end
##########################################################################
function RKM_rho_in(y0, L, h, cutoff1, maxdim1)                            #__@__
    #Apply Runge Kutta Formulas to find next value of y
    f(x) = -im * add(apply(L, x; cutoff=cutoff1, maxdim=maxdim1), apply(x, -L; cutoff=cutoff1, maxdim=maxdim1); cutoff=0.1 * cutoff1, maxdim=200)

    k1 = h * f(y0)
    k2 = h * f(add(y0, 0.5 * k1; cutoff=0.1 * cutoff1, maxdim=200))
    k3 = h * f(add(y0, 0.5 * k2; cutoff=0.1 * cutoff1, maxdim=200))
    k4 = h * f(add(y0, k3; cutoff=0.1 * cutoff1, maxdim=200))
    # Update next value of y
    y = (1.0 / 6.0) * add(6.0 * y0, k1, 2 * k2, 2 * k3, k4; cutoff=0.1 * cutoff1, maxdim=200)
    return y
end
##########################################################################
function findrho2_en_in(ρ_in, stbath, dh)
    rho_in = mode_only_en_in(dh, ρ_in, stbath)
    nt = Int(log2(dh))
    st = stbath[1:nt]
    rho = Matrix{ComplexF64}(undef, dh, dh)

    for row = (1:dh)
        for col = (1:dh)
            el = Matrix{Any}(undef, 2 * nt, 1)
            el[1:nt] = collect(gen_state2(row - 1, dh))
            el[nt+1:2*nt] = collect(gen_state2(col - 1, dh))
            x = MPS(st, collect(el[1:nt]))
            y = MPS(st, collect(el[(1+nt):end]))'
            V = inner(x, rho_in, y)
            rho[row, col] = V
        end
    end
    return rho
end
##########################################################################
function mode_only_en_in(dh, ρ, stbath)
    nt = Int(log2(dh))
    tot_chain = lastindex(stbath)
    Φ = MPO(nt)
    x = ITensor(1.0)
    for i in 1+nt:tot_chain
        x *= ρ[i] * delta(stbath[i]', stbath[i])
    end
    for i = 1:nt
        if i == nt
            Φ[i] = ρ[i] * x
        else
            Φ[i] = ρ[i]
        end
    end
    return Φ
end
##########################################################################
function states_en2_in(ex_ph, dh, ek, st)
    nt = Int(log2(dh))
    nb = 2
    r = reverse(findr(ex_ph, dh))
    n = Int(log2(dh))
    states1 = [i == 1 ? "ρdot" : (Int(r[i-1]) == 0 ? "0" : "1") for i = 1:n+1]

    states = []
    qposr = Int(1 + nt)
    b = 2
    for i in 1:Int(length(st)[1])
        if i == qposr
            append!(states, [string(states1[1])])
        elseif i <= nt
            append!(states, [string(states1[b])])
            b += 1
        else
            append!(states, [string("Emp")])
        end
    end
    ρ_fin = MPO(st, [states])
    return ρ_fin
end
##########################################################################
function Abosedag(op, sysp, N1, λ, bpose_list)
    instlist = findinstance(N1)
    nty = 1 + N1
    os1 = OpSum()
    for i in 0:size(instlist)[1]-1 #N1 
        osb = OpSum()
        osb += λ * sqrt(i + 1), "$op", sysp, "Adag", bpose_list[1]
        r = instlist[i+1]
        for j in 1:N1
            y = r[j]
            if y == 1
                osb *= "Adag * A", bpose_list[j]
            else
                osb *= "A * Adag", bpose_list[j]
            end
        end
        os1 += osb
    end
    for i in 0:size(instlist)[1]-1 # This loop runs for all excitation levels of the phonon mode
        r = instlist[i+1] # For a given excitation, calls the r values.
        for k in 2:nty-1 # This loop runs for creating the B operator 
            osb = OpSum()
            osb += λ * sqrt(i + 1), "$op", sysp, "Adag", bpose_list[k]

            for l = 1:k-1
                osb *= "A", bpose_list[l]
            end
            for j in 1:N1
                y = r[j]
                if y == 1
                    osb *= "Adag * A", bpose_list[j]
                else
                    osb *= "A * Adag", bpose_list[j]
                end
            end
            os1 += osb
        end
    end
    return os1
end
##########################################################################
function Abose(op, sysp, N1, λ, bpose_list)
    instlist = findinstance(N1)
    nty = 1 + N1
    os1 = OpSum()
    for i in 0:size(instlist)[1]-1 #N1 
        osb = OpSum()
        osb += sqrt(i + 1), "Id", sysp
        r = instlist[i+1]
        for j in 1:N1
            y = r[j]
            if y == 1
                osb *= "Adag * A", bpose_list[j]
            else
                osb *= "A * Adag", bpose_list[j]
            end
        end
        osb *= λ, "$op", sysp, "A", bpose_list[1]
        os1 += osb
    end
    for i in 0:size(instlist)[1]-1 # This loop runs for all excitation levels of the phonon mode
        r = instlist[i+1] # For a given excitation, calls the r values.
        for k in 2:nty-1 # This loop runs for creating the B operator 
            osb = OpSum()
            osb += sqrt(i + 1), "Id", bpose_list[k]
            for j in 1:N1
                y = r[j]
                if y == 1
                    osb *= "Adag * A", bpose_list[j]
                else
                    osb *= "A * Adag", bpose_list[j]
                end
            end
            osb *= λ, "$op", sysp, "A", bpose_list[k]
            for l = 1:k-1
                osb *= "Adag", bpose_list[l]
            end
            os1 += osb
        end
    end
    return os1
end
##########################################################################
# Hs_chain_gen is the system Hamiltonian
function Hs_chain_gen_fm_in(sst, ϵ, ω, λ, edh) #ek1 is the bath chain  length             #__@__
    N1 = Int(log2(edh)) #number of pseudosites
    #ek = nb * ek1 + 1 # after this the boson part starts (real) (kintu ebar ulto)
    sysp = N1 + 1 # this is the q-dot position (real)
    bpose_list = collect(N1:-1:1)
    #------------------------------------------------------------------------------------------
    os1 = OpSum()
    # The fermion-phonon system and its tilde space H_s    
    for n in 1:1+N1 # running for core system length     
        if n == 1  # system 
            os1 += ϵ, "N", sysp
        else
            os1 += (2)^(n - 2) * ω, "Adag", bpose_list[n-1], "A", bpose_list[n-1]
        end
    end
    #------------------------------------------------------------------------------------------
    # os1+= λ,"N",1,"Adag",2    
    os1 += Abosedag("N", sysp, N1, λ, bpose_list)
    #os+= im*g,"N",1,"A",2
    os1 += Abose("N", sysp, N1, λ, bpose_list)
    #------------------------------------------------------------------------------------------
    LCC = MPO(os1, sst)
    return LCC
end
##########################################################################
function unit_gates_fm_in(recurr_list, ek1, dh, tau, nb, s_total) #δ_list, ω_bath_list,  #__@__
    nt = Int(log2(dh))
    tot_chain = Int(length(s_total)[1])
    S_posr = 1 + nt  # this is the q-dot position

    #------------------------------------------------------------------------------------------
    ω_n_LR = recurr_list[1] #=recur_coeff=#
    t_n_LR = recurr_list[2] #=recur_coeff=#
    c_0_LR = recurr_list[3] 

    #------------------------------------------------------------------------------------------
    
    fpst_list_r = [Int(2 + nt) Int(3 + nt)]
    fpst_list_t = [Int(4 + nt) Int(5 + nt)]
    gates = ITensor[]

    #global fl = 1     #__@__
    for p in fpst_list_r
        hj = c_0_LR[p] * op("Cdag", s_total[S_posr]) * op("C", s_total[p]) +
             c_0_LR[p] * op("C", s_total[S_posr]) * op("Cdag", s_total[p])
        Gj = exp(-im * (tau / 2) * hj)
        push!(gates, Gj)
    end
    for p in fpst_list_t
        hj = c_0_LR[p] * op("Cdag", s_total[S_posr]) * op("Cdag", s_total[p]) +
             c_0_LR[p] * op("C", s_total[S_posr]) * op("C", s_total[p])
        Gj = exp(-im * (tau / 2) * hj)
        push!(gates, Gj)
    end
    j = S_posr + 1
    while j <= tot_chain - 2 * nb    ##__@__
        s1 = s_total[j]
        s2 = s_total[j + 2 * nb]
        ω_n = ω_n_LR[j]
        t_n = t_n_LR[j]
        hj = ω_n * op("N", s1) * op("Id", s2) +
            t_n * op("Cdag", s1) * op("C", s2) +
            t_n * op("C", s1) * op("Cdag", s2)
        if iseven(div(j-S_posr-1, 2)) 
            Gj = exp(-im * (tau / 2) * hj)
            push!(gates, Gj)
        else
            Gj = exp(im * (tau / 2) * hj)
            push!(gates, Gj)
        end
        j += 1
    end

    for i in 1:nb
        hj = (ω_n_LR[end-nb+i]) * op("N", s_total[tot_chain-2*nb+i])
        Gj = exp(-im * (tau / 2) * hj)
        push!(gates, Gj)

        hj = (-ω_n_LR[end-nb+i]) * op("N", s_total[tot_chain-nb+i])
        Gj = exp(-im * (tau / 2) * hj)
        push!(gates, Gj)
    end

    return append!(gates, reverse(gates))
end
##########################################################################
function call_recurr_in(ek1, Γ_list, β_list, μ_list, δ_list, ω_bath_list, N_coeff, dh, nb) #
    Nquad = 10^7                                          ## Number of quadrature points
    supp = (min(μ_list...) - 10 * max(Γ_list...), max(μ_list...) + 10 * max(Γ_list...))           #__@__
    ω_n_total = []
    t_n_total = []
    c_0_list = []
    for j in 1:2 ##T-R
        if j == 1
            for i in nb:-1:1
                ω_bath = ω_bath_list[i]
                μ = μ_list[i]
                δ = δ_list[i]
                β = β_list[i]
                Γ = Γ_list[i] / 2
                #------------------------------------------------------------------------------------------
                n(ω) = 1 / (exp(β * (ω - μ)) + 1)
                #------------------------------------------------------------------------------------------
                #w_fn2(k) = (Γ * δ^2) / ((k - ω_bath)^2 + δ^2) * (n(k))                   ## weight function for tilde space bath
                w_fn2(k) = (Γ / (2 * pi)) * (n(k))
                #------------------------------------------------------------------------------------------
                ab2 = Matrix{Float64}(undef, ek1 + 1, 2)
                if ek1 >= N_coeff
                    ab2[1:N_coeff, 1:2] = recur_coeff(w_fn2, supp, N_coeff, Nquad)    ## recurrence coefficients for for tilde space bath
                    a2_100 = ab2[N_coeff, 1]
                    b2_100 = ab2[N_coeff, 2]
                    [ab2[i, 1] = a2_100 for i = N_coeff+1:ek1+1]
                    [ab2[i, 2] = b2_100 for i = N_coeff+1:ek1+1]
                else
                    #= ab2[1:(ek1+1), 1:2] = recur_coeff(w_fn2, supp, ek1 + 1, Nquad) =#
                    ab2[1:(ek1+1), 1:2] = recur_coeff(w_fn2, supp, N_coeff, Nquad)[1:(ek1+1), 1:2]
                end
                ω_n_TILD = ab2[1:ek1, 1]
                ω_n_total = append!(ω_n_total, ω_n_TILD)
                t_n_TILD = sqrt.(ab2[2:ek1+1, 2])
                t_n_total = append!(t_n_total, t_n_TILD)
                #------------------------------------------------------------------------------------------                
                η02 = quadgk(w_fn2, supp[1], supp[2])
                c_02 = sqrt(η02[1])
                c_0_list = append!(c_0_list, c_02)
            end
        else
            for i in 1:nb
                ω_bath = ω_bath_list[i]
                μ = μ_list[i]
                δ = δ_list[i]
                β = β_list[i]
                Γ = Γ_list[i] / 2
                #------------------------------------------------------------------------------------------
                n(ω) = 1 / (exp(β * (ω - μ)) + 1)
                #------------------------------------------------------------------------------------------
                #w_fn1(k) = (Γ * δ^2) / ((k - ω_bath)^2 + δ^2) * (1 + n(k))               ## weight function for real space bath
                w_fn1(k) = (Γ / (2 * pi)) * (1 - n(k))
                #------------------------------------------------------------------------------------------
                ab1 = Matrix{Float64}(undef, ek1 + 1, 2)
                if ek1 >= N_coeff
                    ab1[1:N_coeff, 1:2] = recur_coeff(w_fn1, supp, N_coeff, Nquad)    ## recurrence coefficients for for real space bath
                    a1_100 = ab1[N_coeff, 1]
                    b1_100 = ab1[N_coeff, 2]
                    [ab1[i, 1] = a1_100 for i = N_coeff+1:ek1+1]
                    [ab1[i, 2] = b1_100 for i = N_coeff+1:ek1+1]
                else
                    #= ab1[1:(ek1+1), 1:2] = recur_coeff(w_fn1, supp, ek1 + 1, Nquad)   =#
                    ab1[1:ek1+1, 1:2] = recur_coeff(w_fn1, supp, N_coeff, Nquad)[1:ek1+1, 1:2]
                end
                ω_n_REAL = ab1[1:ek1, 1]
                ω_n_total = append!(ω_n_total, ω_n_REAL)
                t_n_REAL = sqrt.(ab1[2:ek1+1, 2])
                t_n_total = append!(t_n_total, t_n_REAL)
                #------------------------------------------------------------------------------------------
                η01 = quadgk(w_fn1, supp[1], supp[2])
                c_01 = sqrt(η01[1])
                c_0_list = append!(c_0_list, c_01)

            end
        end
    end
    # Resuffle: zigzag
    function process_list(lst, ek)             #__@__
        # venge dilam 2 vag e
        half_length = div(length(lst), 2)
        first_half = lst[1:half_length]
        second_half = lst[half_length+1:end]
        # bath e vangbo 
        function break_and_interleave(sublist, ek)
            interleaved = []
            [[push!(interleaved, sublist[j]) for j in i:ek:length(sublist)] for i in 1:ek]
            return interleaved
        end
        processed_first_half = reverse(break_and_interleave(first_half, ek))      #__@__
        processed_second_half = break_and_interleave(second_half, ek)
        #final_result = vcat(processed_first_half, zeros(n1), processed_second_half)
        list = []
        for i = 1:2:half_length                                                  #__@__
            append!(list, processed_second_half[i:i+1])
            append!(list, processed_first_half[i:i+1])
        end
        return list 
    end
    n1 = 1 + Int(log2(dh))
    ω_n_total_fin = vcat(zeros(n1), process_list(ω_n_total, ek1))       #__@__
    t_n_total_fin = vcat(zeros(n1), process_list(t_n_total, ek1))
    c_0_list_fin = vcat(zeros(n1), process_list(c_0_list, 1))

    return [ω_n_total_fin, t_n_total_fin, c_0_list_fin]
end

##########################################################################


function write_for_loop(filename::String, i::String, output1::String, output2::String)
    # Get the current date and time
    current_time = now()
    # Open the file in append mode
    open(filename, "a") do file
        if i == string(1)
            # Add a separator for readability
            write(file, "----------------------\n")
            # Write the current date and time to the file
            write(file, ":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n")
            write(file, "                                     Date and Time: $(current_time)\n")
            write(file, ":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n")
            write(file, "$output1\n")
            # Write the output to the file
            write(file, "$(output2);\n")
        else
            write(file, "$(output2);\n")
        end
    end
end