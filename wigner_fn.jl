
# Add worker processes for parallelization
addprocs()

"""
    extract_diag(mat, l)

Extracts the l-th diagonal of a matrix.

# Arguments
- `mat`: Matrix from which to extract the diagonal.
- `l`: Offset of the diagonal to extract (0 for the main diagonal, >0 for superdiagonals, <0 for subdiagonals).

# Returns
- A vector containing the elements of the specified diagonal.
"""
@everywhere function extract_diag(mat, l)
    if l >= 0
        idx = 1:min(size(mat, 1), size(mat, 2) - l)
        return [mat[i, i+l] for i in idx]
    else
        l = abs(l)
        idx = 1:min(size(mat, 1) - l, size(mat, 2))
        return [mat[i+l, i] for i in idx]
    end
end


"""
    _wig_laguerre_val(L, x, c)

Computes the weighted Laguerre polynomial values using Clenshaw recursion.

# Arguments
- `L`: Integer degree of the Laguerre polynomial.
- `x`: 2D array of squared amplitudes `|A2|^2`.
- `c`: Vector of coefficients.

# Returns
- A 2D array of Laguerre polynomial values weighted by the coefficients.
"""
@everywhere function _wig_laguerre_val(L, x, c)
    """
    Evaluation of polynomial series using Clenshaw recursion.
    Returns polynomial series sum_n b_n * LL_n^L, where:
    LL_n^L = (-1)^n * sqrt(L! * n! / (L + n)!) * LaguerreL[n, L, x].
    """
    n = length(c)

    # Special cases for short coefficient arrays
    if n == 1
        return c[1]  # Single coefficient case
    elseif n == 2
        y0, y1 = c[1], c[2]
    else
        y0, y1 = c[end-1], c[end]
        for k in reverse(3:n)
            factor1 = sqrt((k - 2) * (L + k - 2) / ((L + k - 1) * (k - 1)))
            factor2 = ((L + 2k - 3) .- x) ./ sqrt((L + k - 1) * (k - 1))  # Broadcasting subtraction
            y0, y1 = c[k-2] .- y1 .* factor1, y0 .- y1 .* factor2  # Broadcasting subtraction and multiplication
        end
    end

    # Final term
    return y0 .- y1 .* ((L + 1) .- x) ./ sqrt(L + 1)  # Broadcasting subtraction and division
end

"""
    wigner_clenshaw(rho, xvec, yvec, g = sqrt(2), sparse = false)

Calculates the Wigner function using Clenshaw summation for numerical stability and efficiency.

# Arguments
- `rho`: Density matrix, either dense (2D array) or sparse (CSR format).
- `xvec`: 1D array of x-coordinates for the grid.
- `yvec`: 1D array of y-coordinates for the grid.
- `g`: Scaling factor for the Wigner function (default `sqrt(2)`).
- `sparse`: Boolean indicating if the density matrix is sparse (default `false`).

# Returns
- A 2D array representing the Wigner function over the specified grid.
"""
@everywhere function parallel_wigner_clenshaw(rho, xvec, yvec, g=sqrt(2), sparse=false)
    """
    The Wigner function is calculated as
    :math:`W = e^(-0.5*x^2)/pi * sum_{L} c_L (2x)^L / sqrt(L!)` where 
    :math:`c_L = sum_n rho_{n,L+n} LL_n^L` where
    :math:`LL_n^L = (-1)^n sqrt(L!n!/(L+n)!) LaguerreL[n,L,x]`
    
    """

    M = size(rho, 1)  # Dimension of the density matrix

    # Compute the 2D grid of complex coordinates
    X = xvec'  # Transposed x-coordinates for broadcasting
    Y = yvec   # y-coordinates
    A2 = g * (X .+ 1im .* Y)  # Scaled complex amplitudes A2 = 2 * A

    # Precompute squared magnitude of A2
    B = abs2.(A2)

    # Initialize the Wigner function with the last diagonal's contribution
    w0 = (2 * real(rho[1, end])) * ones(Complex{Float64}, size(A2))
    L = M - 1  # Maximum degree of the summation



    # Function to process a single diagonal in parallel
    @everywhere function process_diagonal(diag, l, B, A2, w0)
        for j in eachindex(B)
            w0[j] += _wig_laguerre_val(l, B[j], diag) * A2[j] / sqrt(l + 1)
        end
    end

    # Launch parallel processing for each diagonal
    if !sparse
        local_rho = copy(rho)
        local_rho .= local_rho .* (2 .- I(M))
        diag_cache = [extract_diag(local_rho, l) for l in 0:L]

        @distributed for l in reverse(0:L-1)
            process_diagonal(diag_cache[l+1], l, B, A2, w0)
        end
    else
        data, indices, indptr = sparsematrix_parts(rho)
        @distributed for l in reverse(0:L-1)
            diag = _csr_get_diag(data, indices, indptr, l, M, M)
            process_diagonal(diag, l, B, A2, w0)
        end
    end

    # Compute grid spacing
    dx = step(xvec)
    dy = step(yvec)

    # The Wigner function is calculated using Clenshaw summation:
    # W = (e^(-0.5 * B) / π) * Σ_L [c_L * (2 * A2)^L / √(L!)]
    # where c_L = Σ_n [rho_{n, L+n} * LL_n^L]
    # LL_n^L = (-1)^n * √(L! * n! / (L+n)!) * LaguerreL[n, L, B]

    @show normalization = sum(real(w0) .* exp.(-B * 0.5)) .* (g^2 / π) * dx * dy
    W = real(w0) .* exp.(-B * 0.5) .* (g^2 / π) .* (1 / normalization)

    # Compute the final Wigner function
    return W
end

