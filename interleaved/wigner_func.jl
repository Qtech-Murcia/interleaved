
"""
    extract_diag(mat, l)

Extracts the l-th diagonal of a matrix.

# Arguments
- `mat`: Matrix from which to extract the diagonal.
- `l`: Offset of the diagonal to extract (0 for the main diagonal, >0 for superdiagonals, <0 for subdiagonals).

# Returns
- A vector containing the elements of the specified diagonal.
"""
function extract_diag(mat, l)
    if l >= 0
        idx = 1:min(size(mat, 1), size(mat, 2) - l)
        return [mat[i, i + l] for i in idx]
    else
        l = abs(l)
        idx = 1:min(size(mat, 1) - l, size(mat, 2))
        return [mat[i + l, i] for i in idx]
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
function _wig_laguerre_val(L, x, c)
    #c_L = _wig_laguerre_val(L, x, diag(rho, L))
    """
    this is evaluation of polynomial series inspired by hermval from numpy.    
    Returns polynomial series
    sum_n b_n LL_n^L,
    where
    LL_n^L = (-1)^n sqrt(L!n!/(L+n)!) LaguerreL[n,L,x]    
    The evaluation uses Clenshaw recursion
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
function wigner_clenshaw(rho, xvec, yvec, g, sparse)
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

    if !sparse
        # Dense matrix mode
        local_rho = copy(rho)  # Make a local copy to avoid modifying input
        local_rho .= local_rho .* (2 .- I(M))  # Subtract identity matrix I(M)
        diag_cache = [extract_diag(local_rho, l) for l in 0:L]  # Precompute all diagonals
        for l in reverse(0:L-1)  # Iterate over degrees in reverse
            diag = diag_cache[l + 1]
            w0 .= _wig_laguerre_val(l, B, diag) .+ w0 .* A2 .* inv(sqrt(l + 1))
        end
    else
        # Sparse matrix mode
        data, indices, indptr = sparsematrix_parts(rho)  # Extract CSR components
        for l in reverse(0:L-1)  # Iterate over degrees in reverse
            diag = _csr_get_diag(data, indices, indptr, l, M, M)  # Extract L-th diagonal
            if l != 0
                diag .= diag .* 2  # Scale non-zero diagonals
            end
            w0 .= _wig_laguerre_val(l, B, diag) .+ w0 .* A2 .* inv(sqrt(l + 1))
        end
    end
    dx = step(xvec)  # Grid spacing in x-direction
    dy = step(yvec)  # Grid spacing in y-direction
    @show normalization = sum(real(w0) .* exp.(-B * 0.5)) .* (1/ π) * dx * dy
    W = real(w0) .* exp.(-B * 0.5) .* (1/ π) .*(1/normalization)


    # Compute the final Wigner function
    return W
end


function wigner_clenshaw(
    ρ::AbstractArray{T1},
    xvec::AbstractVector{T},
    yvec::AbstractVector{T},
    g::Real)

    
    g = convert(T, g)
    M = size(ρ, 1)
    X, Y = meshgrid(xvec, yvec)
    A = g * (X + 1im * Y)

    B = abs.(A)
    B .*= B
    W = similar(A)
    W .= 2 * ρ[1, end]
    L = M - 1

    y0 = similar(B, T1)
    y1 = similar(B, T1)
    y0_old = copy(y0)
    res = similar(y0)

    while L > 0
        L -= 1
        ρdiag = _wig_laguerre_clenshaw!(res, L, B, lmul!(1 + Int(L != 0), diag(ρ, L)), y0, y1, y0_old)
        @. W = ρdiag + W * A / √(L + 1)
    end

    return @. real(W) * exp(-B / 2) * g^2 / 2 / π
end
