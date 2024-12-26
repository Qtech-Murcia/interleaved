
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
    @show total = sum(real(w0) .* exp.(-B * 0.5)) .* (1/ π) * dx * dy

    # Compute the final Wigner function
    return real(w0) .* exp.(-B * 0.5) .* (1/ π)
end


