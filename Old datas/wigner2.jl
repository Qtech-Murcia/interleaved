using DrWatson
using Plots, LinearAlgebra, Dates, FFTW, ZChop, IteratorSampling, Polynomials, LaTeXStrings
using SpecialFunctions, SparseArrays
using Distributed




# Add worker processes for parallelization
addprocs()


# Function to evaluate the weighted Laguerre polynomial using Clenshaw recursion
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

@everywhere function parallel_wigner_clenshaw(rho, xvec, yvec, g=sqrt(2), sparse=false)


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

# Function to save the Wigner function to a binary file
function save_wigner_to_file(filename, W)
    open(filename, "w") do io
        serialize(io, W)
    end
    println("Wigner function saved to: ", filename)
end



# Example usage
rho = [0.5 0.3; 0.3 0.5]  # Example density matrix
xvec = range(-10, 10, length=100)
yvec = range(-10, 10, length=100)
dx = step(xvec)  # Grid spacing in x-direction
dy = step(yvec)  # Grid spacing in y-direction
# Compute the Wigner function
W1 = parallel_wigner_clenshaw(rho, xvec, yvec, sqrt(2), false)
@show sum(W1) * dx * dy


# Plot the Wigner function as a heatmap
# Define custom colors with white for zero
color_min = RGB(50 / 255, 200 / 255, 100 / 255)   # Bright green for negative values
color_mid = RGB(1, 1, 1)                    # White for zero
color_max = RGB(255 / 255, 100 / 255, 100 / 255)  # Bright pink for positive values

# Create a custom color gradient with white at the midpoint
custom_colormap_with_white = cgrad([color_min, color_mid, color_max], [0.0, 0.5, 1.0])

# Plot the Wigner function as a heatmap
p = heatmap(
    xlim=(minimum(xvec), maximum(xvec)), ylim=(minimum(yvec), maximum(yvec)),
    xvec, yvec, W1,
    color=custom_colormap_with_white,  # Custom colormap with white at zero
    xlabel=L"x", ylabel=L"p",
    title=L"W(x,p)",
    #colorbar_title="Value",           # Label for the color bar
    clim=(-0.2, 0.2),                # Adjust the range to highlight negative regions
    aspect_ratio=1.0               # Set the aspect ratio to 1:1
)
display("image/png", p)
# Save the Wigner function to a file
#save_wigner_to_file("wigner_function.dat", W)
