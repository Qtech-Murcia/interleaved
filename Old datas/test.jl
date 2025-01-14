#using LinearAlgebra, Plots

# Generate the density matrix for the Schrödinger cat state
function cat_state_density(alpha, M)
    # Fock space basis size M
    coherent_plus = zeros(ComplexF64, M)
    coherent_minus = zeros(ComplexF64, M)

    # Populate the coherent states |α⟩ and |-α⟩
    for n in 1:M
        coherent_plus[n] = exp(-0.5 * abs(alpha)^2) * (alpha^(n - 1)) / sqrt(factorial(big(n - 1)))
        coherent_minus[n] = exp(-0.5 * abs(-alpha)^2) * ((-alpha)^(n - 1)) / sqrt(factorial(big(n - 1)))
    end

    # Schrödinger cat state: |ψ⟩ = (|α⟩ + |-α⟩) / √2
    psi = (coherent_plus + coherent_minus) / sqrt(2)

    # Density matrix: ρ = |ψ⟩⟨ψ|
    rho = psi * psi'
    return rho
end


# Parameters
alpha = 2.0        # Displacement for coherent states
M = 30             # Number of Fock states
xvec = range(-5, 5, length=200)  # x-coordinates
yvec = range(-5, 5, length=200)  # y-coordinates

# Generate the density matrix
rho = cat_state_density(alpha, M)

# Compute the Wigner function using the code we wrote
W = _wigner_clenshaw(rho, xvec, yvec, sqrt(2), false)

# Visualize the Wigner function
heatmap(
    xlim=(minimum(xvec), maximum(xvec)), ylim=(minimum(yvec), maximum(yvec)),
    xvec, yvec, W,
    color=custom_colormap_with_white,  # Custom colormap with white at zero
    xlabel=L"x", ylabel=L"p",
    title=L"W(x,p)",
    colorbar_title="Value",           # Label for the color bar
    clim=(-0.5, 0.5),                # Adjust the range to highlight negative regions
    aspect_ratio=1.               # Set the aspect ratio to 1:1
)
