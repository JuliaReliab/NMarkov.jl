"""
Example 4: Sensitivity Analysis of Stationary Vector

This example computes the first derivative of the stationary vector
with respect to a transition parameter.
"""

using NMarkov
using NMarkov.SparseMatrix
using SparseArrays

# Define the infinitesimal generator Q
Q = [
    -1.0  1.0  0.0;
     0.0 -0.1  0.1;
     3.0  0.5 -3.5
]

# Compute stationary vector
piv = gth(Q)
println("Stationary vector π:")
println(piv)
println()

# Define the derivative of Q with respect to transition rate λ_01
# (the (1,2) element, which is currently 1.0)
dQ = Float64[
    -1  1  0;
     0  0  0;
     0  0  0
]

# Compute b = dQ^T * π
b = dQ' * piv
println("Derivative parameter b = dQ' * π:")
println(b)
println()

# Compute sensitivity vector using QR decomposition
dpi = stsen(Q, piv, b)
println("Sensitivity vector ∂π/∂λ_01 (using QR):")
println(dpi)
println()

# Alternatively, for sparse matrices, use Gauss-Seidel
spQ = sparse(Q)
dpi_sparse = stsengs(spQ, piv, b)
println("Sensitivity vector ∂π/∂λ_01 (using Gauss-Seidel on sparse):")
println(dpi_sparse)
println()

# Create SparseCSC format and compute sensitivity
csc = SparseCSC(Q)
dpi_csc = stsengs(csc, piv, b)
println("Sensitivity vector ∂π/∂λ_01 (using SparseCSC):")
println(dpi_csc)
