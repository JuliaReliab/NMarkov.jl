"""
Example 3: Stationary Analysis of CTMC

This example computes the stationary probability vector of a CTMC
using the GTH algorithm and Gauss-Seidel iteration.
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

# Compute stationary vector using GTH algorithm (for dense matrices)
piv1 = gth(Q)
println("Stationary vector using GTH algorithm:")
println("π = $piv1")
println()

# Create sparse matrix and compute stationary vector using Gauss-Seidel
spQ = sparse(Q)
piv2 = stgs(spQ)
println("Stationary vector using Gauss-Seidel (sparse):")
println("π = $piv2")
println()

# Using SparseCSC format
csc = SparseCSC(Q)
piv3 = stgs(csc)
println("Stationary vector using Gauss-Seidel (SparseCSC):")
println("π = $piv3")
println()

# Verify: π * Q should be close to 0
println("Verification: π * Q =")
println(piv1' * Q)
println()

# Sum of π should be 1
println("Sum of π = $(sum(piv1))")
