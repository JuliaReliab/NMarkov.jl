"""
Example 5: Quasi-Stationary Analysis

This example computes the quasi-stationary vector for a CTMC with
absorbing states. The quasi-stationary vector is the conditional stationary
vector given that the process has not yet reached an absorbing state.
"""

using NMarkov
using NMarkov.SparseMatrix
using SparseArrays

# Define the infinitesimal generator with absorbing states
# Q = [T | ξ]
#     [0 | 0]
# where T is the generator over transient states and ξ is the
# transition rate from transient to absorbing states

T = [
    -4.0  1.0  0.0;
     0.0 -1.0  0.1;
     3.0  0.5 -3.5
]

# Transition rates to absorbing state
xi = -T * ones(3)
println("Transient state generator T:")
println(T)
println()

println("Transition rates to absorbing state ξ:")
println(xi)
println()

# Compute quasi-stationary vector using Gauss-Seidel with sparse matrix
spT = sparse(T)
qv1, gam1, conv1, iter1, rerror1 = qstgs(spT, xi)
println("Quasi-stationary vector (sparse):")
println(qv1)
println("(decay rate gamma: $gam1, converged: $conv1, iterations: $iter1)")
println()

# Using SparseCSC format
csc_T = SparseCSC(T)
qv2, gam2, conv2, iter2, rerror2 = qstgs(csc_T, xi)
println("Quasi-stationary vector (SparseCSC):")
println(qv2)
println("(decay rate gamma: $gam2, converged: $conv2, iterations: $iter2)")
println()

# Verify: υ * T = γ * υ where γ is the minimum eigenvalue of T
# and υ * 1 = 1
println("Verification:")
println("Sum of quasi-stationary vector = $(sum(qv1))")
