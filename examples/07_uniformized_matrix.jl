"""
Example 7: Uniformized Transition Probability Matrix

This example shows how to create a uniformized transition probability matrix
from the infinitesimal generator Q and use it for analysis.
"""

using NMarkov

# Define the infinitesimal generator Q
Q = [
    -1.0  1.0  0.0;
     0.0 -0.1  0.1;
     3.0  0.5 -3.5
]

println("Infinitesimal generator Q:")
println(Q)
println()

# Create uniformized transition probability matrix P
# P = I - Q / q, where q is the maximum absolute value of diagonal elements
P, qv = unif(Q)

println("Uniformization rate q:")
println(qv)
println()

println("Uniformized transition probability matrix P:")
println(P)
println()

# Compute stationary vector of P using power method
piv = stpower(P)
println("Stationary vector using power method:")
println(piv)
println()

# Sensitivity analysis using uniformized matrix
# Define derivative of Q
dQ = Float64[
    -1  1  0;
     0  0  0;
     0  0  0
]

# Derivative of P
dP = dQ / qv
b = dP' * piv

println("Sensitivity parameter b = dP' * π:")
println(b)
println()

# Compute sensitivity using power method
dpi = stsenpower(P, piv, b)
println("Sensitivity vector ∂π/∂λ_01 (power method):")
println(dpi)
println()

# Quasi-stationary analysis with uniformized matrix
# For CTMC with absorbing states
T = [
    -4.0  1.0  0.0;
     0.0 -1.0  0.1;
     3.0  0.5 -3.5
]
xi = -T * ones(3)

# Uniformize T
U, qv_t = unif(T)
xi_dash = xi / qv_t

println("Quasi-stationary vector using power method:")
qv_quasi = qstpower(U, xi_dash)
println(qv_quasi)
