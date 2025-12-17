"""
Example 2: Transient Analysis of CTMC

This example computes the state probability vector at a given time t using
the matrix exponential: x_t = x_0 * exp(Q*t)
"""

using NMarkov
using NMarkov.SparseMatrix

# Define the infinitesimal generator Q
Q = [
    -1.0  1.0  0.0;
     0.0 -0.1  0.1;
     3.0  0.5 -3.5
]

# Initial probability vector (start from state 0)
x0 = Float64[1, 0, 0]
t = 2.0

# Compute state probability at time t: x_t = x_0 * exp(Q*t)
xt = mexp(Q, x0, t, transpose=:T)
println("State probability at t=$t (transpose=:T):")
println("x_t = $xt")
println()

# Compute state probability at time t: x_t = exp(Q*t) * x_0
xt_forward = mexp(Q, x0, t, transpose=:N)
println("State probability at t=$t (transpose=:N):")
println("x_t = $xt_forward")
println()

# Compute cumulative state probability integral
# xt, barxt = x_0 * int_0^t exp(Q*u) du
xt, barxt = mexpc(Q, x0, t, transpose=:T)
println("Cumulative state probability:")
println("x_t = $xt")
println("bar{x}_t = $barxt")
println()

# Compute mixture with exponential distribution
# y_t = x_0 * int_0^inf exp(Q*u) * f(u) du
# where f(u) = 1.0 * exp(-1.0*u)
yt = mexpmix(Q, x0, transpose=:T) do u
    1.0 * exp(-1.0 * u)
end
println("Mixture with exponential distribution:")
println("y_t = $yt")
println()

# Compute mixture with cumulative integral
# y_t, bary_t = x_0 * int_0^inf int_0^u exp(Q*s) ds * f(u) du
yt, baryt = mexpcmix(Q, x0, transpose=:T) do u
    1.0 * exp(-1.0 * u)
end
println("Mixture with cumulative integral:")
println("y_t = $yt")
println("bar{y}_t = $baryt")
