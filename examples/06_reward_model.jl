"""
Example 6: Markov Reward Model

This example computes reward-based metrics for a CTMC using
reward vectors and time intervals.
"""

using NMarkov

# Define the infinitesimal generator Q
Q = [
    -1.0  1.0  0.0;
     0.0 -0.1  0.1;
     3.0  0.5 -3.5
]

# Initial probability vector
x = Float64[1, 0, 0]

# Reward vector (reward for being in each state)
r = Float64[1, 1, 0]

# Time intervals where we compute rewards
ts = LinRange(0.0, 10.0, 10)

println("Markov Reward Model Example")
println("============================")
println()

println("Initial state probability x:")
println(x)
println()

println("Reward vector r:")
println(r)
println()

println("Time points:")
println(ts)
println()

# Compute transient reward analysis
# irwd: instantaneous reward at each time point
# crwd: cumulative reward up to each time point
# y:    state probability at the LAST time point
# cy:   cumulative state probability over the whole interval
irwd, crwd, y, cy = tran(Q, x, r, ts)

println("Instantaneous rewards at each time point:")
println(irwd)
println()

println("Cumulative rewards at each time point:")
println(crwd)
println()

println("State probability at the last time point t=$(last(ts)):")
println(y)
println()

println("Cumulative state probability over [0, $(last(ts))]:")
println(cy)
println()

# The state probability at every time point comes from mexpc, which returns one
# vector per time point.
println("State probability vectors at each time point:")
probs, cprobs = mexpc(Q, x, ts, transpose=:T)
for (i, t) in enumerate(ts)
    println("t=$t: $(probs[i])")
end
