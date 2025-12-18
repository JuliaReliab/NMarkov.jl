"""
Example 1: Definition of CTMC

This example shows how to define a Continuous-Time Markov Chain (CTMC)
using the infinitesimal generator matrix Q.
"""

using NMarkov
using SparseArrays

# Define CTMC with three states using a dense matrix
Q = [
    -1.0  1.0  0.0;
     0.0 -0.1  0.1;
     3.0  0.5 -3.5
]

println("Dense CTMC infinitesimal generator Q:")
println(Q)
println()

# Define CTMC using sparse matrices
using NMarkov.SparseMatrix

spQ = spzeros(3, 3)
spQ[1, 2] = 1.0
spQ[2, 3] = 0.1
spQ[3, 1] = 3.0
spQ[3, 2] = 0.5

spQ[1, 1] = -1.0
spQ[2, 2] = -0.1
spQ[3, 3] = -3.5

# Convert to different sparse matrix formats
csr = SparseCSR(spQ)
csc = SparseCSC(spQ)
coo = SparseCOO(spQ)

println("Sparse matrix formats:")
println("CSR format: $(typeof(csr))")
println("CSC format: $(typeof(csc))")
println("COO format: $(typeof(coo))")
