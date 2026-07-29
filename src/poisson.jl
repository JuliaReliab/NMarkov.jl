# Poisson

"""
    _checkpoirange(name, lambda, prob, left, right)

Validate the preconditions of `poipmf!`/`cpoipmf!` and return the mode of the
distribution.

The p.m.f. is built by seeding the recurrence at `prob[mode]` and walking it
outwards. Under `@origin (prob => left)` that write lands at the physical index
`mode - left + 1`, so a `mode` outside `[left, right]` writes past either end of
the buffer -- and the bodies run under `@inbounds`, which turns that into a
segfault or, worse, a silent heap corruption that still returns plausible
numbers. Check once here, then the loops can stay unchecked.
"""
function _checkpoirange(name::String, lambda::Tv, prob::Vector{Tv}, left::Ti, right::Ti) where {Tv,Ti}
    lambda >= 0 || throw(ArgumentError(
        "$name: lambda must be non-negative, got $lambda"))
    left <= right || throw(ArgumentError(
        "$name: the domain must satisfy left <= right, got ($left, $right)"))
    length(prob) >= right - left + 1 || throw(ArgumentError(
        "$name: the output vector holds $(length(prob)) elements, but the " *
        "domain [$left, $right] needs $(right - left + 1)"))
    mode = floor(Ti, lambda)
    left <= mode <= right || throw(ArgumentError(
        "$name: the domain [$left, $right] must contain the mode $mode of the " *
        "Poisson distribution with mean $lambda; size the domain with " *
        "`rightbound(lambda)` and keep left <= $mode"))
    mode
end

"""
    poipmf!(lambda, prob; left = 0, right = length(prob)-1+left)

Compute the probability mass function (p.m.f.) of Poisson distribution in-place.

The p.m.f. values are stored directly into the provided vector `prob` for efficiency.

### Arguments
- `lambda::Number`: Mean parameter of Poisson distribution
- `prob::Vector`: Output vector where p.m.f. values are stored
- `left::Integer`: Left boundary of the domain (default: 0)
- `right::Integer`: Right boundary of the domain

### Returns
- `weight`: Normalizing constant ensuring the sum of probabilities equals 1

### Notes
- This in-place version is more memory-efficient than `poipmf`
- The domain [left, right] should be chosen to capture significant probability mass
- Use `rightbound()` to automatically compute an appropriate right boundary

### Example
```julia
lambda = 5.0
prob = Vector{Float64}(undef, rightbound(lambda) + 1)
weight = poipmf!(lambda, prob, left=0, right=rightbound(lambda))
```
"""

@origin (prob => left) function poipmf!(lambda::Tv, prob::Vector{Tv};
    left::Ti = 0, right::Ti = length(prob)-1+left) where {Tv, Ti}
    mode::Ti = _checkpoirange("poipmf!", lambda, prob, left, right)
    @inbounds begin
        log2piOver2::Tv = log(2*pi) / 2
        if mode >= 1
            prob[mode] = exp(-lambda + mode * log(lambda) 
                - log2piOver2 - (mode + 1/2) * log(mode) + mode)
        else
            prob[mode] = exp(-lambda)
        end
        # down
        for j = mode:-1:left+1
            prob[j-1] = j / lambda * prob[j]
        end
        # up
        for j = mode:right-1
            prob[j+1] = lambda / (j+1) * prob[j]
        end
        # compute W
        weight::Tv = 0
        s::Ti = left
        t::Ti = right
        while s < t
            if prob[s] <= prob[t]
                weight += prob[s]
                s += 1
            else
                weight += prob[t]
                t -= 1
            end
        end
        weight += prob[s]
    end
end

"""
    poipmf(lambda, right; left = 0)

Compute the probability mass function (p.m.f.) of Poisson distribution and return new vectors.

This function allocates new vectors and returns both the p.m.f. values and normalizing weight.

### Arguments
- `lambda::Number`: Mean parameter of Poisson distribution
- `right::Integer`: Right boundary of the domain
- `left::Integer`: Left boundary of the domain (default: 0)

### Returns
- `weight`: Normalizing constant ensuring the sum of probabilities equals 1
- `prob`: Vector of p.m.f. values in the domain [left, right]

### Notes
- This allocating version is convenient but less memory-efficient than `poipmf!`
- Use `poipmf!` for in-place computation when performance is critical
- Use `rightbound()` to automatically compute an appropriate right boundary

### Example
```julia
lambda = 5.0
weight, prob = poipmf(lambda, rightbound(lambda), left=0)
```
"""
function poipmf(lambda::Tv, right::Ti; left::Ti = 0) where {Tv, Ti}
    prob = Vector{Tv}(undef, right-left+1)
    weight = poipmf!(lambda, prob, left=left, right=right)
    (weight, prob)
end

"""
    cpoipmf!(lambda, prob, cprob; left = 0, right = length(prob)-1+left)

Compute the p.m.f. and complementary c.d.f. of Poisson distribution in-place.

Both p.m.f. and complementary c.d.f. values are stored directly into provided vectors for efficiency.

### Arguments
- `lambda::Number`: Mean parameter of Poisson distribution
- `prob::Vector`: Output vector for p.m.f. values
- `cprob::Vector`: Output vector for complementary c.d.f. values
- `left::Integer`: Left boundary of the domain (default: 0)
- `right::Integer`: Right boundary of the domain

### Returns
- `weight`: Normalizing constant ensuring probabilities sum to 1

### Notes
- This in-place version is more memory-efficient than `cpoipmf`
- The complementary c.d.f. `cprob[k] = P(X > k)` is computed from the p.m.f.
- Use `rightbound()` to automatically compute an appropriate right boundary

### Example
```julia
lambda = 5.0
right = rightbound(lambda)
prob = Vector{Float64}(undef, right + 1)
cprob = Vector{Float64}(undef, right + 1)
weight = cpoipmf!(lambda, prob, cprob, left=0, right=right)
```
"""
@origin (prob => left, cprob => left) function cpoipmf!(lambda::Tv, prob::Vector{Tv}, cprob::Vector{Tv}; left::Ti = 0, right::Ti = length(prob)-1+left) where {Tv, Ti}
    # poipmf! validates lambda and prob; cprob is written here, so its length is
    # this function's own precondition.
    _checkpoirange("cpoipmf!", lambda, cprob, left, right)
    weight::Tv = poipmf!(lambda, prob, left=left, right=right)
    @inbounds begin
        cprob[right] = 0
        for k = right-1:-1:left
            cprob[k] = cprob[k+1] + prob[k+1]
        end
        weight
    end
end

"""
    cpoipmf(lambda, right; left = 0)

Compute the p.m.f. and complementary c.d.f. of Poisson distribution and return new vectors.

This function allocates new vectors and returns both distributions and normalizing weight.

### Arguments
- `lambda::Number`: Mean parameter of Poisson distribution
- `right::Integer`: Right boundary of the domain
- `left::Integer`: Left boundary of the domain (default: 0)

### Returns
- `weight`: Normalizing constant ensuring probabilities sum to 1
- `prob`: Vector of p.m.f. values in domain [left, right]
- `cprob`: Vector of complementary c.d.f. values (P(X > k)) for each k

### Notes
- This allocating version is convenient but less memory-efficient than `cpoipmf!`
- Use `cpoipmf!` for in-place computation when performance is critical
- The complementary c.d.f. `cprob[k] = P(X > k)` is computed efficiently from the p.m.f.

### Example
```julia
lambda = 5.0
weight, prob, cprob = cpoipmf(lambda, rightbound(lambda), left=0)
```
"""
function cpoipmf(lambda::Tv, right::Ti; left::Ti = 0) where {Tv, Ti}
    prob = Vector{Tv}(undef, right-left+1)
    cprob = Vector{Tv}(undef, right-left+1)
    weight = cpoipmf!(lambda, prob, cprob, left=left, right=right)
    (weight, prob, cprob)
end

"""
    rightbound(lambda, q = 1.0e-8)
    rightbound(Ti, lambda, q = 1.0e-8)

Compute the right boundary of the domain for Poisson distribution with mean lambda.

This function finds the smallest integer k such that P(X > k) ≤ q, where X follows a Poisson distribution with mean lambda.
This is useful for determining the appropriate domain for Poisson p.m.f. computations.

### Arguments
- `Ti::Type`: Integer type for return value (default: Int)
- `lambda::Float`: Mean parameter of Poisson distribution
- `q::Float`: Tail probability threshold (default: 1.0e-8)

### Returns
- Minimum value k such that the complementary c.d.f. P(X > k) ≤ q

### Notes
- For small lambda (< 3.0), uses direct summation for accuracy
- For larger lambda, uses normal approximation for efficiency
- Smaller q values give larger boundaries, capturing more of the distribution

### Example
```julia
lambda = 5.0
right = rightbound(lambda, 1.0e-8)  # Returns ≈ 16
weight, prob = poipmf(lambda, right)
```
"""

function rightbound(lambda::Tv, q::Tv = Tv(1.0e-8))::Int where {Tv}
    rightbound(Int, lambda, q)
end

function rightbound(::Type{Ti}, lambda::Tv, q::Tv = Tv(1.0e-8))::Ti where {Tv, Ti}
    lambda >= 0 || throw(ArgumentError("lambda must be non-negative, got $lambda"))
    if lambda < 3
        ll = exp(-lambda)
        total = ll
        right::Ti = 0
        while true
            right += 1
            ll *= lambda / right
            prev = total
            total += ll
            total + q >= 1 && break
            # In a narrow element type the running sum stops moving before it
            # reaches 1 - q, and the tail test above would never fire. Once an
            # added term changes nothing, every remaining term is below the
            # resolution of Tv, so the domain is as wide as it can usefully be.
            total == prev && break
        end
        right
    else
        z = cquantile(Normal(), q)
        right = floor(Ti, (z + sqrt(4 * lambda - 1))^2 / 4 + 1)
    end
end
