# Poisson

using Distributions
export poipmf!, poipmf, cpoipmf!, cpoipmf, rightbound

"""
@prob
@cprob

Macros for arranging the index
"""

macro prob(i)
    esc(:(prob[$i-left+1]))
end

macro cprob(i)
    esc(:(cprob[$i-left+1]))
end

"""
poipmf!(lambda, prob; left = 0, right = length(prob)-1+left)
poipmf(lambda, right; left = 0)

Compute the p.m.f. of Poisson distribution with mean lambda.
`left` and `right` are the domain of Poisson distribution.
They should be choosen so that the total probability in the domain becomes 1.
In `poipmf!``, the p.m.f. is saved to the vector `prob`.
The right can be obtained from `rightbound`.
The retuen value is the normalizing constant so that the total sum of `prob` is 1, called the weight.
`poipmf` returns a tuple of (weight, prob).
"""

function poipmf!(lambda::Tv, prob::Vector{Tv};
    left::Ti = 0, right::Ti = length(prob)-1+left) where {Tv, Ti}
    log2piOver2::Tv = log(2*pi) / 2
    mode::Ti = floor(Ti, lambda)
    if mode >= 1
        @prob(mode) = exp(-lambda + mode * log(lambda) - log2piOver2 - (mode + 1/2) * log(mode) + mode)
    else
        @prob(mode) = exp(-lambda)
    end
    # down
    for j = mode:-1:left+1
        @prob(j-1) = j / lambda * @prob(j)
    end
    # up
    for j = mode:right-1
        @prob(j+1) = lambda / (j+1) * @prob(j)
    end
    # compute W
    weight::Tv = 0
    s::Ti = left
    t::Ti = right
    while s < t
        if @prob(s) <= @prob(t)
            weight += @prob(s)
            s += 1
        else
            weight += @prob(t)
            t -= 1
        end
    end
    weight += @prob(s)
end

function poipmf(lambda::Tv, right::Ti; left::Ti = 0) where {Tv, Ti}
    prob = Vector{Tv}(undef, right-left+1)
    weight = poipmf!(lambda, prob, left=left, right=right)
    (weight, prob)
end

"""
cpoipmf!(lambda, prob, cprob; left = 0, right = length(prob)-1+left)
cpoipmf(lambda, right; left = 0)

Compute the p.m.f. and complementary c.d.f. of Poisson distribution with mean lambda.
`left` and `right` are the domain of Poisson distribution.
They should be choosen so that the total probability in the domain becomes 1.
In `cpoipmf!``, the p.m.f. and c.d.f. are stored to `prob` and `cprob`, respectively.
The right can be obtained from `rightbound`.
The retuen value is the normalizing constant so that the total sum of `prob` is 1, called the weight.
`cpoipmf` returns a tuple of (weight, prob, cprob).
"""

function cpoipmf!(lambda::Tv, prob::Vector{Tv}, cprob::Vector{Tv}; left::Ti = 0, right::Ti = length(prob)-1+left) where {Tv, Ti}
    weight::Tv = poipmf!(lambda, prob, left=left, right=right)
    @cprob(right) = 0
    for k = right-1:-1:left
        @cprob(k) = @cprob(k+1) + @prob(k+1)
    end
    weight
end

function cpoipmf(lambda::Tv, right::Ti; left::Ti = 0) where {Tv, Ti}
    prob = Vector{Tv}(undef, right-left+1)
    cprob = Vector{Tv}(undef, right-left+1)
    weight = cpoipmf!(lambda, prob, cprob, left=left, right=right)
    (weight, prob, cprob)
end

"""
rightbound(::Type{Ti} = Int, lambda::Tv, q::Tv = Tv(1.0e-8))

Compute the rightbound of Poisson distribution with mean lambda.
The rightbound finds a quantile so that the complementary c.d.f. becomes `q`.
"""

function rightbound(lambda::Tv, q::Tv = Tv(1.0e-8))::Int where {Tv}
    rightbound(Int, lambda, q)
end

function rightbound(::Type{Ti}, lambda::Tv, q::Tv = Tv(1.0e-8))::Ti where {Tv, Ti}
    z = Distributions.cquantile(Normal(), q)
    if lambda < 3.0
        ll = exp(-lambda)
        total = ll
        right::Ti = 0
        while true
            right += 1
            ll *= lambda / right
            total += ll
            if total + q >= 1.0
                break
            end
        end
        right
    else
        right = floor(Ti, (z + sqrt(4.0 * lambda - 1.0))^2 / 4.0 + 1.0)
    end
end
