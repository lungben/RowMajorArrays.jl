using BenchmarkTools
using RowMajorArrays

function sum_idx1(a)
    s = zero(eltype(a))
    @inbounds for i = 1:size(a, 1), j = 1:size(a, 2)
        s += a[i, j]
    end
    s
end

function sum_idx2(a)
    s = zero(eltype(a))
    @inbounds for j = 1:size(a, 2), i = 1:size(a, 1)
        s += a[i, j]
    end
    s
end

a1 = rand(1000, 1000)
a1_r = RowMajorArray(a1)

println("Array row major sum")
@btime sum_idx1($a1)
println("Array column major sum")
@btime sum_idx2($a1)

println("RowMajorArray row major sum")
@btime sum_idx1($a1_r)
println("RowMajorArray column major sum")
@btime sum_idx2($a1_r)
