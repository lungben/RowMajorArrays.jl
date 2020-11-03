using RowMajorArrays
using Test

@testset "RowMajorArrays.jl" begin
    a = [1 2 3; 4 5 6]
    a_r = RowMajorArray(a)

    @test a[1, 3] == a_r[3, 1]
    @test size(a) == reverse(size(a_r)) == (2, 3)
    @test size(a_r, 2) == 2

    @test RowMajorArray{Float64}(a) == RowMajorArray(convert.(Float64, a))

    setindex!(a_r, 42, 3, 2)
    @test getindex(a_r, 3, 2) == 42
    setindex!(a_r, 43, 2,1)
    @test getindex(a_r, 2, 1) == 43

    @test a_r[4] == a[4] # linear indices are the same for standard and RowMajorArrays

    z = RowMajorArrays.zeros(Int, 5, 3)
    @test size(z) == (5, 3)
    @test zeros(Int, 3, 5) == z.data
    @test eltype(z) == Int
    @test all(z .== 0)

    z2 = RowMajorArrays.zeros(5, 3)
    @test size(z2) == (5, 3)
    @test zeros(3, 5) == z.data
    @test eltype(z2) == Float64
    @test all(z2 .== 0)

    o = RowMajorArrays.ones(Int, 5, 3)
    @test size(o) == (5, 3)
    @test ones(Int, 3, 5) == o.data
    @test all(o .== 1)

    r1 = RowMajorArrays.rand(7, 2)
    @test size(r1.data) == (2, 7)
    @test eltype(r1) == Float64
    r1b = RowMajorArrays.rand(Float32, 7, 2)
    @test size(r1b.data) == (2, 7)
    @test eltype(r1b) == Float32

    r2 = RowMajorArrays.randn(7, 2)
    @test size(r2.data) == (2, 7)
    @test eltype(r2) == Float64
    r2b = RowMajorArrays.randn(Float32, 7, 2)
    @test size(r2b.data) == (2, 7)
    @test eltype(r2b) == Float32

    a42 = RowMajorArrays.fill(42, 5, 3)
    @test size(a42) == (5, 3)
    @test a42[2, 1] == 42
    
    a42b = RowMajorArrays.fill(42, (5, 3))
    @test size(a42b) == (5, 3)
    @test a42 == a42b

    undef_array = RowMajorArray{Float64, 2, Array{Float64, 2}}(undef, 3, 2)
    @test size(undef_array) == (3, 2)
    undef_array2 = RowMajorArray{Float64, 3, Array{Float64, 3}}(undef, 5, 4, 3)
    @test size(undef_array2) == (5, 4, 3)
    @test size(undef_array2.data) == (3, 4, 5)

    a_r = RowMajorArray([1 2 3; 4 5 6])
    one_r = RowMajorArrays.ones(3, 2)
    @test a_r + one_r == RowMajorArray([2. 3. 4.; 5. 6. 7.])
    @test a_r - one_r == RowMajorArray([0. 1. 2.; 3. 4. 5.])

    b_r = RowMajorArray([1 2; 3 4; 4 6])
    @test (a_r * b_r).data == transpose(a_r.data) * transpose(b_r.data)

    # inconsistent array types
    a = [1 2 3; 4 5 6]
    @test RowMajorArray(a) == RowMajorArray{Int64, 2, Array{Int64, 2}}(a) # consistent types
    @test_throws TypeError RowMajorArray(a) == RowMajorArray{Int64, 3, Array{Int64, 2}}(a)
    @test_throws TypeError RowMajorArray(a) == RowMajorArray{Int64, 2, Array{Float64, 2}}(a)

    # similar
    similar_1 = similar(a_r)
    @test size(similar_1) == (3,2)
    @test typeof(similar_1) == typeof(a_r) == RowMajorArray{Int64,2,Array{Int64,2}}
    similar_2 = similar(a_r, (10,4))
    @test size(similar_2) == (10,4)
    @test typeof(similar_2) == typeof(a_r) == RowMajorArray{Int64,2,Array{Int64,2}}
    similar_3 = similar(a_r, Int64, (10,4)) # same eltype as original array
    @test size(similar_3) == (10,4)
    @test typeof(similar_3) == typeof(a_r) == RowMajorArray{Int64,2,Array{Int64,2}}
    similar_4 = similar(a_r, Float64, (10,4)) # other eltype than original array
    @test size(similar_4) == (10,4)
    @test typeof(similar_4) == RowMajorArray{Float64,2,Array{Float64,2}}

    # broadcasting
    @test typeof(one_r .* a_r) == RowMajorArray{Float64,2,Array{Float64,2}}
    @test typeof(one_r .+ a_r) == RowMajorArray{Float64,2,Array{Float64,2}}
    @test typeof(a_r .* one_r) == RowMajorArray{Float64,2,Array{Float64,2}}
    @test typeof(a_r .+ one_r) == RowMajorArray{Float64,2,Array{Float64,2}}

    # check that standard array broadcasting is not broken
    @test typeof(a .+ ones(2, 3)) == Array{Float64,2}
    @test typeof(ones(2, 3) .+ a) == Array{Float64,2}

    # test pretty printing - 2d RowMajorArrays are printed with reversed dimensions of the underlying array
    io = IOBuffer(append=true)
    print(io, a_r)
    @test read(io, String) == "[1 4; 2 5; 3 6]"

    
end
