using RowMajorArrays
using Test

@testset "RowMajorArrays.jl" begin
    a = [1 2 3; 4 5 6]
    a_r = RowMajorArray(a)

    @test a[1, 3] == a_r[3, 1]
    @test size(a) == reverse(size(a_r)) == (2, 3)

    setindex!(a_r, 42, 3, 2)
    @test getindex(a_r, 3, 2) == 42
    setindex!(a_r, 43, 2,1)
    @test getindex(a_r, 2, 1) == 43

    z = RowMajorArrays.zeros(Int, 5, 3)
    @test size(z) == (5, 3)
    @test zeros(Int, 3, 5) == z.data
    @test eltype(z) == Int
    @test all(z .== 0)

    z2 = RowMajorArrays.zeros(5, 3)
    @test size(z2) == (5, 3)
    @test zeros(3, 5) == z.data
    @test eltype(z2) == Float64
    # @test z .== 0

    o = RowMajorArrays.ones(Int, 5, 3)
    @test size(o) == (5, 3)
    @test ones(Int, 3, 5) == o.data
    # @test o .== 0

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

    undef_array = RowMajorArray{Array{Float64, 2}}(undef, 3, 2)
    @test size(undef_array) == (3, 2)
    undef_array2 = RowMajorArray{Array{Float64, 3}}(undef, 5, 4, 3)
    @test size(undef_array2) == (5, 4, 3)
    @test size(undef_array2.data) == (3, 4, 5)

    a_r = RowMajorArray([1 2 3; 4 5 6])
    one_r = RowMajorArrays.ones(3, 2)
    @test RowMajorArray([1 2 3; 4 5 6]) + one_r == RowMajorArray([2. 3. 4.; 5. 6. 7.])
    @test RowMajorArray([1 2 3; 4 5 6]) - one_r == RowMajorArray([0. 1. 2.; 3. 4. 5.])

    b_r = RowMajorArray([1 2; 3 4; 4 6])
    @test (a_r * b_r).data == transpose(a_r.data) * transpose(b_r.data)

end
