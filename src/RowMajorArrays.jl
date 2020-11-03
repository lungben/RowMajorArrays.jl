module RowMajorArrays

export RowMajorArray

using LinearAlgebra

"""
Wrapper of a column major array (e.g. `Array` or `CuArray`) to make it a row-major array.

The default constructor takes a column major array as input, it is interpreted as row-major array.
This causes an implicit transpose of the input array, e.g. a `(2, 3)` column major array is converted to a 
`(3, 2)` row major array.

For more than 2 dimensions, the order of the dimensions is reversed, e.g. a `(2, 3, 4, 5)` element column major array
is converted to a `(5, 4, 3, 2)` RowMajorArray.

Example:

    a = [1 2 3; 4 5 6] # 2x3 (column major) array
    a_r = RowMajorArray(a) # 3x2 row-major array

"""
struct RowMajorArray{T, N, A <: AbstractArray{T, N}} <: AbstractArray{T, N}
    data:: A
end

"""
Constructor which allows modification of array eltype.

Example:

    a = [1 2 3; 4 5 6] # eltype: Int64
    a_r = RowMajorArray{Float64}(a) # eltype Float64

"""
function RowMajorArray{T}(a:: AbstractArray{X, N}) where {X, T, N}
    a_converted = convert.(T, a)
    RowMajorArray{T, N, typeof(a_converted)}(a_converted)
end

# initializing with undefined fields
RowMajorArray{T, N, A}(::UndefInitializer, I::Vararg{<: Int, N}) where {T, N, A <: AbstractArray} = RowMajorArray(A(undef, reverse(I)...))

import Base

# getindex and setindex! methods
Base.getindex(a:: RowMajorArray, I::Vararg{<: Int, N}) where {N} = getindex(a.data, reverse(I)...)
Base.getindex(a:: RowMajorArray, i:: Int) = getindex(a.data, i) # linear indexing is done row-major here to have optimal memory alignment

Base.setindex!(a:: RowMajorArray, v, i:: Int) = setindex!(a.data, v, i)
Base.setindex!(a:: RowMajorArray, v, I::Vararg{<: Int, N}) where {N} = setindex!(a.data, v, reverse(I)...)

# standard function definitions
Base.size(a:: RowMajorArray) = reverse(size(a.data))
Base.size(a:: RowMajorArray, dim:: Int) = reverse(size(a.data))[dim]

Base.:(==)(a1:: RowMajorArray, a2:: RowMajorArray) = a1.data == a2.data

# for display for 2d RowMajorArrays, transpose data for printing so that the dimensions are printed in the right order
Base.show_nd(io::IO, a::RowMajorArray{T, 2, A}, print_matrix::Function, label_slices::Bool) where {T, A} = Base.show_nd(io, collect(transpose(a)), print_matrix, label_slices)

forward_methods = (:length, :eltype)
for m in forward_methods
    @eval Base.$m(a:: RowMajorArray) = Base.$m(a.data)
end

Base.similar(a:: RowMajorArray{T, N, A}) where {T, N, A} = similar(a, T, size(a))
Base.similar(a:: RowMajorArray{T, N, A}, dims:: Dims) where {T, N, A} =  RowMajorArray{T, N, A}(A(undef, reverse(dims)...))

function Base.similar(a:: RowMajorArray{X, N, A}, ::Type{T}, dims:: Dims) where {X, T, N, A}
    data = similar(a.data, T, reverse(dims))
    RowMajorArray(data)
end

# defining broadcasting
Base.BroadcastStyle(::Type{<: RowMajorArray}) = Broadcast.ArrayStyle{RowMajorArray}()

# see https://docs.julialang.org/en/v1/manual/interfaces/#Selecting-an-appropriate-output-array
function Base.similar(bc:: Broadcast.Broadcasted{Broadcast.ArrayStyle{RowMajorArray}}, ::Type{ElType}) where ElType
    # Scan the inputs for the ArrayAndChar:
    A = find_row_major_array(bc)
    # Use the char field of A to create the output
    similar(A, ElType, size(A))
end

# utility function for similar
# `A = find_row_major_array(As)` returns the first RowMajorArray among the arguments.
find_row_major_array(bc::Base.Broadcast.Broadcasted) = find_row_major_array(bc.args)
find_row_major_array(args::Tuple) = find_row_major_array(find_row_major_array(args[1]), Base.tail(args))
find_row_major_array(a::RowMajorArray) = a
find_row_major_array(::Tuple{}) = nothing
find_row_major_array(a::RowMajorArray, rest) = a
find_row_major_array(::Any, rest) = find_row_major_array(rest)

# constructor methods
# separate from `Base.zeros`, etc. - execute with qualified module name, e.g. `RowMajorArrays.zeros`, etc.
forward_methods = (:zeros, :ones, :rand, :randn)
for m in forward_methods
    @eval $m(T:: Type, I::Vararg{<: Int, N}) where {N} = RowMajorArray(Base.$m(T, reverse(I)...))
    @eval $m(I::Vararg{<: Int, N}) where {N} = RowMajorArray(Base.$m(reverse(I)...))
end

fill(x, I::Vararg{<: Int, N}) where {N} = RowMajorArray(Base.fill(x, reverse(I)...))
fill(x, I::Tuple) = RowMajorArray(Base.fill(x, reverse(I)))

# operations
forward_methods = (:(+), :(-))
for m in forward_methods
    @eval Base.$m(a1:: RowMajorArray, a2:: RowMajorArray) = RowMajorArray(Base.$m(a1.data, a2.data))
end

# Matrix multiplication - very crude implementation by converting to standard arrays.
# This should not be too slow for large matrices because converting is O(n²) whereas matrix multiplication is O(n³).
# Tdeally, this should be replaced with calls to OpenBlas Row-Major functions
Base.:(*)(a1:: RowMajorArray, a2:: RowMajorArray) = RowMajorArray(collect(transpose(a1.data)) * collect(transpose(a2.data)))

end
