module RowMajorArrays

export RowMajorArray

using LinearAlgebra

struct RowMajorArray{A <: AbstractArray}
    data:: A
end

RowMajorArray{T}(::UndefInitializer, I::Vararg{<: Integer, N}) where {T <: AbstractArray, N} = RowMajorArray(T(undef, reverse(I)...))

import Base

Base.getindex(a:: RowMajorArray, I::Vararg{<: Integer, N}) where {N} = getindex(a.data, reverse(I)...)
Base.getindex(a:: RowMajorArray, i:: Integer) = getindex(a.data, i) # linear indexing is done row-major here to have optimal memory alignment

Base.setindex!(a:: RowMajorArray, v, i:: Integer) = setindex!(a.data, v, i)
Base.setindex!(a:: RowMajorArray, v, I::Vararg{<: Integer, N}) where {N} = setindex!(a.data, v, reverse(I)...)

Base.size(a:: RowMajorArray) = reverse(size(a.data))
Base.size(a:: RowMajorArray, dim:: Integer) = reverse(size(a.data))[dim]

Base.:(==)(a1:: RowMajorArray, a2:: RowMajorArray) = a1.data == a2.data

forward_methods = (:length, :eltype)
for m in forward_methods
    @eval Base.$m(a:: RowMajorArray) = Base.$m(a.data)
end

Base.iterate(a:: RowMajorArray) = Base.iterate(a.data)
Base.iterate(a:: RowMajorArray, i) = Base.iterate(a.data, i)

# constructor methods
# separate from `Base.zeros`, etc. - execute with qualified module name, e.g. `RowMajorArrays.zeros`, etc.
forward_methods = (:zeros, :ones, :rand, :randn)
for m in forward_methods
    @eval $m(T:: Type, I::Vararg{<: Integer, N}) where {N} = RowMajorArray(Base.$m(T, reverse(I)...))
    @eval $m(I::Vararg{<: Integer, N}) where {N} = RowMajorArray(Base.$m(reverse(I)...))
end

fill(x, I::Vararg{<: Integer, N}) where {N} = RowMajorArray(Base.fill(x, reverse(I)...))
fill(x, I::Tuple) = RowMajorArray(Base.fill(x, reverse(I)))

# operations
## operations where element dimension is not relevant
forward_methods = (:(+), :(-))
for m in forward_methods
    @eval Base.$m(a1:: RowMajorArray, a2:: RowMajorArray) = RowMajorArray(Base.$m(a1.data, a2.data))
end

# ideally, this should be replaced with calls to OpenBlas Row-Major functions
Base.:(*)(a1:: RowMajorArray, a2:: RowMajorArray) = RowMajorArray(collect(transpose(a1.data)) * collect(transpose(a2.data)))

end
