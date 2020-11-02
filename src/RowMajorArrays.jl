module RowMajorArrays

export RowMajorArray

struct RowMajorArray{A <: AbstractArray}
    data:: A
end

import Base

Base.getindex(a:: RowMajorArray, I::Vararg{<: Integer, N}) where {N} = getindex(a.data, reverse(I)...)
Base.getindex(a:: RowMajorArray, i:: Integer) = getindex(a.data, i) # linear indexing is done row-major here to have optimal memory alignment

Base.setindex!(a:: RowMajorArray, v, i:: Integer) = setindex!(a.data, v, i)
Base.setindex!(a:: RowMajorArray, v, I::Vararg{<: Integer, N}) where {N} = setindex!(a.data, v, reverse(I)...)

Base.size(a:: RowMajorArray) = reverse(size(a.data))

forward_methods = (:length, :eltype)
for m in forward_methods
    @eval Base.$m(a:: RowMajorArray) = Base.$m(a.data)
end

# separate from `Base.zeros`, etc. - execute with qualified module name, e.g. `RowMajorArrays.zeros`, etc.
forward_methods = (:zeros, :ones, :rand, :randn)
for m in forward_methods
    @eval $m(T:: Type, I::Vararg{<: Integer, N}) where {N} = RowMajorArray(Base.$m(T, reverse(I)...))
    @eval $m(I::Vararg{<: Integer, N}) where {N} = RowMajorArray(Base.$m(reverse(I)...))
end

end
