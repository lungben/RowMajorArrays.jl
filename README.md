# RowMajorArrays

## Introduction

In Julia, arrays are column-major, i.e. the data is stored in memory column by column. This is analogue to languages like Fortran and Matlab, but different to C and Python (Numpy), where arrays are stored row-major.

This package provides a light-weight implementation of row-major arrays in Julia.

It is a wrapper over a column-major array which reverses the dimensions for outside usage, effectively making it a row-major array.

## Installation

    ] add https://github.com/lungben/RowMajorArrays.jl

## Usage

    using RowMajorArrays
    a = [1 2 3; 4 5 6] # 2x3 (column major) array
    a_r = RowMajorArray(a) # 3x2 row-major array - the data is not copied, just wrapped. The dimensions are reverted.
    @assert a[1, 3] == a_r[3, 1]

    ones_r = RowMajorArrays.ones(5, 3) # constructor functions similar to the ones in Base.
    zeros_r = RowMajorArrays.zeros(5, 3)
    rand_r = RowMajorArrays.rand(5, 3)
    randn_r = RowMajorArrays.randn(5, 3)

    combined_r = ones_r .+ rand_r .* randn_r # broadcasting, gives RowMajorArray
    combined_r.data # gives underlying column major array (with reversed dimensions)
    transpose(combined_r.data) == combined_r # transpose is only defined in 2 dimensions

## Performance

As expected, RowMajorArrays are faster when iterating over the row-by-row.

Output of `test/performance.jl`:

    Array row major sum
    1.636 ms (0 allocations: 0 bytes)
    Array column major sum
    973.700 μs (0 allocations: 0 bytes)
    RowMajorArray row major sum
    971.900 μs (0 allocations: 0 bytes)
    RowMajorArray column major sum
    1.618 ms (0 allocations: 0 bytes)

## Disclaimer

This package is still experimental and has primarily been written for my own education.

Most Julia code is optimized for column-major arrays, usage of row-major arrays can significantly reduce performance in 3rd party libraries.
