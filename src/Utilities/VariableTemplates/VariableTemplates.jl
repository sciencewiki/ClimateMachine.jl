module VariableTemplates

export varsize, Vars, Grad, @vars, units, value,
       unit_scale, get_T, UV

using StaticArrays, Unitful
import Unitful: AbstractQuantity

"""
    varsize(S)

The number of elements specified by the template type `S`.
"""
varsize(::Type{T}) where {T<:Real} = 1
varsize(::Type{Tuple{}}) = 0
varsize(::Type{NamedTuple{(),Tuple{}}}) = 0
varsize(::Type{SVector{N,T}}) where {N,T<:Real} = N

include("var_names.jl")

# TODO: should be possible to get rid of @generated
@generated function varsize(::Type{S}) where {S}
  types = fieldtypes(S)
  isempty(types) ? 0 : sum(varsize, types)
end

function process_vars!(syms, typs, expr)
  if expr isa LineNumberNode
     return
  elseif expr isa Expr && expr.head == :block
    for arg in expr.args
      process_vars!(syms, typs, arg)
    end
    return
  elseif expr.head == :(::)
    push!(syms, expr.args[1])
    push!(typs, expr.args[2])
    return
  else
    error("Invalid expression")
  end
end

macro vars(args...)
  syms = Any[]
  typs = Any[]
  for arg in args
    process_vars!(syms, typs, arg)
  end
  :(NamedTuple{$(tuple(syms...)), Tuple{$(esc.(typs)...)}})
end

struct GetVarError <: Exception
  sym::Symbol
end
struct SetVarError <: Exception
  sym::Symbol
end

UV{N<:AbstractFloat} = Union{Quantity{N,D,U}, N} where {D,U}

"""
    Vars{S,A,offset}(array::A)

Defines property overloading for `array` using the type `S` as a template. `offset` is used to shift the starting element of the array.
"""
struct Vars{S,A,offset}
  array::A
end
Vars{S}(array) where {S} = Vars{S,typeof(array),0}(array)

Base.parent(v::Vars) = getfield(v,:array)
Base.eltype(v::Vars) = eltype(parent(v))
Base.propertynames(::Vars{S}) where {S} = fieldnames(S)
Base.similar(v::Vars{S,A,offset}) where {S,A,offset} = Vars{S,A,offset}(similar(parent(v)))

"""
    units(FT::Type{T} where {T<:Number}, u::Unitful.Units)

```jldoctest
julia> units(Float64, u"m*s^-2")
Quantity{Float64,𝐋*𝐓^-2,Unitful.FreeUnits{(m, s^-2),𝐋*𝐓^-2,nothing}}
```

Returns the type variable for the choice of numeric backing type `FT` and preferred units `u`.
"""
units(FT::Type{T} where {T<:Number}, u::Unitful.Units) = Quantity{FT, dimension(u), typeof(upreferred(u))}

"""
    unit_scale(::Type{NamedTuple{S, T}} where {S, T<:Tuple}, factor)

Scale the output of a @vars call by the provided units.
"""
@generated function unit_scale(::Type{NamedTuple{S, T}}, factor) where {S, T<:Tuple}
  p(Q,u) = begin
    if Q <: NamedTuple
      return unit_scale(Q, u)
    elseif Q <: SArray
      N = Q.parameters[2]
      return SArray{Q.parameters[1], typeof(oneunit(N)*u), Q.parameters[3], Q.parameters[4]}
    elseif Q <: SHermitianCompact
      N = Q.parameters[2]
      return SHermitianCompact{Q.parameters[1], typeof(oneunit(N)*u), Q.parameters[3]}
    else
      return typeof(oneunit(Q)*u)
    end
  end
  :(return $(NamedTuple{S, Tuple{p.(T.parameters, factor())...}}))
end

"""
Remove unit annotations, return float in SI units.
"""
value(x::Number) = ustrip(upreferred(x))

"""
Get the numeric backing type for a quantity or numeric scalar type.
"""
get_T(::Type{T}) where {T<:Number} = T
get_T(::Type{T}) where {T<:AbstractQuantity} = Unitful.get_T(T)


@generated function Base.getproperty(v::Vars{S,A,offset}, sym::Symbol) where {S,A,offset}
  expr = quote
    Base.@_inline_meta
    array = parent(v)
  end
  for k in fieldnames(S)
    T = fieldtype(S,k)
    ST = eltype(T)
    if T <: Number
      retexpr = :($T(array[$(offset+1)]))
      offset += 1
    elseif T <: SHermitianCompact
      LT = StaticArrays.lowertriangletype(T)
      N = length(LT)
      U = unit(ST)
      retexpr = :($T($LT($([:(array[$(offset + i)]*$U) for i = 1:N]...))))
      offset += N
    elseif T <: StaticArray
      N = length(T)
      U = unit(ST)
      retexpr = :($T($([:(array[$(offset + i)]*$U) for i = 1:N]...)))
      offset += N
    else
      retexpr = :(Vars{$T,A,$offset}(array))
      offset += varsize(T)
    end
    push!(expr.args, :(if sym == $(QuoteNode(k))
      return @inbounds $retexpr
    end))
  end
  push!(expr.args, :(throw(GetVarError(sym))))
  expr
end

@generated function Base.setproperty!(v::Vars{S,A,offset}, sym::Symbol, val::RT) where {S,A,offset,RT}
  expr = quote
    Base.@_inline_meta
    array = parent(v)
  end

  R = eltype(RT)
  for k in fieldnames(S)
    T = fieldtype(S,k)
    ST = eltype(T)
    if T <: Number
      U = inv(unit(R))
      retexpr = :(array[$(offset+1)] = val * $U)
      offset += 1
    elseif T <: SHermitianCompact
      LT = StaticArrays.lowertriangletype(T)
      N = length(LT)
      U = inv(unit(R))
      retexpr = :(array[$(offset + 1):$(offset + N)] .= $T(val).lowertriangle*$U)
      offset += N
    elseif T <: StaticArray
      N = length(T)
      U = inv(unit(R))
      retexpr = :(array[$(offset + 1):$(offset + N)] .= val[:]*$U)
      offset += N
    else
      offset += varsize(T)
      continue
    end
    push!(expr.args, :(if sym == $(QuoteNode(k))
      unit(eltype(RT)) != unit($ST) && eltype(RT) <: AbstractQuantity && error("Incompatible units for assignment: $($(unit(ST))) and $($(unit(eltype(RT))))")
      return @inbounds $retexpr
    end))
  end
  push!(expr.args, :(throw(SetVarError(sym))))
  expr
end

"""
    Grad{S,A,offset}(array::A)

Defines property overloading along slices of the second dimension of `array` using the type `S` as a template. `offset` is used to shift the starting element of the array.
"""
struct Grad{S,A,offset}
  array::A
end
Grad{S}(array) where {S} = Grad{S,typeof(array),0}(array)

Base.parent(g::Grad) = getfield(g,:array)
Base.eltype(g::Grad) = eltype(parent(g))
Base.propertynames(::Grad{S}) where {S} = fieldnames(S)
Base.similar(g::Grad{S,A,offset}) where {S,A,offset} = Grad{S,A,offset}(similar(parent(g)))

@generated function Base.getproperty(v::Grad{S,A,offset}, sym::Symbol) where {S,A,offset}
  M = size(A,1)
  expr = quote
    Base.@_inline_meta
    array = parent(v)
  end

  for k in fieldnames(S)
    T = fieldtype(S,k)
    ST = eltype(T)
    if T <: Number
      U = unit(T)
      retexpr = :(SVector{$M,$T}($([:(array[$i,$(offset+1)]*$U) for i = 1:M]...)))
      offset += 1
    elseif T <: StaticArray
      N = length(T)
      U = unit(ST)
      retexpr = :(SMatrix{$M,$N,$(eltype(T))}($([:(array[$i,$(offset + j)]*$U) for i = 1:M, j = 1:N]...)))
      offset += N
    else
      retexpr = :(Grad{$T,A,$offset}(array))
      offset += varsize(T)
    end
    push!(expr.args, :(if sym == $(QuoteNode(k))
      return @inbounds $retexpr
    end))
  end
  push!(expr.args, :(throw(GetVarError(sym))))
  expr
end

@generated function Base.setproperty!(v::Grad{S,A,offset}, sym::Symbol, val::RT) where {S,A,offset,RT}
  M = size(A,1)
  expr = quote
    Base.@_inline_meta
    array = parent(v)
  end
  for k in fieldnames(S)
    T = fieldtype(S,k)
    ST = eltype(T)
    if T <: Number
      U = inv(unit(T))
      retexpr = :(array[:, $(offset+1)] .= val*$U)
      offset += 1
    elseif T <: StaticArray
      U = inv(unit(ST))
      N = length(T)
      retexpr = :(array[:, $(offset + 1):$(offset + N)] .= val*$U)
      offset += N
    else
      offset += varsize(T)
      continue
    end
    push!(expr.args, :(if sym == $(QuoteNode(k))
      unit(eltype(RT)) != unit($ST) && eltype(RT) <: AbstractQuantity && error("Incompatible units for assignment: $($(unit(ST))) and $($(unit(eltype(RT))))")
      return @inbounds $retexpr
    end))
  end
  push!(expr.args, :(throw(SetVarError(sym))))
  expr
end

end # module
