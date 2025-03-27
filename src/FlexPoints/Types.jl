module Types

export Points2D, x, y

# First element of the tuple is a value of independent variable
# Second element of the tuple is a value of dependent variable
Points2D = Vector{Tuple{T,T}} where {T<:Real}

x(data::Points2D, index::Integer) = data[index][1]
x(point::Tuple{Real,Real}) = point[1]

y(data::Points2D, index::Integer) = data[index][2]
y(point::Tuple{Real,Real}) = point[2]

end