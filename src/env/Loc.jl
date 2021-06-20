module Loc
@enum MoveDirection begin
	stand
	forward
	backward
	left
	right
end
struct Location
	x::Float64
	y::Float64
end

export Location, MoveDirection, stand, forward, backward, left, right
end
