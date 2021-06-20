include("../env/location.jl")

struct State
	location::Location
	moveDirection::MoveDirection
	velocity:Float16 = 1

	State(loc::Location, md::MoveDirection) = new(loc, md)
	State(loc::Location, md::MoveDirection, vel::Float16) = new(loc, md, vel)
end

