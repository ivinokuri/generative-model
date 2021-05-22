include("../env/location.jl")

struct State
	location::Location
	moveDirection::MoveDirection
	velocity:Float16 = 1
end

