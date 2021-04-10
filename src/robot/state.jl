include("../env/location.jl")

struct State
	location::Location
	moveDirection::MoveDirection
end

function get_state(location::Location, direction::MoveDirection) 
	return State(location, direction)
end

