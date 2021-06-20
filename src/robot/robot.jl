module Robot
include("sensor.jl")
include("robot-components.jl")
import Loc.Location
import Loc.MoveDirection

struct State
	location::Location
	moveDirection::MoveDirection
	velocity:Float16 = 1

	State(loc::Location, md::MoveDirection) = new(loc, md)
	State(loc::Location, md::MoveDirection, vel::Float16) = new(loc, md, vel)
end


mutable struct GenerativeRobot
	# sensors::Sensor
	currentState::State
	
	# rightRearWheel::Wheel
	# leftRearWheel::Wheel
	# rightFrontWheel::Wheel
	# leftFrontWheel::Wheel
end

function move(robot::GenerativeRobot, direction::MoveDirection, location::Location) 
	s = State(location, direction)
	robot.currentState = s
	data = Dict(
		"topic" => Topics[:position],
		"data" => s
	)
	publish(Topics[:position], data)
end

export GenerativeRobot, State
end