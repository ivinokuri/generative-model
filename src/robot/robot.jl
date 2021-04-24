include("sensor.jl")
include("state.jl")
include("robot-components.jl")
include("../system/pubsub.jl")

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