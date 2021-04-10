include("sensor.jl")
include("state.jl")
include("robot-components.jl")

mutable struct GenerativeRobot
	# sensors::Sensor
	currentState::State
	
	# rightRearWheel::Wheel
	# leftRearWheel::Wheel
	# rightFrontWheel::Wheel
	# leftFrontWheel::Wheel
end

function move(robot, direction::MoveDirection, location::Location) 
	s = get_state(location, direction)
	robot.currentState = s
end