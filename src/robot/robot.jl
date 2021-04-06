module Robots

include("sensor.jl")
include("state.jl")
include("robot-components.jl")

struct GenerativeRobot
	sensors::Sensor
	currentState::State
	
	rightRearWheel::Wheel
	leftRearWheel::Wheel
	rightFrontWheel::Wheel
	leftFrontWheel::Wheel
end


end