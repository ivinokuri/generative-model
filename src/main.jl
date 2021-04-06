include("robot/robot.jl")

import Robot

function init()
	print("Init Generative module")
	robot::Robot = Robot()
end

init()