include("robot/robot.jl")
include("system/system-clock.jl")
include("system/user-input.jl")
include("system/pubsub.jl")

import Base.Threads.@spawn

using .PubSub
using .GenEnv
using .UserInput

function main_loop(robot::GenerativeRobot)
	shutdown = false
	sleep_time = 0.1
	while !shutdown
		# run update
		move(robot, forward, Location(1.0, robot.currentState.location.y + sleep_time))
		GenEnv.increment_time(sleep_time)
		sleep(sleep_time)
	end
end

function init_robot()
	location = Location(0.0, 0.0)
	state = State(location, stand)
	robot::GenerativeRobot = GenerativeRobot(state)
	return robot
end

function main()
	println("Init Generative module")
	robot = init_robot()
	# init world

	@spawn UserInput.wait_for_user_input()
	main_loop(robot)
end


if ! isinteractive()
    main()
end