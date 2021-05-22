include("robot/robot.jl")
include("system/system-clock.jl")
include("system/user-input.jl")
include("system/pubsub.jl")
include("env/world.jl")
include("logic/move-simulator.jl")

import Base.Threads.@spawn

using .PubSub
using .GenEnv
using .UserInput
using .WorldEnv
using .MoveSimulator

function mainloop(robot::GenerativeRobot, world::World, comm_channel::Channel)
	shutdown = false
	sleep_time = 0.1
	c = PubSub.subscribe(Topics[:position])
	MoveSimulator.setrunning(true)
	while !shutdown
		# run update
		if isready(comm_channel)
			ui = take!(comm_channel)
			print(ui)
		end
		if isready(c)
			println("from channel", take!(c))
		end
		# move(robot, forward, Location(1.0, robot.currentState.location.y + sleep_time))
		MoveSimulator.randomdirection()
		GenEnv.incrementtime(sleep_time)
		sleep(sleep_time)
	end
end

function initrobot()
	location::Location = Location(0.0, 0.0)
	state::State = State(location, stand)
	robot::GenerativeRobot = GenerativeRobot(state)
	return robot
end

function inituserinput()
	comm_channel = Channel(1)
	@spawn UserInput.userinput(comm_channel)
	return comm_channel
end

function main()
	println("Init Generative module")
	robot = initrobot()
	comm_channel = inituserinput()
	w = WorldEnv.loadworld()
	println(w)
	mainloop(robot, w, comm_channel)
end


if ! isinteractive()
    main()
end