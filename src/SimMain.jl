module SimMain
push!(LOAD_PATH, joinpath(pwd(), "system"))
push!(LOAD_PATH, joinpath(pwd(), "robot"))
push!(LOAD_PATH, joinpath(pwd(), "logic"))
push!(LOAD_PATH, joinpath(pwd(), "env"))

using Distributed

	using Loc
	import Base.Threads.@spawn
	import PubSub
	import GenEnv
	import UserInput
	import MoveSimulator
	import WorldEnv.World
	import WorldEnv
	import Loc.Location
	import Robot.GenerativeRobot
	import Robot.State


	function mainloop(robot::GenerativeRobot, world::World, comm_channel::Channel)
		shutdown = false
		sleep_time = 0.1
		c = PubSub.subscribe(PubSub.Topics[:position])
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
	export main
end
	if ! isinteractive()
		SimMain.main()
	end