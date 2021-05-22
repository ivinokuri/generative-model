
include("../env/location.jl")
include("../robot/robot.jl")


module MoveSimulator

	_isrunning = false
	_velocity = 0
	_robot:GenerativeRobot

	function simulatemove(sim_channel::Channel)
		while isrunning
			sleep(nextinterval())
			dir = randomdirection()

		end
	end

	function nextinterval() 
		return 100
	end

	function randomdirection()
		return forward
	end

	function calcnextloc(direction)
		

	end

	function setrunning(isrunning)
		MoveSimulator._isrunning = isrunning
	end

	function setvelocity(vel)
		MoveSimulator._velocity = vel
	end

	function setrobot(robot)
	end

	export setrunning, setvelocity

end