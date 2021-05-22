module MoveSimulator

	include("../env/location.jl")
	include("../robot/robot.jl")
	using Distributions 
	_isrunning = false
	_velocity = 0
	# _robot::GenerativeRobot

	function simulatemove(sim_channel::Channel)
		while isrunning
			sleep(nextinterval())
			dir = randomdirection()
			location = calcnextloc(dir)
		end
	end

	function nextinterval()
		return 100
	end

	function randomdirection()
		direction = rand([forward, backward, stand, left, right])
		return direction
	end

	function calcnextloc(direction)
		if dir == forward
			println("forward")
		elseif dir == backward
			println("backward")
		elseif dir == left
			println("left")
		elseif dir == right
			println("right")
		else
			println("stand")
		end
	end

	function setrunning(isrunning)
		global _isrunning = isrunning
	end

	function setvelocity(vel)
		global _velocity = vel
	end

	function setrobot(robot)
	end

	export setrunning, setvelocity, nextinterval

end