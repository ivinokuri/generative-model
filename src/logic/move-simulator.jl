module MoveSimulator

	using Distributions 
	_isrunning = false
	_velocity = 0
	_currenttime = 0
	# _robot::GenerativeRobot = GenerativeRobot()

	c = PubSub.subscribe(PubSub.Topics[:clock])

	function simulatemove(sim_channel::Channel)
		while isrunning
			sleep(nextinterval())
			if isready(c)
				global _currenttime = take!(c).data
			end
			dir = randomdirection()
			location = calcnextloc(dir)
			put!(sim_channel, location)
		end
	end

	function nextinterval()
		return rand(Distributions.Gamma(2, 5), 1) * 100
	end

	function randomdirection()
		direction = rand([forward, backward, stand, left, right])
		return direction
	end

	function calcnextloc(direction)
		loc = _robot.state.location
		x = loc.x
		y = loc.y
		if dir == forward
			println("forward")
			y = y + _velocity * _currenttime
		elseif dir == backward
			println("backward")
			y = y - _velocity * _currenttime
		elseif dir == left
			println("left")
			x = x - _velocity * _currenttime
		elseif dir == right
			println("right")
			x = x + _velocity * _currenttime
		else
			println("stand")
		end
		loc = Location(x, y)
		move(_robot, direction, loc)
		return loc
	end

	function setrunning(isrunning)
		global _isrunning = isrunning
	end

	function setvelocity(vel)
		global _velocity = vel
	end

	function setrobot(robot)
		global _robot = robot
	end

	export setrunning, setvelocity, nextinterval

end