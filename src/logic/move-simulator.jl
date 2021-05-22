
include("../env/location.jl")

module MoveSimulator
	using 

	isrunning = false

	function simulatemove(sim_channel::Channel)
		while isrunning
			sleep(nextinterval())
			
		end
	end

	function nextinterval() 
		return 100
	end

	function randomdirection()
		return forward
	end

end