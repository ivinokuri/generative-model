module GenEnv
	include("./pubsub.jl")

	using .PubSub
	using Printf

	mutable struct SystemClock
		system_time::Float64
	end

	const systemClock = GenEnv.SystemClock(0)

	function incrementtime(sleep_time=1.0) 
		systemClock.system_time += sleep_time
		round_time = round(systemClock.system_time)
		publish(PubSub.Topics[:clock], Dict(
			"topic" => PubSub.Topics[:clock],
			"data" => round_time
		))
		if round_time % 10 == 0 && systemClock.system_time - round_time > 0 && systemClock.system_time - round_time < sleep_time
			@printf("System Clock: %d epoch passed \n", round_time)
		end

	end

	export incrementtime

end
