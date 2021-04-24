module GenEnv

using Printf

mutable struct SystemClock
	system_time::Float64
end

systemClock = SystemClock(0)

function incrementtime(sleep_time=1.0) 
	systemClock.system_time += sleep_time
	round_time = round(systemClock.system_time)
	if round_time % 10 == 0 && systemClock.system_time - round_time > 0 && systemClock.system_time - round_time < sleep_time
		@printf("System Clock: %d epoch passed \n", round_time)
	end
end

export incrementtime

end