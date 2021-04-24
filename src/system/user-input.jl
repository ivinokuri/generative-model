module UserInput

function userinput(comm_channel::Channel)
	println("============= Menu ============")
	println("Use v to change the velocity") 
	println("Use q to quite")
	println("===============================")
	println("Running simulation ...")
	while true
		user_input = readline()
		if user_input == "q"
			println("Bye!")
			exit(0)
		else
			put!(comm_channel, user_input)
		end
	end
end

export userinput

end