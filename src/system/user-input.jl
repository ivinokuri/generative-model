module UserInput

function wait_for_user_input()
	println("============= Menu ============")
	println("Use v to change the velocity") 
	println("Use q to quite")
	println("===============================")
	print("Running simulation ...")
	while true
		user_input = readline()
		if user_input == "v"
			println("Not implemented")
		elseif user_input == "q"
			println("Bye!")
			exit(0)
		else
			println("Unknown input")
		end
	end
end

# input_channel = Channel() do 
# 	wait_for_user_input()
# end

export wait_for_user_input

end