module UserInput

function wait_for_user_input() 
	while true
		user_input = readline()
		println(user_input)
		if user_input == "q"
			println("Bye!")
			exit(0)
		end
	end
end

# input_channel = Channel() do 
# 	wait_for_user_input()
# end

export wait_for_user_input

end