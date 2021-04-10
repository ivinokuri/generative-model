module PubSub

Topics = Base.ImmutableDict(
	:position => "/pose", 
	:velocity => "/vel")

callbacks = Dict()

for (k, v) in Topics
	callbacks[v] = Set()
end

function publish(topic, data::Any)
	if haskey(callbacks, topic)
		cs = callbacks[topic]
		for c in cs
			c(data)
		end
	end
end

function subscribe(topic, callback)
	cs = get!(callbacks, topic, Set())
	push!(cs, callback)
end

function unsubscribe(topic, callback)
	if haskey(callbacks, topic)
		cs = get!(callbacks, topic, [])
		delete!(cs, callback)
	end
end

export Topics, publish, subscribe, unsubscribe

end