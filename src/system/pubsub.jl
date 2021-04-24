module PubSub

Topics = Base.ImmutableDict(
	:position => "/pose", 
	:velocity => "/vel")

channels = Dict{String, Set{Channel}}()

for (k, v) in Topics
	channels[v] = Set{Channel}()
end

function publish(topic::String, data::Any)
	if haskey(channels, topic)
		cs = channels[topic]
		for c in cs
			put!(c, data)
		end
	end
end

function subscribe(topic::String)
	cs = get!(channels, topic, Set())
	channel = Channel(1)
	push!(cs, channel)
	return channel
end

function unsubscribe(topic::String, channel::Channel)
	if haskey(channels, topic)
		cs = get!(channels, topic, [])
		delete!(cs, channel)
	end
end

export Topics, publish, subscribe, unsubscribe

end