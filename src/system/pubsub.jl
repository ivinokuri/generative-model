module PubSub
using InteractiveUtils
const global Topics = Base.ImmutableDict(
	:clock => "/clock",
	:position => "/pose", 
	:velocity => "/vel")

_topics() = Topics
	
const global channels = Dict{String, Set{Channel}}()
@code_llvm _topics()

for (k, v) in Topics
	channels[v] = Set{Channel}()
end

function publish(topic::String, data::Any)
	if haskey(channels, topic)
		cs = channels[topic]
		println(length(cs))
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