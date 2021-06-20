module WorldEnv
import JSON
import Loc.Location
import Loc.MoveDirection

@enum WorldDirection begin
	north
	south
	east
	west
end

struct Obstacle
	position::Location
	length::Float16
	width::Float16
	direction::WorldDirection
end
struct World
	width::Float16
	height::Float16
	obstacles::Array{Obstacle}
end

function loadworld(path="./world.json")
	println("Loading world from json")
	data = open(f -> read(f, String), path, "r")
	wdict = Dict()
	wdict = JSON.parse(data) 
	obstacles = []
	for o in wdict["obstacles"]
		l = Location(o["location"]["x"], o["location"]["y"])
		push!(obstacles, Obstacle(l, o["length"], o["width"], WorldDirection(o["direction"])))
	end
	world = World(wdict["bounds"]["width"], wdict["bounds"]["height"], obstacles)
	return world
end

export WorldDirection, World, Obstacle, loadworld

end