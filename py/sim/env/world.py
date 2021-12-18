from enum import Enum
from .location import Location
import json

class WorldDirection(Enum):
	NORTH = 0
	SOUTH = 1
	EAST = 2
	WEST = 3


class Obstacle:
	def __init__(self, position:Location, length, width, direction:WorldDirection):
		self.position = position
		self.length = length
		self.width = width
		self.direction = direction

	def __str__(self):
		return "At " + str(self.position) + " length: " + str(self.length) + " width: " + str(self.width) + " direction: " + str(self.direction.name)

class World:

	def __init__(self, width, height, obstacles):
		self.width = width
		self.height = height
		self.obstacles = obstacles

	def __str__(self) -> str:
		return "W: " + str(self.width) + " H: " + str(self.height) + " obs: " + str(self.obstacles)

	@staticmethod
	def loadWorld(path="./world.json"):
		print("Loading world from json");
		f = open(path)
		data = json.load(f)
		obstacles = []
		for o in data['obstacles']:
			l = Location(o['location']['x'], o['location']['y'])
			obstacles.append(Obstacle(l, o['length'], o['width'], WorldDirection(o['direction'])))
		world = World(data["bounds"]['width'], data["bounds"]['height'], obstacles)
		return world



