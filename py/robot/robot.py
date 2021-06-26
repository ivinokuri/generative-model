# import RobotComponents as rc
from env.location import Location
from env.location import MoveDirection

class State:
	def __init__(self, location, moveDirection, velocity) -> None:
		self.location = location
		self.moveDirection = moveDirection
		self.velocity = velocity

	def __str__(self) -> str:
		return "Location: " + self.location + " MD: " + self.moveDirection + " vel: " + self.velocity


class GenerativeRobot:
	def __init__(self, state) -> None:
		self.currentState = state

	def __str__(self) -> str:
		return "state: " + self.currentState
	
	def move(self, direction, location):
		s = State(location, direction)
		self.currentState = s
		# TODO send pubsub