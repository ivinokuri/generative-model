# import RobotComponents as rc

class State:
	def __init__(self, location, moveDirection, velocity=1) -> None:
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