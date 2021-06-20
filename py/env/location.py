from enum import Enum

class MoveDirection(Enum):
	STAND = 0
	FORWARD = 1
	BACKWARD = 2
	LEFT = 3
	RIGHT = 4


class Location:
	
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def getX(self):
		return self.x

	def getY(self):
		return self.y

	def __str__(self) -> str:
		return "Location x:" + self.x + " y:" + self.y