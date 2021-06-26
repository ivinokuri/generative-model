from env.location import MoveDirection,Location
from robot.robot import GenerativeRobot
from system.pubsub import PubSubInstance, Topics
class MoveSimulator:

	def __init__(self, robot:GenerativeRobot):
		self.isRunning = False
		self.velocity = 0
		self.currentTime = 0
		self.robot = robot

	def receiveClock(self, data):
		clock = data.data
		self.currentTime = clock

	def startRunning(self, velocity):
		self.isRunning = True
		self.velocity = velocity
		PubSubInstance.subscribe(Topics.clock, self.receiveClock)
		self.simulateMove()

	def simulateMove(self):
		while self.isRunning:
			# TODO sleep
			pass

	def nextinterval(self):
		return 0

	
	def randomDirection(self):
		return MoveDirection.STAND

	def calcNextLoc(self, direction, timePass):
		loc:Location = self.robot.currentState.location
		x = loc.x
		y = loc.y
		if direction == MoveDirection.FORWARD:
			print('Move forward')
			y = y + self.velocity * timePass
		elif direction == MoveDirection.BACKWARD:
			print('Move backward')
			y = y - self.velocity * timePass
		elif direction == MoveDirection.LEFT:
			print('Move left')
			x = x - self.velocity * timePass
		elif direction == MoveDirection.RIGHT:
			print('Move right')
			x = x + self.velocity * timePass
		else:
			print('Standing')
		newLoc = Location(x, y)
		self.robot.move(direction, newLoc)
		return newLoc