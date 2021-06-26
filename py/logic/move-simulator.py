from env.location import MoveDirection,Location
from robot.robot import GenerativeRobot
from system.pubsub import PubSubInstance, Topics
from torch.distributed.gamma import Gamma
import torch

import random
class MoveSimulator:

	def __init__(self, robot:GenerativeRobot):
		self.isRunning = False
		self.velocity = 0
		self.currentTime = 0
		self.prevTime = 0
		self.robot = robot
		self.gamma = Gamma(torch.tensor([2.0]), torch.tensor([5.0]))

	def receiveClock(self, data):
		clock = data.data
		self.prevTime = self.currentTime
		self.currentTime = clock

	def startRunning(self, velocity):
		self.isRunning = True
		self.velocity = velocity
		PubSubInstance.subscribe(Topics.clock, self.receiveClock)
		self.simulateMove()

	def simulateMove(self):
		while self.isRunning:
			# TODO sleep
			timePass = self.currentTime - self.prevTime
			dir = self.randomDirection()
			loc = self.calcNextLoc(dir, timePass)
			PubSubInstance.publish(Topics.simulation, {
				"topic": Topics.simulation,
				"data": {
					"direction": dir,
					"newLocation": loc
				}
			})

	def nextInterval(self):
		nextInter = self.gamma.sample() * 100
		print('Next interval')
		return nextInter

	
	def randomDirection(self):
		return random.choice([MoveDirection.FORWARD, 
			MoveDirection.BACKWARD, 
			MoveDirection.STAND, 
			MoveDirection.LEFT, 
			MoveDirection.RIGHT])

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