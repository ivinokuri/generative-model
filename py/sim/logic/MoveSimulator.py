from sim.env.location import MoveDirection
from sim.robot.robot import GenerativeRobot
from sim.system.pubsub import PubSub, Topics
from pyro.distributions import Gamma
import torch

import random
from sim.env import Location
from sim.env import World

class MoveSimulator:

	def __init__(self, robot:GenerativeRobot, world:World):
		self.isRunning = False
		self.velocity = 0
		self.currentTime = 0
		self.prevTime = 0
		self.waitTime = 0
		self.robot = robot
		self.world = world
		self.gamma:Gamma = Gamma(torch.tensor([2.0]), torch.tensor([5.0]))

	def receiveClock(self, data):
		clock = data['data']
		self.currentTime = clock
		self.simulateMove()

	def startRunning(self, velocity):
		self.isRunning = True
		self.velocity = velocity
		PubSub.instance.subscribe(Topics.clock, self.receiveClock)

	def simulateMove(self):
		timePass = self.currentTime - self.prevTime
		if self.waitTime <= timePass:
			self.waitTime = self.nextInterval()
			self.prevTime = self.currentTime
			print("Wait for " + str(self.waitTime))
			loc, dir = self.calcNextLoc(timePass)
			PubSub.instance.publish(Topics.simulation, {
				"topic": Topics.simulation,
				"data": {
					"direction": dir,
					"newLocation": loc
				}
			})

	def nextInterval(self):
		nextInter = self.gamma.sample() * 10
		print('Next interval')
		return nextInter

	# Improve planning
	# Gen in julia
	@staticmethod
	def randomDirection():
		return random.choice([MoveDirection.FORWARD, 
			MoveDirection.BACKWARD, 
			# MoveDirection.STAND,
			MoveDirection.LEFT, 
			MoveDirection.RIGHT])

	def calcNextLoc(self, timePass):
		locationFit = False
		direction = MoveDirection.STAND
		loc: Location = self.robot.currentState.location
		x = loc.x
		y = loc.y
		while not locationFit:
			direction = self.randomDirection()
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
			if 0 <= x < self.world.width and 0 <= y < self.world.height:
				locationFit = True
			else:
				x = loc.x
				y = loc.y
		newLoc = Location(x, y)
		self.robot.move(direction, newLoc)
		return newLoc, direction
