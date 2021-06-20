
from os import name


class Wheel:

	def __init__(self, velocity, radialVelocity, angle):
		self.velocity = velocity
		self.radialVelocity = radialVelocity
		self.angle = angle

	def __str__(self) -> str:
		return "V: " + self.velocity + " rV: " + self.radialVelocity + " a: " + self.angle


class GrabHand:

	def __init__(self, failProb, value, holding=False):
		self.failProb = failProb
		self.value = value
		self.holding = holding


	def __str__(self) -> str:
		return "fP: " + self.failProb + " v: " + self.value + " h: " + self.holding

class Sensor: 

	def __init__(self, name, failProb, value):
		self.name = name
		self.failProb = failProb
		self.value = value

	def __str__(self) -> str:
		return "Name: " + self.name + " fP: " + self.failProb + " v: " + self.value
		