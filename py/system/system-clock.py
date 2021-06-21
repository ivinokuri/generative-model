from pubsub import PubSubInstance
from pubsub import Topics

class SystemClock:

	def __init__(self):
		self.systemTime = 0

	def incrementTime(self, sleepTime=1.0):
		self.systemTime += sleepTime
		roundTime = round(self.systemTime)
		PubSubInstance(Topics.clock, {
			"topic": Topics.clock,
			"data": roundTime
		})