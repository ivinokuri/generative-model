from .pubsub import PubSub, Topics
from .utils import SingletonDecorator

class SystemClock:

	class __SystemClock:
		def __init__(self):
			self.systemTime = 0.0

		def incrementTime(self, sleepTime=1.0):
			self.systemTime += sleepTime
			roundTime = round(self.systemTime)
			PubSub.instance.publish(Topics.clock, {
				"topic": Topics.clock,
				"data": self.systemTime
			})
			if roundTime % 10 == 0 and 0 < self.systemTime - roundTime < sleepTime:
				print("System Clock: {} epoch passed \n".format(roundTime))

	instance:__SystemClock = None

	def __init__(self):
		if not SystemClock.instance:
			SystemClock.instance = SystemClock.__SystemClock()


_ = SystemClock()