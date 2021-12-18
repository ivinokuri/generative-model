from enum import Enum


class Topics(Enum):
	clock = "/clock"
	position = "/pose"
	velocity = "/vel"
	simulation = '/sim/data'

class PubSub:

	class __PubSub:
		def __init__(self):
			self.callbacks = dict()
			for t in Topics:
				self.callbacks[t] = set()

		def publish(self, topic, data):
			if self.callbacks[topic]:
				callbacks = self.callbacks[topic]
				for c in callbacks:
					c(data)
			else:
				print('Topic not found')

		def subscribe(self, topic, callback):
			callbacks = self.callbacks[topic]
			if not callbacks:
				callbacks = set()
			callbacks.add(callback)
			self.callbacks[topic] = callbacks

		def unsubscribe(self, topic, callback):
			callbacks = self.callbacks[topic]
			if callbacks:
				callbacks.remove(callback)
				self.callbacks = callbacks

	instance = None

	def __init__(self) -> None:
		if not PubSub.instance:
			PubSub.instance = PubSub.__PubSub()

_ = PubSub()
# PubSubInstance = SingletonDecorator(PubSub)