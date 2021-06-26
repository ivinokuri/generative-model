from enum import Enum
import enum
from typing import Dict, Set
from system.utils import SingletonDecorator

class Topics(Enum):
	clock = "/clock"
	position = "/pose"
	velocity = "/vel"

class PubSub:
	def __init__(self) -> None:
		self.callbacks = dict()
		for t in enum(Topics):
			self.callbacks[t] = set()
	
	def publish(self, topic, data):
		if self.callbacks.has_key(topic):
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
		self.callbacks = callbacks

	def unsubscribe(self, topic, callback):
		callbacks = self.callbacks[topic]
		if callbacks:
			callbacks.remove(callback)
			self.callbacks = callbacks

PubSubInstance = SingletonDecorator(PubSub)