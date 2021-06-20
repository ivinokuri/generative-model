from enum import Enum
import enum
from typing import Dict, Set

class Topics(Enum):
	clock = "/clock"
	position = "/pose"
	velocity = "/vel"

class PubSub:
	def __init__(self) -> None:
		self.channels = dict()
		for t in enum(Topics):
			self.channels[t] = set()
	
	def publish(self, topic, data):
		# if self.channels.has_key(topic):
		# 	cs = self.channels[topic]
		# 	for c in c:
		pass

	def subsctibe(self, topic):
		pass

	def unsubscribe(self, topic):
		pass