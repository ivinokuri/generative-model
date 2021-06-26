from system import PubSub, Topics, SystemClock
from robot import GenerativeRobot, State
from env import Location, MoveDirection
from env import World
from logic import MoveSimulator
import time
import threading

shutdown = False
sleepTime = 0.1

def systemTime():
	while not shutdown:
		SystemClock.instance.incrementTime(sleepTime)
		time.sleep(sleepTime)

def simulationSubcription(data):
	print(data)
	print(data)

def mainLoop(robot:GenerativeRobot, world:World):
	moveSim = MoveSimulator(robot, world)
	PubSub.instance.subscribe(Topics.simulation, simulationSubcription)
	moveSim.startRunning(1)
	simTime = threading.Thread(target=systemTime)
	simTime.start()
	simTime.join()
	# while not shutdown:
	# 	pass


if __name__ == "__main__":
	print("Init Generative module")
	robot:GenerativeRobot = GenerativeRobot(State(Location(0, 0), MoveDirection.STAND, 1))
	w = World.loadWorld()
	print(w)
	try:
		mainLoop(robot, w)
	except KeyboardInterrupt as ex:
		print(ex)
		shutdown = True
		print('Bye')

