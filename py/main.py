from system import PubSubInstance, Topics
from robot import GenerativeRobot, State
from env import Location, MoveDirection
from env import World
from logic import MoveSimulator

def simulationSubcription(data):
	print(data.topic)
	print(data.data)

def mainLoop(robot:GenerativeRobot, world:World):
	shutdown = False;
	moveSim = MoveSimulator(robot, world)
	PubSubInstance.subscribe(Topics.simulation, simulationSubcription)
	moveSim.startRunning(1)


if __name__ == "__main__":
	print("Init Generative module")
	robot:GenerativeRobot = GenerativeRobot(State(Location(0, 0), MoveDirection.STAND, 1))
	w = World.loadWorld()
	print(w)