import random

class area:
  def __init__(self, x_min, x_max, y_min, y_max):
    self.x_min = x_min
    self.x_max = x_max
    self.y_min = y_min
    self.y_max = y_max

## ------------------------------------------------------------ areas ------------------------------------------------------------
def building_areas():
    area_1 = area(3.63597559929, 4.56262493134, -1, -0.1)
    area_2 = area(-5.22181129456, -4.34186172485, -0.946628332138, -0.340723633766)
    area_3 = area(-5.32811450958, -3.61236262321, 7.99595689774, 8.32002067566)
    area_4 = area(-0.504257440567, -0.4, 6.49937057495, 7.66467809677)
    return [area_1, area_2, area_3, area_4]


## cans_1
def cans_areas():
    area_1 = area(-5.60300111771, -1.56613826752, 1.17350196838, 1.41825914383)
    area_2 = area(8.85735607147, 8.88743591309, -1.05372858047, 2.52102136612)
    area_3 = area( 2.05089020729, 2.82445645332, 0.45054101944, 1.3149433136)
    return [area_1, area_2, area_3]

def corridor_areas():
    area_1 = area(4.29499530792, 4.96841192245, -1, -1.12654268742)
    area_2 = area(4.37633800507, 4.96841192245, 0.717599868774, 1.2 )
    area_3 = area(9.25203227997, 10.0935201645, -1, -1.12539849281)
    area_4 = area(10.8062801361, 12.3452634811, 0.5, 0.844116687775)
    area_5 = area(15.9066295624, 16.6821746826, -1, 0.706882476807)
    return [area_1, area_2, area_3, area_4, area_5]
    return []

## ------------------------------------------------------------ goals ------------------------------------------------------------

def create_goals(areas):
  goals = []
  for area in areas:
    goal_x = random.uniform(area.x_min, area.x_max)
    goal_y = random.uniform(area.y_min, area.y_max)
    goals.append((goal_x, goal_y))
  return goals

def building_goals_seperatly():
    [area_1, area_2, area_3, area_4] = building_areas()
    areas1 = [area_1,area_2]
    areas2 = [area_3,area_4]
    random.shuffle(areas1)
    random.shuffle(areas2)
    areas = areas1 + areas2
    return create_goals(areas)

def randomaly_areas(araes_func):
    areas = araes_func()
    random.shuffle(areas)
    return create_goals(areas)

def building_goals():
    return randomaly_areas(building_areas)

def cans_goals():
    return randomaly_areas(cans_areas)

def corridor_goals():
    return randomaly_areas(corridor_areas)