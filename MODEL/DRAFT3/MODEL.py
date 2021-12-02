#### AUTOMATA MODEL ####
## Eden Forbes

#### LIBRARIES

#### SCRIPTS
import OBJECTS

#### DEFINE CONSTANTS ####
## World Dimensions
X = 100
Y = 100

## Simulation Features
timesteps = 1000
X = 50
Y = 50
start_CA = 20
CAfood = 100
## Ignore IND food, comes later
INDfood = 0

## Food Features
g1 = 0.1
g2 = 0.05
g3 = 0.03
g4 = 0.01
foodvalue = 0.2
foodloss = 0.01
newfood = 1


#### DEFINE WORLD ####
world = OBJECTS.World(timesteps,X,Y,start_CA,CAfood,INDfood,
                      g1, g2, g3, g4, foodvalue,foodloss,newfood)

for t in range(timesteps):
    if world.checkEnd() == True:
        print("Total Steps: %d" % world.steps)
        break
    else:
        world.step()
        
 


