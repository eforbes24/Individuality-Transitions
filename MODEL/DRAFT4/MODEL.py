#### AUTOMATA MODEL ####
## Eden Forbes

#### LIBRARIES

#### SCRIPTS
import OBJECTS

#### DEFINE CONSTANTS ####
## Simulation Features
timesteps = 1000
X = 50
Y = 50
start_CA = 10
CAfood = 75

## Food Features
g1 = 0.1
g2 = 0.05
g3 = 0.03
g4 = 0.01
foodregenz = 0.1
foodtoughlow = 1
foodtoughhigh = 3
foodvalue = 0.3
foodloss = 0.01
newfood = 1
foodchance = 0.2

## Cell Features
paniclevel = 0.2


#### DEFINE WORLD ####
world = OBJECTS.World(timesteps,X,Y,start_CA,CAfood,foodregenz,
                      g1,g2,g3,g4,foodtoughlow,foodtoughhigh, 
                      foodvalue,foodloss,newfood,foodchance,paniclevel)

for t in range(timesteps):
    if world.checkEnd() == True:
        print("Total Steps: %d" % world.steps)
        break
    else:
        world.step()
        
 


