#### AUTOMATA MODEL ####
## Eden Forbes

#### LIBRARIES

#### SCRIPTS
import numpy
import OBJECTS

#### DEFINE CONSTANTS ####
## Simulation Features
timesteps = 100
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
foodchance = 0.01


#### DEFINE WORLD ####
world = OBJECTS.World(timesteps,X,Y,start_CA,CAfood,foodregenz,
                      g1,g2,g3,g4,foodtoughlow,foodtoughhigh, 
                      foodvalue,foodloss,newfood,foodchance)

for t in range(timesteps):
    if world.checkEnd() == True:
        print("Total Steps: %d" % world.steps)
        break
    else:
        world.step()
        
numpy.savetxt('/Users/eden/Desktop/IU_FALL_2021/RESEARCH/MODEL/DATA/CAdata_11_12_21.csv', world.data, delimiter = ',')


