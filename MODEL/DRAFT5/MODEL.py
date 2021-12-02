#### AUTOMATA MODEL ####
## Eden Forbes

#### LIBRARIES
import numpy as np

#### SCRIPTS
import OBJECTS

#### DEFINE CONSTANTS ####
## Simulation Features
timesteps = 1000
X = 40
Y = 40
start_CA = 15
CAfood = 25

## Gradient Features
g0 = 1
g1 = 0.5
g2 = 0.3
g3 = 0.1
g4 = 0.05
g5 = 0.01

## Food Features
foodregenz = 0.05
foodtoughlow = 1
foodtoughhigh = 3
nutrition = 0.3 ## ratio of reward to toughness of food
foodloss = 0.005
newfood = 1
foodchance = 0.2

## Cell Features
lowfillstart = 0.4
highfillstart = 0.6
paniclevel = 0.4

#### DEFINE WORLD ####
world = OBJECTS.World(timesteps,X,Y,start_CA,CAfood,foodregenz,
                      g0,g1,g2,g3,g4,g5,foodtoughlow,foodtoughhigh,nutrition,
                      lowfillstart,highfillstart,foodloss,newfood,
                      foodchance,paniclevel)

for t in range(timesteps):
    if world.checkEnd() == True:
        print("Total Steps: %d" % world.steps)
        break
    else:
        world.step()
        
np.savetxt('/Users/eden/Desktop/IU_FALL_2021/RESEARCH/MODEL/DATA/MAIN/CAdata_11_12_21.csv', world.data, delimiter = ',', fmt='%s')
 


