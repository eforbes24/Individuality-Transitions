#### AUTOMATA MODEL ####
## Eden Forbes

#### LIBRARIES
import numpy as np

#### SCRIPTS
import OBJECTS
 
#### DEFINE CONSTANTS ####
## Simulation Features
timesteps = 1000 ## Length of the simulation
X = 50 ## X dimension of world
Y = 50 ## Y dimension of world
start_CA = 20 ## Number of initial unicellular CAs
CAfood = 30 ## Number of initial CA food sources

## Gradient Features
g0 = 2 ## Chemotaxis gradient at food source
g1 = 0.5 ## Chemotaxis gradient one Manhatten distance from food source
g2 = 0.3 ## Chemotaxis gradient two Manhatten distance from food source
g3 = 0.1 ## Chemotaxis gradient three Manhatten distance from food source
g4 = 0.05 ## Chemotaxis gradient four Manhatten distance from food source
g5 = 0.01 ## Chemotaxis gradient five Manhatten distance from food source

## Food Features
clusters = 3 ## Number of food clusters in the world
newcenterstep = 10 ## Number of times during the simulation you want new food clusters to form
foodtoughlow = 2 ## Minimum max food toughness in simulation
foodtoughhigh = 10 ## Maximum max food toughness in simulation
foodregenz = 0.4 ## How much food toughness is regenerated per step
nutrition = foodtoughlow * 0.06 ## ratio of reward to toughness of food
foodloss = 0.005 ## How much food agents lose per step
newfood = 1 ## How much food might be introduced in a step
foodchance = 0.25 ## Chance that the amount of new food is introduced in a step

## Cell Features
lowfillstart = 0.6 ## Minimum stomach fill when CAs are initialized
highfillstart = 0.8 ## Maximum stomach fill when CAs are initialized
paniclevel = 0.3  ## Stomach fill level when CAs start seeking one another
enz = 1 ## The amount of food toughness removed per step by a unicellular CA
diffenz = 3 ## The amount of food toughness removed per step by a differentiated CA
germenz = 0.75 ## The amount of food toughness removed per step by a germ CA

#### DEFINE WORLD ####
world = OBJECTS.World(timesteps,X,Y,start_CA,CAfood,clusters,foodregenz,newcenterstep,
                      g0,g1,g2,g3,g4,g5,foodtoughlow,foodtoughhigh,nutrition,
                      lowfillstart,highfillstart,foodloss,newfood,
                      foodchance,paniclevel,enz,diffenz,germenz)

for t in range(timesteps):
    if world.checkEnd() == True:
        print("Total Steps: %d" % world.steps)
        break
    else:
        world.step()
        
np.savetxt('/Users/eden/Desktop/IU_FALL_2021/RESEARCH/MODEL/DATA/MAIN/CAdata_11_12_21.csv', world.data, delimiter = ',', fmt='%s')
 


