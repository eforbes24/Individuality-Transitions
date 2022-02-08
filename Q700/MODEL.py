#### MULTICELLULAR AUTOMATA MODEL ####
## Eden Forbes

#### LIBRARIES
import numpy as np
#import cv2 as cv
#from PIL import Image
#from matplotlib import cm

#### SCRIPTS
import OBJECTS
 
#### DEFINE CONSTANTS ####
## Simulation Features
runs = 20
timesteps = 500 ## Length of the simulation
X = 50 ## X dimension of world
Y = 50 ## Y dimension of world
start_CA = 30 ## Number of initial unicellular CAs
CAfood = 70 ## Number of initial CA food sources

## Gradient Features
g1 = 0.5 ## Chemotaxis gradient one Manhatten distance from food source
g2 = 0.3 ## Chemotaxis gradient two Manhatten distance from food source
g3 = 0.1 ## Chemotaxis gradient three Manhatten distance from food source
g4 = 0.05 ## Chemotaxis gradient four Manhatten distance from food source
g5 = 0.01 ## Chemotaxis gradient five Manhatten distance from food source
g0 = 20*g1 ## Chemotaxis gradient at food source

## Food Features
clusters = 7 ## Number of food clusters in the world
cdev = 7 ## How internally spread clusters are, smaller numbers are greater distances
newcenterstep = 10 ## Number of times during the simulation you want new food clusters to form
foodtoughlow = 2 ## Minimum max food toughness in simulation
foodtoughhigh = 15 ## Maximum max food toughness in simulation
foodregenz = 0.4 ## How much food toughness is regenerated per step
nutrition = foodtoughlow * 0.06 ## ratio of reward to toughness of food
foodloss = 0.01 ## How much food agents lose per step
newfood = 5 # round(X/20) ## How much food might be introduced in a step
foodchance = 0.7 ## Chance that the amount of new food is introduced in a step

## Cell Features
lowfillstart = 0.6 ## Minimum stomach fill when CAs are initialized
highfillstart = 0.8 ## Maximum stomach fill when CAs are initialized
paniclevel = 0.3  ## Stomach fill level when CAs start seeking one another
dissolve_thresh = 1.0 ## Average stomach value in an individual at which point it splits back into CAs
diffusion = 0.1 ## Proportion of average fill in a multi diffused away from fuller and into less full cells
enz = 1 ## The amount of food toughness removed per step by a unicellular CA
diffenz = 3 ## The amount of food toughness removed per step by a differentiated CA
germenz = 0.5 ## The amount of food toughness removed per step by a germ CA

## Display
display = True

## RUN SIMULATION
CAtotal = list()
POPtotal = list()
for r in range(runs):
    ID = r
    #### DEFINE WORLD ####
    world = OBJECTS.World(ID,timesteps,X,Y,start_CA,CAfood,clusters,cdev,foodregenz,
                      newcenterstep,g0,g1,g2,g3,g4,g5,foodtoughlow,
                      foodtoughhigh,nutrition,lowfillstart,highfillstart,
                      foodloss,newfood,foodchance,paniclevel,dissolve_thresh,
                      diffusion,enz,diffenz,germenz,display)
    #### RUN WORLD ####
    for t in range(timesteps):
        if world.checkEnd() == True:
            for c in world.CAs:
                world.data.append(["cell", "world_end_uni", c.ID, c.start, c.end, 
                                  c.lifespan, c.inmulti, c.children])
            for m in world.Multis:
                world.data.append(["multi", "world_end", m.ID, m.start, 
                                      m.end, m.lifespan, m.size, 
                                      list(m.CAs)])
                for c in m.CAs:
                    world.data.append(["cell", "world_end_multi", c.ID, c.start, c.end, 
                                  c.lifespan, c.inmulti, c.children])
            print("Total Steps: %d" % world.steps)
#           writer.release()
            break
        else:
            world.step()
#           frame = (world.arrays[t]*255).astype(np.uint8)
#           writer.write(frame)
    #### SAVE DATA ####
    CAdata = world.data
    for row in CAdata:
        row.insert(0,r)
        CAtotal.append(row)
    POPdata = world.run
    for row in POPdata:
        row.insert(0,r)
        POPtotal.append(row)
outputCA_string = '/Users/eden/Desktop/IU_FALL_2021/RESEARCH/MODEL/DATA/MAIN/12_6_21(D8)_CAdata.csv'
outputPOP_string = '/Users/eden/Desktop/IU_FALL_2021/RESEARCH/MODEL/DATA/MAIN/12_6_21(D8)_POPdata.csv'
#np.savetxt(outputCA_string, CAtotal, delimiter = ',', fmt='%s')
#np.savetxt(outputPOP_string, POPtotal, delimiter = ',', fmt='%s')
        
## RECORD VIDEO
#fourcc = cv.VideoWriter_fourcc(*'MJPG')
#writer = cv.VideoWriter('/Users/eden/Desktop/IU_FALL_2021/RESEARCH/MODEL/DATA/MAIN/TEST1.avi',
#                         fourcc, 30, (100, 100), isColor = True)
#for frame in world.arrays:
#    im = Image.fromarray(np.uint8(cm.gist_earth(frame) * 255))
#    writer.write(im)
#writer.release()
