#### MULTICELLULAR AUTOMATA MODEL ####
## Eden Forbes

#### LIBRARIES
import numpy as np
# import cv2

#### SCRIPTS
import OBJECTS
 
#### DEFINE CONSTANTS ####
## Simulation Features
timesteps = 1000 ## Length of the simulation
X = 80 ## X dimension of world
Y = 80 ## Y dimension of world
start_CA = 50 ## Number of initial unicellular CAs
CAfood = 100 ## Number of initial CA food sources

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
foodregenz = 0.3 ## How much food toughness is regenerated per step
nutrition = foodtoughlow * 0.1 ## ratio of reward to toughness of food
foodloss = 0.005 ## How much food agents lose per step
newfood = round(X/20) ## How much food might be introduced in a step
foodchance = 0.5 ## Chance that the amount of new food is introduced in a step

## Cell Features
lowfillstart = 0.7 ## Minimum stomach fill when CAs are initialized
highfillstart = 0.9 ## Maximum stomach fill when CAs are initialized
paniclevel = 0.3  ## Stomach fill level when CAs start seeking one another
enz = 1 ## The amount of food toughness removed per step by a unicellular CA
diffenz = 3 ## The amount of food toughness removed per step by a differentiated CA
germenz = 0.5 ## The amount of food toughness removed per step by a germ CA

#### DEFINE WORLD ####
world = OBJECTS.World(timesteps,X,Y,start_CA,CAfood,clusters,cdev,foodregenz,
                      newcenterstep,g0,g1,g2,g3,g4,g5,foodtoughlow,
                      foodtoughhigh,nutrition,lowfillstart,highfillstart,
                      foodloss,newfood,foodchance,paniclevel,enz,diffenz,
                      germenz)

for t in range(timesteps):
    if world.checkEnd() == True:
        print("Total Steps: %d" % world.steps)
        break
    else:
        world.step()
        

## initialize video writer
#fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
#fps = 10
#video_filename = 'output.avi'
#out = cv2.VideoWriter(video_filename, fourcc, fps, (world.X, world.Y))
#
## new frame after each addition of water
#for i in range(timesteps):
#    random_locations = np.random.random_integers(200,450, size=(200, 2))
#    for item in random_locations:
#        water_depth[item[0], item[1]] += 0.1
#        #add this array to the video
#        gray = cv2.normalize(water_depth, None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#        gray_3c = cv2.merge([gray, gray, gray])
#        out.write(gray_3c)
#
## close out the video writer
#out.release()        
#        
#        
#        
        
        
np.savetxt('/Users/eden/Desktop/IU_FALL_2021/RESEARCH/MODEL/DATA/MAIN/CAdata_11_12_21.csv', world.data, delimiter = ',', fmt='%s')
 


