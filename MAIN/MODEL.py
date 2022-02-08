#### INDIVIDUALITY MODEL ####
## Eden Forbes
date = "2_7/_22"
worldsave = "1"

#################
#### SCRIPTS ####
import OBJECTS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# import infotheory
 
##########################
#### DEFINE CONSTANTS ####
## Simulation Features
runs = 1 ## Number of simulations runs
timesteps = 1000 ## Length of the simulation
X = 30 ## X dimension of world
Y = 30 ## Y dimension of world
start_CA = 5 ## Number of initial unicellular CAs
CAfood = 15 ## Number of initial CA food sources
newfood = round(CAfood/5) ## How much food might be introduced in a step
foodchance = 0.03 ## Chance that the amount of new food is introduced in a step

## Cell Interaction Features
## Note "Proportional Strength" is not entirely correct, as they are 
## not yet exactly proportional
caf_scal = 2.0 ## Proportional strength of food chemical signals
adhesion_scal = 1.0 ## Proportional strength of adjacent agents' desires
agent_chemo_scal = 1.0 ## Proportional strength of agent chemical signal

## Display Simulation?
display = False

########################
#### RUN SIMULATION ####

CAtotal = list()
POPtotal = list()
for r in range(runs):
    ID = r
    #### DEFINE WORLD ####
    world = OBJECTS.World(ID,timesteps,X,Y,start_CA,CAfood,
                          newfood,foodchance,
                          caf_scal,adhesion_scal,agent_chemo_scal,
                          display)

    #### RUN WORLD ####
    for t in range(timesteps):
        if world.checkEnd() == True:
            for c in world.CAs:
                world.data.append(["cell", "world_end", c.ID, "gen", c.start, c.end, 
                                  c.lifespan, c.children])
            print("Total Steps: %d" % world.steps)
            break
        else:
            world.step()
    #### SAVE DATA ####
    CAdata = world.data
    for row in CAdata:
        row.insert(0,r)
        CAtotal.append(row)
    POPdata = world.run
    for row in POPdata:
        row.insert(0,r)
        POPtotal.append(row)
outputCA_string = '/Users/eden/Desktop/IU_SPRING_2022/RESEARCH/MODEL/DATA/{}_GEN_CAdata_{}.csv'.format(date,worldsave)
outputPOP_string = '/Users/eden/Desktop/IU_SPRING_2022/RESEARCH/MODEL/DATA/{}_GEN_POPdata_{}.csv'.format(date,worldsave)

np.savetxt(outputCA_string, CAtotal, delimiter = ',', fmt='%s')
np.savetxt(outputPOP_string, POPtotal, delimiter = ',', fmt='%s')

#### SAVE ANIMATION ####
fig = plt.figure() # make figure
# make axesimage object
# the vmin and vmax here are very important to get the color map correct
im = plt.imshow(world.arrays[0], cmap=plt.get_cmap('gist_earth'),
                vmin = 0, vmax = world.G0 + 9)
time_text = fig.text(0.45, 0.9, '')
# function to update figure
def updatefig(j):
    time = list(range(0,timesteps))
    # set the data in the axesimage object
    im.set_array(world.arrays[j])
    time_text.set_text('time = %.1f' % time[j])
    # return the artists set
    return [im]
# kick off the animation
ani = animation.FuncAnimation(fig, updatefig, frames=range(len(world.arrays)),
                              interval=50, blit=True)
ani.save('/Users/eden/Desktop/IU_SPRING_2022/RESEARCH/MODEL/DATA/{}_WORLD_VID_{}.mp4'.format(date,worldsave), 
         writer = 'ffmpeg', fps = 10)

#plt.show()


######################
#### ZOO CREATION ####

## First select start frame and area of interest
xmin = 5
xmax = 15
ymin = 5
ymax = 15
seed = world.arrays[999]
## Then collect cells from that frame and area
seedCAs = []
for cell in world.CAs:
    if cell.x >= xmin and cell.x <= xmax:
        if cell.y >= ymin and cell.y <= ymax:
            seedCAs.append(cell)
seed = OBJECTS.frameconvert(seed,xmin,xmax,ymin,ymax)

## Zoo simulation parameters
zoosave = 1

zooRuns = 1
zooTime = 1000
zoofood = 1
zoochance = 0.005
## Display Zoo?
zoodisplay = True

## MAKE ZOO

CAtotal_zoo = list()
POPtotal_zoo = list()
# Wants_zoo = list()
for r in range(zooRuns):
    ID = r
    #### DEFINE ZOO ####
    
    zoo = OBJECTS.Zoo(ID, seed, seedCAs, zooTime, xmax, xmin, ymax, ymin,
                 zoofood,zoochance,caf_scal,adhesion_scal, 
                 agent_chemo_scal, zoodisplay)

    #### RUN ZOO ####
    for t in range(zooTime):
        if zoo.checkEnd() == True:
            for c in zoo.CAs:
                zoo.data.append(["cell", "world_end", c.ID, "gen", c.start, c.end, 
                                  c.lifespan, c.children])
                zoo.wants.append([c.food_wants, c.friend_wants, 
                                  c.desire_wants, c.choices])
            print("Total Steps: %d" % zoo.steps)
            break
        else:
            zoo.step()
    #### SAVE DATA ####
    CAdata_zoo = zoo.data
    for row in CAdata_zoo:
        row.insert(0,r)
        CAtotal_zoo.append(row)
    POPdata_zoo = zoo.run
    for row in POPdata_zoo:
        row.insert(0,r)
        POPtotal_zoo.append(row)
    # Wantdata_zoo = zoo.wants
    # for row in Wantdata_zoo:
    #     row.insert(0,r)
    #     Wantdata_zoo.append(row)
outputCA_string_zoo = '/Users/eden/Desktop/IU_SPRING_2022/RESEARCH/MODEL/DATA/{}_ZOO_CAdata_{}.csv'.format(date,zoosave)
outputPOP_string_zoo = '/Users/eden/Desktop/IU_SPRING_2022/RESEARCH/MODEL/DATA/{}_ZOO_POPdata_{}.csv'.format(date,zoosave)
# outputWANT_string_zoo = '/Users/eden/Desktop/IU_SPRING_2022/RESEARCH/MODEL/DATA/{}_ZOO_WANTdata_{}.csv'.format(date,zoosave)

np.savetxt(outputCA_string_zoo, CAtotal_zoo, delimiter = ',', fmt='%s')
np.savetxt(outputPOP_string_zoo, POPtotal_zoo, delimiter = ',', fmt='%s')
# np.savetxt(outputWANT_string_zoo, Wants_zoo, delimiter = ',', fmt='%s')

#### SAVE ANIMATION ####
fig_zoo = plt.figure() # make figure
# make axesimage object
# the vmin and vmax here are very important to get the color map correct
im_zoo = plt.imshow(zoo.arrays[0], cmap=plt.get_cmap('gist_earth'),
                    vmin = 0, vmax = zoo.G0 + 9)
zoo_time_text = fig_zoo.text(0.45, 0.9, '')
# function to update figure
def updatefig_zoo(j):
    time = list(range(0,zooTime))
    # set the data in the axesimage object
    im_zoo.set_array(zoo.arrays[j])
    zoo_time_text.set_text('time = %.1f' % time[j])
    # return the artists set
    return [im_zoo]
# kick off the animation
ani_zoo = animation.FuncAnimation(fig_zoo, updatefig_zoo, frames=range(len(zoo.arrays)),
                              interval=50, blit=True)
ani_zoo.save('/Users/eden/Desktop/IU_SPRING_2022/RESEARCH/MODEL/DATA/{}_ZOO_VID_{}.mp4'.format(date,zoosave), 
         writer = 'ffmpeg', fps = 10)

#plt.show()



























# ############################
# #### DATA RESTRUCTURING ####

# arrays = np.array(world.arrays).astype(int)
# conc_arrays = []
# cells = []
# rest = []

# for i in range(world.timesteps):
#     w = np.concatenate(arrays[i])
#     conc_arrays.append(w)
# conc_arrays_final = np.concatenate(conc_arrays)

# for i in range(world.timesteps):
#     x = []
#     y = []
#     for j in range(len(conc_arrays[i])):
#         if conc_arrays[i][j] == world.G0 + 9:
#             x.append(1)
#             y.append(0)
#         else:
#             x.append(0)
#             y.append(1)
#     cells.append(x)
#     rest.append(y)
# cells = np.concatenate(cells)
# rest = np.concatenate(rest)

# data = [conc_arrays_final,cells,rest]

# #### INFORMATION THEORY ANALYSIS ####

# ## Setup
# dims = 3  # total dimensionality of all variables
# nreps = 1 # number of shifted binnings over which data is binned and averaged
# nbins = [10]*dims # number of bins along each dimension of the data
# mins = [0]*dims # min value or left edge of binning for each dimension
# maxs = [1]*dims # max value or right edge of binning for each dimension

# ## Creating object
# it = infotheory.InfoTools(dims, nreps)

# ## Specify binning
# it.set_equal_interval_binning(nbins, mins, maxs)

# ## Adding data - concatenate data from all vars

# for i in range(len(conc_arrays_final)):
#     it.add_data_point([conc_arrays_final[i],cells[i],rest[i]])

# varIDs = [0,1,1]

# mi = it.mutual_info(varIDs)
# mi /= np.log2(np.min(nbins))
# print('Mutual information = {}'.format(mi))

# # ent = it.entropy(varIDs)
# # ent /= np.log2(np.min(nbins))
# # print('Entropy = {}'.format(ent))


# ### BELOW DON'T WORK

# # ri = it.redundant_info(varIDs)
# # ri /= np.log2(np.min(nbins))
# # print('Redundant information = {}'.format(ri))

# # ui = it.unique_info(varIDs)
# # ui /= np.log2(np.min(nbins))
# # print('Unique information = {}'.format(ui))

# # syn = it.synergy(varIDs)
# # syn /= np.log2(np.min(nbins))
# # print('Synergy = {}'.format(syn))



















