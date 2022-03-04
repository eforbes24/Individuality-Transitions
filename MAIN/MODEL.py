#### INDIVIDUALITY MODEL ####
## Eden Forbes
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import OBJECTS
config = "[FIXED_PARAMS_115]"

#################
#### SCRIPTS ####

##########################
#### DEFINE CONSTANTS ####
## Simulation Features
runs = 1  # Number of simulations runs
timesteps = 500 # Length of the simulation
X = 50  # X dimension of world
Y = 50  # Y dimension of world
start_CA = 15  # Number of initial unicellular CAs
CAfood = 25  # Number of initial CA food sources
newfood = round(CAfood/10)  # How much food might be introduced in a step
foodchance = 0.02  # Chance that the amount of new food is introduced in a step

## Cell Interaction Features
caf_scal = 1.0  # Proportional strength of food chemical signals
adhesion_scal = 1.0  # Proportional strength of adjacent agents' desires
agent_chemo_scal = 5.0  # Proportional strength of agent chemical signal

## Display Simulation?
display = True

########################
#### RUN SIMULATION ####

CAtotal = list()
POPtotal = list()
inittotal = list()
for r in range(runs):
    ID = r
    #### DEFINE WORLD ####
    world = OBJECTS.World(ID, timesteps, X, Y, start_CA, CAfood,
                          newfood, foodchance,
                          caf_scal, adhesion_scal, agent_chemo_scal,
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
            print(world.steps)
            world.step()

    #### SAVE DATA ####
    CAdata = world.data
    for row in CAdata:
        row.insert(0, r)
        CAtotal.append(row)
    POPdata = world.run
    for row in POPdata:
        row.insert(0, r)
        POPtotal.append(row)
    initdata = world.fills
    for frame in initdata:
        for cell in frame:
            inittotal.append(cell)
outputCA_string = '/Users/eden/Desktop/RESEARCH/MODEL/DATA/MAIN/{}_GEN_CAdata.csv'.format(
    config)
outputPOP_string = '/Users/eden/Desktop/RESEARCH/MODEL/DATA/MAIN/{}_GEN_POPdata.csv'.format(
    config)
outputArray_string = '/Users/eden/Desktop/RESEARCH/MODEL/DATA/MAIN/{}_ARRAYdata.csv'.format(
    config)
outputInits_string = '/Users/eden/Desktop/RESEARCH/MODEL/DATA/MAIN/{}_INITSdata.csv'.format(
    config)

np.savetxt(outputCA_string, CAtotal, delimiter=',', fmt='%s')
np.savetxt(outputPOP_string, POPtotal, delimiter=',', fmt='%s')
np.savetxt(outputInits_string, inittotal, delimiter = ',', fmt='%s')
arrays_out = np.asarray(world.arrays)
print(arrays_out.shape)
arrays_out = arrays_out.reshape(arrays_out.shape[0],-1)
np.savetxt(outputArray_string, arrays_out, delimiter=',', fmt='%d')

#### SAVE ANIMATION ####
fig = plt.figure()
im = plt.imshow(world.arrays[0], cmap=plt.get_cmap('gist_earth'),
                vmin=0, vmax=world.G0 + 9)
time_text = fig.text(0.45, 0.9, '')


def updatefig(j):
    time = list(range(0, timesteps))
    im.set_array(world.arrays[j])
    time_text.set_text('time = %.1f' % (time[j]-1))
    return [im]


ani = animation.FuncAnimation(fig, updatefig, frames=range(len(world.arrays)),
                              interval=50, blit=True)
ani.save('/Users/eden/Desktop/RESEARCH/MODEL/DATA/VIDEOS/{}_WORLD_VID.mp4'.format(config),
         writer='ffmpeg', dpi = 200, fps=10)
#plt.show()



















# #### INDIVIDUALITY MODEL ####
# ## Eden Forbes
# import matplotlib.animation as animation
# import matplotlib.pyplot as plt
# import numpy as np
# import statistics 
# import OBJECTS
# import MODEL

# ######################
# #### ZOO CREATION ####
# trial = "worldslice"
# zooconfig = "[test]"
# zooconfig2 = "{}{}".format(zooconfig,trial)
# loadstring = '/Users/eden/Desktop/RESEARCH/MODEL/DATA/MAIN/{}_ARRAYdata.csv'.format(zooconfig)
# initstring = '/Users/eden/Desktop/RESEARCH/MODEL/DATA/MAIN/{}_INITSdata.csv'.format(zooconfig)

# zooworld = np.loadtxt(loadstring, dtype = float, delimiter = ',')
# zooworld = zooworld.reshape(-1,MODEL.X,MODEL.Y)
# zooinits = np.loadtxt(initstring, dtype = float, delimiter = ',')

# ## First select start frame and area of interest
# #### AXES ARE BACKWARDS: Y IS X, X IS Y
# frame = 80
# xmin = 0
# xmax = 50
# ymin = 0
# ymax = 50
# seed = zooworld[frame]

# ## Then collect cells from that frame and area
# seedCAs = []
# for row in zooinits:
#     if row[3] == frame:
#         if row[1] >= xmin and row[1] <= xmax:
#             if row[2] >= ymin and row[2] <= ymax:
#                 cell = OBJECTS.CA(MODEL.world)
#                 cell.fill = row[0]
#                 cell.x = int(row[1])
#                 cell.y = int(row[2])
#                 seedCAs.append(cell)

# # seedCAs = []
# # for cell in world.CA_history[frame]:
# #     if cell.x >= xmin and cell.x <= xmax:
# #         if cell.y >= ymin and cell.y <= ymax:
# #             seedCAs.append(cell)
# seed = OBJECTS.frameconvert(seed,xmin,xmax,ymin,ymax)

# ## Zoo simulation parameters
# zooRuns = 1
# zooTime = 50
# zoofood = 1
# zoochance = 0.005

# caf_scal = 0.0
# adhesion_scal = 0.0
# agent_chemo_scal = 0.0


# ## Display Zoo?
# zoodisplay = False

# metabolism = False
# foodintro = False

# ## MAKE ZOO

# CAtotal_zoo = list()
# POPtotal_zoo = list()
# # Wants_zoo = list()
# for r in range(zooRuns):
#     ID = r
#     #### DEFINE ZOO ####
#     zoo = OBJECTS.Zoo(ID, seed, seedCAs, zooTime, xmax, xmin, ymax, ymin,
#                  metabolism,zoofood,foodintro,zoochance,caf_scal,adhesion_scal, 
#                  agent_chemo_scal, zoodisplay)

#     #### RUN ZOO ####
#     for t in range(zooTime):
#         if zoo.checkEnd() == True:
#             for c in zoo.CAs:
#                 zoo.data.append(["cell", "world_end", c.ID, "gen", c.start, c.end, 
#                                   c.lifespan, c.children])
#                 zoo.wants.append([c.food_wants, c.friend_wants, 
#                                   c.desire_wants, c.choices])
#             print("Total Steps: %d" % zoo.steps)
#             break
#         else:
#             zoo.step()
            
#     #### SAVE DATA ####
#     CAdata_zoo = zoo.data
#     for row in CAdata_zoo:
#         row.insert(0,r)
#         CAtotal_zoo.append(row)
#     POPdata_zoo = zoo.run
#     for row in POPdata_zoo:
#         row.insert(0,r)
#         POPtotal_zoo.append(row)
#     # Wantdata_zoo = zoo.wants
#     # for row in Wantdata_zoo:
#     #     row.insert(0,r)
#     #     Wantdata_zoo.append(row)
# outputCA_string_zoo = '/Users/eden/Desktop/RESEARCH/MODEL/DATA/MAIN/ZOO/{}_ZOO_CAdata.csv'.format(zooconfig2)
# outputPOP_string_zoo = '/Users/eden/Desktop/RESEARCH/MODEL/DATA/MAIN/ZOO/{}_ZOO_POPdata.csv'.format(zooconfig2)
# # outputWANT_string_zoo = '/Users/eden/Desktop/RESEARCH/MODEL/DATA/MAIN/ZOO/{}_ZOO_WANTdata.csv'.format(date2)

# np.savetxt(outputCA_string_zoo, CAtotal_zoo, delimiter = ',', fmt='%s')
# np.savetxt(outputPOP_string_zoo, POPtotal_zoo, delimiter = ',', fmt='%s')
# # np.savetxt(outputWANT_string_zoo, Wants_zoo, delimiter = ',', fmt='%s')

# #### SAVE ZOO ANIMATION ####
# fig_zoo = plt.figure()
# im_zoo = plt.imshow(zoo.arrays[0], cmap=plt.get_cmap('gist_earth'),
#                     vmin = 0, vmax = zoo.G0 + 9)
# zoo_time_text = fig_zoo.text(0.45, 0.9, '')

# def updatefig_zoo(j):
#     time = list(range(0,zooTime))
#     im_zoo.set_array(zoo.arrays[j])
#     zoo_time_text.set_text('time = %.1f' % time[j])
#     return [im_zoo]

# ani_zoo = animation.FuncAnimation(fig_zoo, updatefig_zoo, frames=range(len(zoo.arrays)),
#                               interval=50, blit=True)
# ani_zoo.save('/Users/eden/Desktop/RESEARCH/MODEL/DATA/VIDEOS/MENAGERIE/{}_ZOO_VID.mp4'.format(zooconfig2), 
#          writer = 'ffmpeg', fps = 10)

# #plt.show()

# #################################
# #### PERSISTENCE & PROXIMITY ####

# def metric_1(t1, t2, zoo):
#     time1 = t1
#     time2 = t2
#     time = list(range(time1,time2+1))
#     distset = list()
#     for i in time:
#         dist = statistics.mean(map(float,zoo.dists[i]))
#         distset.append(dist)
#         lil_metric_1 = statistics.mean(map(float,distset))
#     return(lil_metric_1)

# def metric_2(t1, t2, zoo):
#     distchange = list()
#     tm = t2 - t1
#     for i in range(tm): 
#         pt1 = zoo.dists[i]
#         pt2 = zoo.dists[i+1]
#         array1 = np.array(pt1)
#         array2 = np.array(pt2)
#         subtracted_array = np.subtract(array1, array2)
#         subtracted = list(subtracted_array)
#         change = [abs(ele) for ele in subtracted]
#         changemean = statistics.mean(map(float, change))
#         distchange.append(changemean)
#         lil_metric_2 = statistics.mean(map(float,distchange))
#     return(lil_metric_2)

# ####################
# #### WORM PLOTS ####

# # Select Data
# zoo_input = zoo.arrays[0:99]
# zoo_output = []

# for array in zoo_input:
#     blank = []
#     for row in array:
#         blank2 = []
#         for item in row:
#             if item < zoo.G0 + 9:
#                 item = 0
#             blank2.append(item)
#         blank2 = np.asarray(blank2)
#         blank.append(blank2)
#     blank = np.asarray(blank)
#     zoo_output.append(blank)
# zoo_output = np.asarray(zoo_output)

# # Create Axes
# axes = [(xmax-xmin),(ymax-ymin),zooTime]

# # Plot 
# fig_worm = plt.figure()
# ax = fig_worm.add_subplot(111, projection='3d')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Time')
# ax.voxels(zoo_output, facecolors='red', edgecolors='gray')

# # Make Animations
# def animate_rotate(i):
#     ax.view_init(elev=10., azim=i)
#     return fig_worm,

# anim = animation.FuncAnimation(fig_worm, animate_rotate, frames=360, interval=20, blit=True)

# anim.save('/Users/eden/Desktop/RESEARCH/MODEL/DATA/VIDEOS/MENAGERIE/PLOTS/{}_animation.mp4'.format(zooconfig2),
#           fps=30, extra_args=['-vcodec', 'libx264'])

    
    
    





