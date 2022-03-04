#### INDIVIDUALITY ZOO ####
## Eden Forbes
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import statistics 
import OBJECTS
import MODEL

# 2: "frm112:x2332:y312:2c:fNmN" ## 1
# 3: "frm340:x3342:y2332:3c:fNmN" ## 4
# 4: "frm175:x2433:y1019:4c:fNmN" ## 6
# 5: "frm191:x2332:y817:5c:fNmN" ## 6
# 6: "frm223:x2534:y716:6c:fNmN" ## 9
# 7: "frm232:x3847:y2837:7c:fNmN" ## 5
# 8: "frm225:x1019:y1322:8c:fNmN" ## (1)
# 9: "frm215:x918:y1322:9c:fNmN" ## 4

######################
#### ZOO CREATION ####
trial = "frm215:x918:y1322:9c:fYmY"
zooconfig = "[1,1,1]"
zooconfig2 = "{}{}".format(zooconfig,trial)
loadstring = '/Users/eden/Desktop/RESEARCH/MODEL/DATA/MAIN/{}_ARRAYdata.csv'.format(zooconfig)
initstring = '/Users/eden/Desktop/RESEARCH/MODEL/DATA/MAIN/{}_INITSdata.csv'.format(zooconfig)

zooworld = np.loadtxt(loadstring, dtype = float, delimiter = ',')
zooworld = zooworld.reshape(-1,MODEL.X,MODEL.Y)
zooinits = np.loadtxt(initstring, dtype = float, delimiter = ',')

## First select start frame and area of interest
#### AXES ARE BACKWARDS: Y IS X, X IS Y
frame = 215
xmin = 9
xmax = 18
ymin = 13
ymax = 22
seed = zooworld[frame]

## Zoo simulation parameters
zooRuns = 1
zooTime = 300

foodintro = True
metabolism = True
zoofood = 1
zoochance = 0.005

## Match to simulation being called
caf_scal = 1.0
adhesion_scal = 1.0
agent_chemo_scal = 1.0

## Display Zoo?
zoodisplay = False

## Collect cells from given frame and area
seedCAs = []
for row in zooinits:
    if row[3] == frame:
        if row[1] >= xmin and row[1] <= xmax:
            if row[2] >= ymin and row[2] <= ymax:
                cell = OBJECTS.CA(MODEL.world)
                cell.fill = row[0]
                cell.x = int(row[1])
                cell.y = int(row[2])
                seedCAs.append(cell)

# seedCAs = []
# for cell in world.CA_history[frame]:
#     if cell.x >= xmin and cell.x <= xmax:
#         if cell.y >= ymin and cell.y <= ymax:
#             seedCAs.append(cell)
seed = OBJECTS.frameconvert(seed,xmin,xmax,ymin,ymax)

## MAKE ZOO

CAtotal_zoo = list()
POPtotal_zoo = list()
FILLtotal_zoo = list()
# Wants_zoo = list()
for r in range(zooRuns):
    ID = r
    #### DEFINE ZOO ####
    zoo = OBJECTS.Zoo(ID, seed, seedCAs, zooTime, xmax, xmin, ymax, ymin,
                 metabolism,zoofood,foodintro,zoochance,caf_scal,adhesion_scal, 
                 agent_chemo_scal, zoodisplay)

    #### RUN ZOO ####
    for t in range(zooTime):
        if zoo.checkEnd() == True:
            for c in zoo.CAs:
                zoo.data.append(["cell", "world_end", c.ID, "gen", c.start, c.end, 
                                  c.lifespan, c.children, c.filllist])
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
    for cell in zoo.CAs:
        FILLtotal_zoo.append(cell.filllist)
        
outputCA_string_zoo = '/Users/eden/Desktop/RESEARCH/MODEL/DATA/MAIN/ZOO/{}_ZOO_CAdata.csv'.format(zooconfig2)
outputPOP_string_zoo = '/Users/eden/Desktop/RESEARCH/MODEL/DATA/MAIN/ZOO/{}_ZOO_POPdata.csv'.format(zooconfig2)
outputFILL_string_zoo = '/Users/eden/Desktop/RESEARCH/MODEL/DATA/MAIN/ZOO/{}_ZOO_FILLdata.csv'.format(zooconfig2)
outputDIST_string_zoo = '/Users/eden/Desktop/RESEARCH/MODEL/DATA/MAIN/ZOO/{}_ZOO_DISTdata.csv'.format(zooconfig2)
outputSTEPDIST_string_zoo = '/Users/eden/Desktop/RESEARCH/MODEL/DATA/MAIN/ZOO/{}_ZOO_STEPDISTdata.csv'.format(zooconfig2)

np.savetxt(outputCA_string_zoo, CAtotal_zoo, delimiter = ',', fmt='%s')
np.savetxt(outputPOP_string_zoo, POPtotal_zoo, delimiter = ',', fmt='%s')
np.savetxt(outputFILL_string_zoo, FILLtotal_zoo, delimiter = ',', fmt='%s')
blank = [0,0,0,0]
zoo.dists.insert(0,blank)
np.savetxt(outputDIST_string_zoo, zoo.dists, delimiter = ',', fmt='%s')
np.savetxt(outputSTEPDIST_string_zoo, zoo.stepdists, delimiter = ',', fmt='%s')

#### SAVE ZOO ANIMATION ####
fig_zoo = plt.figure()
im_zoo = plt.imshow(zoo.arrays[0], cmap=plt.get_cmap('gist_earth'),
                    vmin = 0, vmax = zoo.G0 + 9)
zoo_time_text = fig_zoo.text(0.45, 0.9, '')

def updatefig_zoo(j):
    time = list(range(0,zooTime))
    im_zoo.set_array(zoo.arrays[j])
    zoo_time_text.set_text('time = %.1f' % time[j])
    return [im_zoo]

ani_zoo = animation.FuncAnimation(fig_zoo, updatefig_zoo, frames=range(len(zoo.arrays)),
                              interval=50, blit=True)
ani_zoo.save('/Users/eden/Desktop/RESEARCH/MODEL/DATA/VIDEOS/MENAGERIE/{}_ZOO_VID.mp4'.format(zooconfig2), 
          writer = 'ffmpeg', fps = 10)

#plt.show()

#################################
#### PERSISTENCE & PROXIMITY ####

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

####################
#### WORM PLOTS ####

# Select Data
zoo_input = zoo.arrays[0:99]
zoo_output = []

for array in zoo_input:
    blank = []
    for row in array:
        blank2 = []
        for item in row:
            if item < zoo.G0 + 9:
                item = 0
            blank2.append(item)
        blank2 = np.asarray(blank2)
        blank.append(blank2)
    blank = np.asarray(blank)
    zoo_output.append(blank)
zoo_output = np.asarray(zoo_output)

# Create Axes
axes = [(xmax-xmin),(ymax-ymin),zooTime]

# Plot 
fig_worm = plt.figure()
ax = fig_worm.add_subplot(111, projection='3d')
ax.set_xlabel('Time')
ax.set_ylabel('X')
ax.set_zlabel('Y')
ax.voxels(zoo_output, facecolors='green', edgecolors='gray')

# Make Animations
def animate_rotate(i):
    ax.view_init(elev=10., azim=i)
    return fig_worm,

anim = animation.FuncAnimation(fig_worm, animate_rotate, frames=360, interval=20, blit=True)

anim.save('/Users/eden/Desktop/RESEARCH/MODEL/DATA/VIDEOS/MENAGERIE/PLOTS/{}_animation.mp4'.format(zooconfig2),
          fps=30, extra_args=['-vcodec', 'libx264'])

    
###################
#### ECO PLOTS ####


# distlist = []
# steplist = []
# count = 0
# for d in zoo.dists:
#     for i in d:
#         distlist.append(i)
#         steplist.append(count)
#     count = count + 1
    

# plt.scatter(steplist,distlist)
# plt.show










    
