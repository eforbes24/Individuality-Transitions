#### INDIVIDUALITY MODEL ####
## Eden Forbes
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import OBJECTS
config = "[NAME]"

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
agent_chemo_scal = 1.0  # Proportional strength of agent chemical signal

## Display Simulation?
display = False

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
outputCA_string = '{}_GEN_CAdata.csv'.format(
    config)
outputPOP_string = '{}_GEN_POPdata.csv'.format(
    config)
outputArray_string = '{}_ARRAYdata.csv'.format(
    config)
outputInits_string = '{}_INITSdata.csv'.format(
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
ani.save('{}_WORLD_VID.mp4'.format(config),
         writer='ffmpeg', dpi = 200, fps=10)
#plt.show()

