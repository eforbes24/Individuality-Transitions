#### AUTOMATA MODEL ####
## Eden Forbes
## Function Definitions

#### LIBRARIES
import numpy as np
from matplotlib import pyplot
import random
import OBJECTS

#### ACCESS FUNCTIONS
def getFeat(feat, world):
    featlist = list()
    for i in range(len(world)):
        for j in range(len(world)):
            posx = i
            posy = j
            pos = world[i,j]
            if pos.feat == True:
                featlist.append((posx,posy))
    return featlist



foodlist = getFeat(food, world)





def getNeighbors(world, posx, posy, X, Y):
    P1 = [posx,posy+1]
    P2 = [posx+1,posy]
    P3 = [posx,posy-1]
    P4 = [posx-1,posy]
    neighbors = [P1,P2,P3,P4]
    neighbor_values = np.zeros(4)
    for i in range(4):
        neighbor = neighbors[i]
        if neighbor[0] < 0 or neighbor[1] < 0 or neighbor[0] > X-1 or neighbor[1] > Y-1:
            neighbor_values[i] = None
        else:
            neighbor_values[i] = world[neighbor[0],neighbor[1]]
    return(neighbor_values)
        
#### WORLD GENERATION
def generateWorld(X, Y, start_CA, CAfood, INDfood):
    dim = np.asarray([[0] * X for _ in range(Y)])
    dim = dim.shape
    world = list()
    for i in range(X*Y):
        world.append(OBJECTS.tile())
    CAinds = np.random.choice(len(world), size=start_CA)
    for i in CAinds:
        tile = world[i]
        tile.CA = True
    CAFinds = np.random.choice(len(world), size=CAfood)
    for i in CAFinds:
        tile = world[i]
        tile.CAfood = True
    INDFinds = np.random.choice(len(world), size=INDfood)
    for i in INDFinds:
        tile = world[i]
        tile.INDfood = True   
    world = np.asarray(world)
    world = world.reshape(dim)
   # world = replaceRandom(world, start_CA, CAfood, INDfood)
    return world

def generateCAs(world):
    CAlist = getFeat(world, 'CA')
    for i in CAlist:
        x = i[0]
        y = i[1]
        CA = OBJECTS.CA()
        CA.__init__(x,y)
    return CAlist
    
### SO BROKEN NOW
def showWorld(world):
    # make a color map of fixed colors
    #cmap = mpl.colors.ListedColormap(['white','red','green','blue'])
    #bounds=[0,0.5,1,1.5,2,2.5,3]
    #norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    # tell imshow about color map so that only set colors are used
    pyplot.figure(figsize = (5,5))
    pyplot.imshow(world,interpolation='nearest')#,
                    #cmap = cmap,norm=norm)
    pyplot.show()

#### STEP FUNCTIONS
def updateGradients(world, feat, g1, g2, g3):
    feat = np.where(world[0][0].CAfood == True)
    featx = list()
    featy = list()
    for i in feat:
        featx.append(i[0])
        featy.append(i[1])
    for i in range(len(featx)):
        posx = featx[i]
        posy = featy[i]
        ## First degree
        world[posx+1,posy] = world[posx+1,posy] + g1
        world[posx,posy+1] = world[posx,posy+1] + g1
        world[posx-1,posy] = world[posx-1,posy] + g1
        world[posx,posy-1] = world[posx,posy-1] + g1
        ## Second degree
        world[posx+1,posy+1] = world[posx+1,posy+1] + g2
        world[posx-1,posy+1] = world[posx-1,posy+1] + g2
        world[posx-1,posy-1] = world[posx-1,posy-1] + g2
        world[posx+1,posy-1] = world[posx+1,posy-1] + g2
        world[posx+2,posy] = world[posx+2,posy] + g2
        world[posx,posy+2] = world[posx,posy+2] + g2
        world[posx-2,posy] = world[posx-2,posy] + g2
        world[posx,posy-2] = world[posx,posy-2] + g2
        ## Third degree
        world[posx+2,posy+1] = world[posx+2,posy+1] + g3
        world[posx-2,posy+1] = world[posx-2,posy+1] + g3
        world[posx-2,posy-1] = world[posx-2,posy-1] + g3
        world[posx+2,posy-1] = world[posx+2,posy-1] + g3
        world[posx+1,posy+2] = world[posx+1,posy+2] + g3
        world[posx-1,posy+2] = world[posx-1,posy+2] + g3
        world[posx-1,posy-2] = world[posx-1,posy-2] + g3
        world[posx+1,posy-2] = world[posx+1,posy-2] + g3
        world[posx+3,posy] = world[posx+3,posy] + g3
        world[posx,posy+3] = world[posx,posy+3] + g3
        world[posx-3,posy] = world[posx-3,posy] + g3
        world[posx,posy-3] = world[posx,posy-3] + g3
    return(world)

def move(neighbors, world, posx, posy):
    options = np.where(neighbors == 0)
    choice = random.choice(options[0])
    if choice == 0:
        world[posx,posy+1] = 1
        world[posx,posy] = 0
    elif choice == 1:
        world[posx+1,posy] = 1
        world[posx,posy] = 0
    elif choice == 2:
        world[posx,posy-1] = 1
        world[posx,posy] = 0
    else:
        world[posx-1,posy] = 1
        world[posx,posy] = 0
    return(world)

def eat(neighbors, world, posx, posy):
    if (2 not in neighbors):
        return(world)
    else:
        food = np.where(neighbors == 2)
        food = food[0]
        for i in range(len(food)):
            if food[i] == 0:
                world[posx,posy+1] = 0
            elif food[i] == 1:
                world[posx+1,posy] = 0
            elif food[i] == 2:
                world[posx,posy-1] = 0
            else:
                world[posx-1,posy] = 0
        return(world)

def step(world, X, Y, g1, g2, g3):
    CAs = np.where(world == 1)
#    world = updateGradients(world,g1,g2,g3)
#    world[CAs] = 1
    CAs_x = CAs[0]
    CAs_y = CAs[1]
    for i in range(len(CAs_x)):
        posx = CAs_x[i]
        posy = CAs_y[i]
        ## CAs check what's around them
        neighbors = getNeighbors(world, posx, posy, X, Y)
        ## CAs eat food around them
        world = eat(neighbors, world, posx, posy)
        ## CAs move
        world = move(neighbors, world, posx, posy)
    return(world)


#### OLD SHIT
    
#def replaceRandom(world, start_CA, CAfood, INDfood):
#    temp = np.asarray(world)   # Cast to numpy array
#    shape = temp.shape       # Store original shape
#    temp = temp.flatten()    # Flatten to 1D
#    inds = np.random.choice(temp.size, size=start_CA)   # Get random indices
#    temp[inds] = 1      # Fill with something
#    inds = np.random.choice(temp.size, size=CAfood)
#    temp[inds] = 2
#    inds = np.random.choice(temp.size, size=INDfood)
#    temp[inds] = 3
#    temp = temp.reshape(shape)                     # Restore original shape
#    return temp

