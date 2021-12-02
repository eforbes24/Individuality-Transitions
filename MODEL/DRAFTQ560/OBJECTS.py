#### AUTOMATA MODEL ####
## Eden Forbes
## Object Class Definitions

#### LIBRARIES
import numpy as np
import random
from matplotlib import pyplot

#### FUNCTIONS
## WRAPPER


#### OBJECTS
## TILE
class Tile:
    CA = False
    CAfood = False
    CAgrad = 0.0
    CAFgrad = 0.0
    foodtough = 0.0
    ftcap = 0.0
    x = 0
    y = 0
    
class World:
    def __init__(self, timesteps, X, Y, start_CA, CAfood, foodregenz,
                 g1, g2, g3, g4, foodtoughlow, foodtoughhigh, 
                 foodvalue, foodloss, newfood, foodchance):
        ## Give world attributes
        self.timesteps = timesteps
        self.X = X
        self.Y = Y
        self.foodvalue = foodvalue
        self.foodloss = foodloss
        self.newfood = newfood
        self.foodchance = foodchance
        self.steps = 0
        self.CAs = list()
        ## Gradient & Enzyme attributes
        self.G1 = g1
        self.G2 = g2
        self.G3 = g3
        self.G4 = g4
        self.foodregenz = foodregenz
        self.ftl = foodtoughlow
        self.fth = foodtoughhigh
        ## Define world dimensions
        dim = np.asarray([[0] * X for _ in range(Y)])
        dim = dim.shape
        self.dim = dim
        self.length = X*Y
        ## Populate world with tiles
        self.tiles = []
        for i in range(self.length):
            space = Tile()
            space.x = i // X
            space.y = i % X
            self.tiles.append(space)
        CAFinds = np.random.choice(self.length, size=CAfood)
        for i in CAFinds:
            til = self.tiles[i]
            til.CAfood = True
            til.foodtough = random.uniform(self.ftl, self.fth)
            til.ftcap = til.foodtough
        self.tiles = np.asarray(self.tiles)
        self.tiles = self.tiles.reshape(self.dim)
        ## Make CAs
        for i in range(start_CA):
            cell = CA(self)
            self.CAs.append(cell)
        ## Make Data Output Storage
        self.data = list()
    
    def __str__(self):
        return 'CA List: %s' % (self.CAs)
    
    def wrapper(self, coordlist):
        xlim = self.X - 1
        ylim = self.Y - 1
        finallist = []
        for i in range(len(coordlist)):
            x = coordlist[i][0]
            y = coordlist[i][1]
            if x > xlim:
                x = x - self.X
            else:
                x = x
            if x < 0:
                x = x + self.X
            else:
                x = x
            if y > ylim:
                y = y - self.Y
            else:
                y = y
            if y < 0:
                y = y + self.Y
            else:
                y = y
            finallist.append((x,y))
        return finallist
    
    ## SHOW WORLD
    def showWorld(self):
        array = np.zeros((self.X, self.Y))
        tilearray = self.tiles.flatten()
        for til in tilearray:
            locx = til.x
            locy = til.y
            if til.CA == True:
                array[locx,locy] = 2
            elif til.CAfood == True:
                array[locx,locy] = 1
            else:
                array[locx,locy] = 0
        pyplot.imshow(array, interpolation='nearest')
        pyplot.show()
    
    ## GRADIENT FUNCTIONS
    def firstDegree(self, x, y):
        coordlist = [(x+1,y), (x,y+1), (x-1,y), (x,y-1)]
        coordlist = self.wrapper(coordlist)
        fList = []
        for i in coordlist:
            fList.append(self.tiles[i[0]][i[1]])
        return fList
    
    def secondDegree(self, x, y):
        coordlist = [(x+1,y+1), (x-1,y+1), (x-1,y-1), (x+1,y-1),
                     (x+2,y), (x,y+2), (x-2,y), (x,y-2)]
        coordlist = self.wrapper(coordlist)
        sList = []
        for i in coordlist:
            sList.append(self.tiles[i[0]][i[1]])
        return sList
    
    def thirdDegree(self, x, y):
        coordlist = [(x+2,y+1), (x-2,y+1), (x-2,y-1), (x+2,y-1),
                     (x+1,y+2), (x-1,y+2), (x-1,y-2), (x+1,y-2),
                     (x+3,y), (x,y+3), (x-3,y), (x,y-3)]
        coordlist = self.wrapper(coordlist)
        tList = []
        for i in coordlist:
            tList.append(self.tiles[i[0]][i[1]])
        return tList
    
    def fourthDegree(self, x, y):
        coordlist = [(x+3,y+1), (x-3,y+1), (x-3,y-1), (x+3,y-1),
                     (x+1,y+3), (x-1,y+3), (x-1,y-3), (x+1,y-3),
                     (x+2,y+2), (x-2,y+2), (x-2,y-2), (x+2,y-2),
                     (x+4,y), (x,y+4), (x-4,y), (x,y-4)]
        coordlist = self.wrapper(coordlist)
        fList = []
        for i in coordlist:
            fList.append(self.tiles[i[0]][i[1]])
        return fList
        
    
    ## UPDATE TILES
    def updateCAGrad(self):
        for row in self.tiles:
            for tile in row:
                tile.CAgrad = 0
        CAx = []
        CAy = []
        for cell in self.CAs:
            CAx.append(cell.x)
            CAy.append(cell.y)
        for i in range(len(CAx)):
            posx = CAx[i]
            posy = CAy[i]
            ## First Degree
            fList = self.firstDegree(posx,posy)
            for tile in fList:
                tile.CAgrad = tile.CAgrad + self.G1
            ## Second Degree
            sList = self.secondDegree(posx,posy)
            for tile in sList:
                tile.CAgrad = tile.CAgrad + self.G2
            ## Third Degree
            tList = self.thirdDegree(posx,posy)
            for tile in tList:
                tile.CAgrad = tile.CAgrad + self.G3
            fList = self.fourthDegree(posx,posy)
            for tile in fList:
                tile.CAgrad = tile.CAgrad + self.G4
                
    def updateCAFGrad(self):
        for row in self.tiles:
            for tile in row:
                tile.CAFgrad = 0
        for i in range(self.X - 1):
            for j in range(self.Y - 1):
                tile = self.tiles[i][j]
                posx = tile.x
                posy = tile.y
                if tile.CAfood == True:
                    fList = self.firstDegree(posx,posy)
                    for tile in fList:
                        tile.CAFgrad = tile.CAFgrad + self.G1
                    sList = self.secondDegree(posx,posy)
                    for tile in sList:
                        tile.CAFgrad = tile.CAFgrad + self.G2
                    tList = self.thirdDegree(posx,posy)
                    for tile in tList:
                        tile.CAFgrad = tile.CAFgrad + self.G3
                    fList = self.fourthDegree(posx,posy)
                    for tile in fList:
                        tile.CAFgrad = tile.CAFgrad + self.G4
                else:
                    tile.CAfood = False
                
    def updateCA(self):
        for row in self.tiles:
            for tile in row:
                tile.CA = False
        for cell in self.CAs:
            posx = cell.x
            posy = cell.y
            self.tiles[posx][posy].CA = True
            
    def updateCAFT(self):
        for row in self.tiles:
            for tile in row:
                if tile.CAfood == True:
                    tile.foodtough = tile.foodtough + self.foodregenz
                    if tile.foodtough > tile.ftcap:
                        tile.foodtough = tile.ftcap
                    else: 
                        tile.foodtough = tile.foodtough
                else:
                    tile.CAfood = False
                    
    ## CHECK FUNCTIONS             
    def checkRepro(self):
        for cell in self.CAs:
            if cell.fill >= 1:
                cell.fill = cell.fill/2
                cell.children = cell.children + 1
                loc = random.choice(cell.neighbors)
                posx = loc.x
                posy = loc.y
                ## Add new cell with same brain/strategy
                self.tiles[posx][posy].CA = True
                newcell = CA(self)
                newcell.x = posx
                newcell.y = posy
                newcell.getNeighbors(self)
                mutation = np.random.normal(0,0.1,size=(4,4))
                newcell.brain = cell.brain + mutation
                newcell.fill = cell.fill
                self.CAs.append(newcell)
               # print("A cell reproduced!")
            else:
                self.CAs = self.CAs
    
    def checkDeath(self):
        for i in self.CAs:
            cell = i
            if cell.fill <= 0:
                cell.end = self.steps
                self.data.append([cell.ID, cell.start, cell.end, cell.children])
                posx = cell.x
                posy = cell.y
                self.tiles[posx][posy].CA = False
                self.CAs.remove(cell)
              #  print("A cell has died :(")
            else:
                self.CAs = self.CAs
#            print(i)
#            lent = len(self.CAs)
#            print("Length: %d" % lent)
                
    def checkEnd(self):
        if len(self.CAs) == 0:
            print("All cells have died.")
            return True
        else:
            return False
                
    def addFood(self):
        count = self.newfood
        while count > 0:
            posx = random.randint(0,(self.X-1))
            posy = random.randint(0,(self.Y-1))
            til = self.tiles[posx,posy]
            if til.CA == False:
                if til.CAfood == False:
                    chance = random.uniform(0,1)
                    if chance < self.foodchance:
                        til.CAfood = True
                        til.foodtough = random.uniform(self.ftl, self.fth)
                        til.ftcap = til.foodtough
                        count = count - 1
                    else:
                        count = count - 1
            if count == 0:
                break
        
    def step(self):
        self.updateCA()
        self.updateCAGrad()
        self.updateCAFT()
        self.addFood()
        self.updateCAFGrad()
        for cell in self.CAs:
            cell.getNeighbors(self)
            ## Move each cell and update chemical gradiants
            ## Note that there are two move functions to choose from, comment
            ## out one.
            ## cell.movebrain(self)
            cell.movesimple(self)
            self.updateCAGrad()
            ## Each cell eats and update chemical gradients
            cell.eat(self)
            self.updateCAFGrad()
            ## Cell food usage
            cell.fill = cell.fill - self.foodloss
        ## Check cell fills for reproduction and death & possible simulation end
        self.checkDeath()
        self.checkRepro()
        ## Add food & update gradients
        self.showWorld()
        self.steps = self.steps + 1

## CELLS
class CA:
    def __init__(self,world):
         ## CA Properties
#         self.genome = np.zeros(5)
#         self.ind = None
#         self.repro = True
#         self.diff = False
         self.ID = len(world.CAs) + 1
         self.start = world.steps
         self.end = 0
         self.children = 0
         self.fill = random.uniform(0.5,0.7)
         self.x = random.randrange(0,(world.X-1),1)
         self.y = random.randrange(0,(world.Y-1),1)
         self.neighbors = []
         self.brain = np.random.uniform(-1,1,size=(4,4))
         self.enz = 1.0
         
    def __str__(self):
        return 'Fill: %s. Ind: %s' % (self.fill, self.ind)
    
    def getNeighbors(self, world):
        x = self.x
        y = self.y
        coordlist = [(x+1,y), (x,y+1),
                     (x-1,y), (x,y-1)]
        coordlist = world.wrapper(coordlist)
        neighborlist = []
        for coord in coordlist:
            ## For some reason, this is backwards
            neighborlist.append(world.tiles[coord[0]][coord[1]])
        self.neighbors = neighborlist
    
    ## MOVEMENT FUNCTIONS
    def checkFill(self, neighbor):
        ## Below line does not work
        if neighbor.CA == False:
            if neighbor.CAfood == False:
                return False
            else: 
                return True
        else:
            return True 
        
    def moveup(self):
        self.y = self.y+1
    def movedown(self):
        self.y = self.y-1
    def moveleft(self):
        self.x = self.x-1
    def moveright(self):
        self.x = self.x+1
    def stay(self):
        self.x = self.x\
        
    def movebrain(self, world):
        sensors = []
        spaces = []
        for tile in self.neighbors:
            sensors.append(tile.CAFgrad)
            isfull = self.checkFill(tile)
            spaces.append(isfull)
        spaces.append(False)
        motors = np.dot(sensors, self.brain)
        motors = motors.tolist()
        checker = 0
        goodmoves = []
        while checker == 0:
            maxmot = max(motors)
            if maxmot > 0 or maxmot < 0:
                move = motors.index(maxmot)
                if spaces[move] == True:
                    motors[move] = -10
                    checker = 0
                else:
                    goodmoves.append(move)
                    checker = 1
            else:
                while 0 in motors:
                    zeroplace = motors.index(0)
                    goodmoves.append(zeroplace)
                    motors[zeroplace] = -10
                checker = 1
        move = random.choice(goodmoves)
        if move == 0:
            self.moveright()
        elif move == 1:
            self.moveup()
        elif move == 2:
            self.moveleft()
        elif move == 3:
            self.movedown()
        elif move == 4:
            self.stay()
        coords = [(self.x, self.y)]
        coords = world.wrapper(coords)
        self.x = coords[0][0]
        self.y = coords[0][1]
        
    def movesimple(self, world):
        motors = []
        spaces = []
        for tile in self.neighbors:
            motors.append(tile.CAFgrad)
            isfull = self.checkFill(tile)
            spaces.append(isfull)
        spaces.append(False)
        checker = 0
        goodmoves = []
        while checker == 0:
            maxmot = max(motors)
            if maxmot > 0:
                move = motors.index(maxmot)
                if spaces[move] == True:
                    motors[move] = -1
                    checker = 0
                else:
                    goodmoves.append(move)
                    checker = 1
            else:
                while 0 in motors:
                    zeroplace = motors.index(0)
                    goodmoves.append(zeroplace)
                    motors[zeroplace] = -1
                checker = 1
        move = random.choice(goodmoves)
        if move == 0:
            self.moveright()
        elif move == 1:
            self.moveup()
        elif move == 2:
            self.moveleft()
        elif move == 3:
            self.movedown()
        elif move == 4:
            self.stay()
        coords = [(self.x, self.y)]
        coords = world.wrapper(coords)
        self.x = coords[0][0]
        self.y = coords[0][1]
    
    ## EATING FUNCTIONS      
    def eat(self, world):
        for neighbor in self.neighbors:
            if neighbor.CAfood == True:
                posx = neighbor.x
                posy = neighbor.y
                world.tiles[posx][posy].foodtough = world.tiles[posx][posy].foodtough - self.enz
                if world.tiles[posx][posy].foodtough <= 0:
                    self.fill = self.fill + world.foodvalue
                    world.tiles[posx][posy].CAfood = False
                    world.tiles[posx][posy].foodtough = 0.0
                    world.tiles[posx][posy].ftcap = 0.0
                else:
                    self.fill = self.fill
            else:
                self.fill = self.fill
                

