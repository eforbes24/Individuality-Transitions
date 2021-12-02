#### AUTOMATA MODEL ####
## Eden Forbes
## Object Class Definitions

#### LIBRARIES
import numpy as np
import random
from matplotlib import pyplot

#### OBJECTS
## TILE
class Tile:
    CA = False
    IND = False
    CAfood = False
    CAgrad = 0.0
    CAFgrad = 0.0
    foodtough = 0.0
    ftcap = 0.0
    foodvalue = 0.0
    x = 0
    y = 0
    
    
class World:
    def __init__(self, timesteps, X, Y, start_CA, CAfood, foodregenz,
                 g0, g1, g2, g3, g4, g5, foodtoughlow, foodtoughhigh, nutrition,
                 lowfillstart, highfillstart, foodloss, newfood, 
                 foodchance, paniclevel):
        ## Give world attributes
        self.timesteps = timesteps
        self.X = X
        self.Y = Y
        self.foodloss = foodloss
        self.newfood = newfood
        self.foodchance = foodchance
        self.steps = 0
        self.CAs = list()
        self.Multis = list()
        self.paniclevel = paniclevel
        ## Gradient & Enzyme attributes
        self.G0 = g0
        self.G1 = g1
        self.G2 = g2
        self.G3 = g3
        self.G4 = g4
        self.G5 = g5
        self.foodregenz = foodregenz
        self.ftl = foodtoughlow
        self.fth = foodtoughhigh
        self.nutrition = nutrition
        self.lowfillstart = lowfillstart
        self.highfillstart = highfillstart
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
        ## World Output Data
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
    
    def showWorld(self):
        array = np.zeros((self.X, self.Y))
        tilearray = self.tiles.flatten()
        for til in tilearray:
            locx = til.x
            locy = til.y
            if til.CA == True:
                array[locx,locy] = self.G0 + 3
            elif til.IND == True:
                array[locx,locy] = self.G0 + 2
            elif til.CAfood == True:
                array[locx,locy] = self.G0 + 1
            else:
                array[locx,locy] = til.CAFgrad
        ## Gotta pick the best color map
        pyplot.imshow(array, interpolation='nearest', cmap='gist_earth')
        pyplot.show()
    
    
    ####################################
    ##  CHEMO GRADIENT FUNCTIONS
    ####################################
    
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
    
    def fifthDegree(self, x, y):
        coordlist = [(x+3,y+2), (x-3,y+2), (x-3,y-2), (x+3,y-2),
                     (x+2,y+3), (x-2,y+3), (x-2,y-3), (x+2,y-3),
                     (x+4,y+1), (x-4,y+1), (x-1,y-4), (x+1,y-4),
                     (x+1,y+4), (x-1,y+4), (x-4,y-1), (x+4,y-1),
                     (x+5,y), (x,y+5), (x-5,y), (x,y-5)]
        coordlist = self.wrapper(coordlist)
        fiList = []
        for i in coordlist:
            fiList.append(self.tiles[i[0]][i[1]])
        return fiList
    
    def addFood(self):
        count = self.newfood
        while count > 0:
            posx = random.randint(0,(self.X-1))
            posy = random.randint(0,(self.Y-1))
            til = self.tiles[posx,posy]
            if til.CA == False:
                if til.CAfood == False:
                    if til.IND == False:
                        chance = random.uniform(0,1)
                        if chance < self.foodchance:
                            til.CAfood = True
                            til.foodtough = random.uniform(self.ftl, self.fth)
                            til.foodvalue = til.foodtough * self.nutrition
                            til.ftcap = til.foodtough
                            count = count - 1
                        else:
                            count = count - 1
                if count == 0:
                    break
    
    ####################################
    ##  UPDATE TILE FUNCTIONS
    ####################################
    
    def updateCAGrad(self):
        for row in self.tiles:
            for tile in row:
                tile.CAgrad = 0
        CAx = []
        CAy = []
        for cell in self.CAs:
            CAx.append(cell.x)
            CAy.append(cell.y)
        #### NOT SURE HOW THIS WILL CHANGE THINGS
        for multi in self.Multis:
            for cell in multi.CAs:
                CAx.append(cell.x)
                CAy.append(cell.y)
        #### SO CHECK IT OUT
        for i in range(len(CAx)):
            posx = CAx[i]
            posy = CAy[i]
            ## Identity
            self.tiles[posx][posy].CAgrad = self.G0
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
            fiList = self.fifthDegree(posx,posy)
            for tile in fiList:
                tile.CAgrad = tile.CAgrad + self.G5
                
    def updateCAFGrad(self):
        for row in self.tiles:
            for tile in row:
                tile.CAFgrad = 0
        for i in range(self.X):
            for j in range(self.Y):
                tile = self.tiles[i][j]
                posx = tile.x
                posy = tile.y
                if tile.CAfood == True:
                    ## Identity
                    self.tiles[posx][posy].CAgrad = self.G0
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
                    fiList = self.fifthDegree(posx,posy)
                    for tile in fiList:
                        tile.CAFgrad = tile.CAFgrad + self.G5
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
            
    def updateIND(self):
        for row in self.tiles:
            for tile in row:
                tile.IND = False
        for multi in self.Multis:
            for cell in multi.CAs:
                posx = cell.x
                posy = cell.y
                self.tiles[posx][posy].IND = True
            
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
                    
    ####################################
    ##  CHECK CONDITION FUNCTIONS
    ####################################
           
    def checkRepro(self):
        for cell in self.CAs:
            if cell.fill >= 1:
                cell.children = cell.children + 1
                cell.fill = cell.fill/2
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
    
    def checkCAAggregate(self):
        for cell in self.CAs:
            if cell.fill < self.paniclevel:
                for neighbor in cell.neighbors:
                    if neighbor.CA == True:
                        coords = self.wrapper([(neighbor.x,neighbor.y)])
                        posx = coords[0][0]
                        posy = coords[0][1]
                        rng = len(self.CAs)
                        for i in (range(rng)):
                            if self.CAs[i].x == posx and self.CAs[i].y == posy:
                                cell2 = self.CAs[i]
                                break
                        if cell2.fill < self.paniclevel:
                            newind = Multi(self)
                            newind.CAs.append(cell)
                            newind.CAs.append(cell2)
                            self.Multis.append(newind)
                            self.tiles[posx][posy].IND = True
                            self.tiles[cell.x][cell.y].IND = True
                            # print("A new individual has formed!")
                            self.CAs.remove(cell)
                            self.CAs.remove(cell2)
                            self.tiles[posx][posy].CA = False
                            self.tiles[cell.x][cell.y].CA = False
                    elif neighbor.IND == True:
                        coords = self.wrapper([(neighbor.x,neighbor.y)])
                        posx = coords[0][0]
                        posy = coords[0][1]
                        for multi in self.Multis:
                            for CA in multi.CAs:
                                if CA.x == posx and CA.y == posy:
                                    if CA.fill < self.paniclevel:
                                        multi.CAs.append(cell)
                                        self.tiles[cell.x][cell.y].IND = True
                                        self.tiles[cell.x][cell.y].CA = False
                                        self.CAs.remove(cell)
                                        break
    
    def checkINDAggregate(self):
        for multi in self.Multis:
            for cell in multi.CAs:
                if cell.fill < self.paniclevel:
                    for neighbor in cell.neighbors:
                        if neighbor not in multi.CAs:
                            if neighbor.IND == True:
                                coords = self.wrapper([(neighbor.x,neighbor.y)])
                                posx = coords[0][0]
                                posy = coords[0][1]
#                                for multiB in self.Multis:
#                                    for cellB in multiB.CAs:
#                                        if cellB.x == posx and cellB.y == posy:
#                                            if cellB.fill < self.paniclevel:
#                                                multi.CAs.append(multiB.CAs)
#                                                self.Multis.remove(multiB)
    
    def checkDeath(self):
        for cell in self.CAs:
            if cell.fill <= 0:
                cell.end = self.steps
                self.data.append(["cell", cell.ID, cell.start, cell.end, cell.children])
                posx = cell.x
                posy = cell.y
                self.tiles[posx][posy].CA = False
                self.CAs.remove(cell)
              #  print("A cell has died :(")
            else:
                self.CAs = self.CAs
        for multi in self.Multis:
            for cell in multi.CAs:
                if cell.fill <= 0:
                    multi.end = self.steps
                    for cell in multi.CAs:
                        posx = cell.x
                        posy = cell.y
                        self.tiles[posx][posy].IND = False
                    self.data.append(["multi", multi.ID, multi.start, multi.end, multi.CAs])
                    self.Multis.remove(multi)
                    break
                    #  print("A multi has died :(")
            else:
                self.Multis = self.Multis
                
    def checkEnd(self):
        if len(self.CAs) == 0 and len(self.Multis) == 0:
            print("All beings have died.")
            return True
        else:
            return False
        
    ####################################
    ##  STEP FUNCTION
    ####################################
        
    def step(self):
        self.addFood()
        self.updateCAFT()
        self.updateCAFGrad()
        self.updateCAGrad()
        
        for multi in self.Multis:
            multi.movemulti(self)
            self.updateIND()
            self.updateCAGrad()
            multi.getNeighborsIND(self)
            multi.multieat(self)
            self.updateCAFGrad()
        
        for cell in self.CAs:
            cell.getNeighbors(self)
            ## Move each cell and update chemical gradiants
            ## Note that there are two move functions to choose from, comment
            ## out one.
            ## cell.movebrain(self)
            cell.movesimple(self)
            self.updateCA()
            self.updateCAGrad()
            cell.getNeighbors(self)
            ## Each cell eats and update chemical gradients
            cell.eat(self)
            self.updateCAFGrad()
        ## Check cell fills for reproduction and death & possible simulation end
        self.checkCAAggregate()
        self.checkDeath()
        #self.checkINDAggregate()
        self.checkRepro()
        ## Update CAs, Show world & Step
        self.showWorld()
        self.steps = self.steps + 1

########################################
########################################
##  UNICELLULARS
########################################                        
########################################
        
class CA:
    def __init__(self,world):
         ## CA Properties
         self.ID = len(world.CAs) + 1
         self.start = world.steps
         self.end = 0
         
         self.genome = np.zeros(5)
         self.diff = False
         self.fill = random.uniform(world.lowfillstart, world.highfillstart)
         self.x = random.randrange(0,(world.X-1),1)
         self.y = random.randrange(0,(world.Y-1),1)
         
         self.neighbors = []
         self.children = 0
         
         self.brain = np.random.uniform(-1,1,size=(4,4))
         
         self.enz = 1.0
         
    def __str__(self):
        return 'Fill: %s.' % (self.fill)
    
    def getNeighbors(self, world):
        x = self.x
        y = self.y
        ## ncl = neighbor coord list
        ncl = [(x+1,y), (x,y+1),
               (x-1,y), (x,y-1)]
        ncl = world.wrapper(ncl)
        neighborlist = []
        for coord in ncl:
            ## For some reason, this is backwards
            neighborlist.append(world.tiles[coord[0]][coord[1]])
        self.neighbors = neighborlist
    
    ####################################
    ##  MOVEMENT FUNCTIONS
    ####################################
    
    def checkFill(self, neighbor):
        if neighbor.CA == False:
            if neighbor.CAfood == False:
                if neighbor.IND == False:
                    return False
                else: 
                    return True
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
        self.x = self.x
        
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
        ## Two different sets of information used as input, depending on 
        ## if the cell is well-enough fed
        ## Unfortunately, panic cells will avoid food for the sake of finding 
        ## others right now (not sure if this is realistic...)
        if self.fill > world.paniclevel:
            for tile in self.neighbors:
                motors.append(tile.CAFgrad)
                isfull = self.checkFill(tile)
                spaces.append(isfull)
        else:
            for tile in self.neighbors:
                motors.append(tile.CAgrad)
                isfull = self.checkFill(tile)
                spaces.append(isfull)
        spaces.append(False)
        goodmoves = []
        trumaxmot = 0
        maxmot = max(motors)
        while maxmot >= 0:
            if maxmot >= trumaxmot:
                move = motors.index(maxmot)
                if spaces[move] == True:
                    motors[move] = -10
                else:
                    goodmoves.append(move)
                    motors[move] = -10
                trumaxmot = maxmot
                maxmot = max(motors)
            else:
                goodmoves.append(4)
                break
        ## goodmoves empty sometimes for some reason
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
    
    ####################################
    ##  EATING FUNCTIONS
    ####################################
     
    def eat(self, world):
        for neighbor in self.neighbors:
            if neighbor.CAfood == True:
                posx = neighbor.x
                posy = neighbor.y
                world.tiles[posx][posy].foodtough = world.tiles[posx][posy].foodtough - self.enz
                if world.tiles[posx][posy].foodtough <= 0:
                    self.fill = self.fill + world.tiles[posx][posy].foodvalue
                    world.tiles[posx][posy].CAfood = False
                    world.tiles[posx][posy].foodvalue = 0.0
                    world.tiles[posx][posy].foodtough = 0.0
                    world.tiles[posx][posy].ftcap = 0.0
                else:
                    self.fill = self.fill
            else:
                self.fill = self.fill
        self.fill = self.fill - world.foodloss
        
########################################
########################################
##  MULTICELLULARS
########################################                        
########################################
              
class Multi:
    def __init__(self,world):
        self.ID = len(world.Multis) + 1
        self.start = world.steps
        self.end = 0
        
        self.CAs = list()
        self.size = len(self.CAs)
        
        self.tnl = list()
        self.tdl = list()
        
        self.enz = 2
        
    def __str__(self):
        return 'ID: %s. Size: %s' % (self.ID, self.size)
    
    def getNeighborsIND(self, world):
        ## tnl = total neighbor list, gives you all the neighbors of the entire
        ## multi body
        tnl = []
        ## tdl = total direct list, tells you where the neighbor is in relation
        ## to the multi
        tdl = []
        for cell in self.CAs:
            cell.getNeighbors(world)
            directcount = 0
            for neighbor in cell.neighbors:
                if neighbor in tnl:
                    tnl = tnl
                    directcount = directcount + 1
                else:
                    tnl.append(neighbor)
                    tdl.append(directcount)
                    directcount = directcount + 1
        for cell in self.CAs:
            posx = cell.x
            posy = cell.y
            while world.tiles[posx][posy] in tnl:
                tnl.remove(world.tiles[posx][posy])
        self.tnl = tnl
        ### RIGHT[0] UP[1] LEFT[2] DOWN[3]
        self.tdl = tdl
    
    ####################################
    ##  MOVEMENT FUNCTIONS
    ####################################
    
    def checkFillIND(self):
        spaces = [False,False,False,False,False]
        counter = 0
        for neighbor in self.tnl:
            nd = self.tdl[counter]
            if neighbor.CA == False:
                if neighbor.CAfood == False:
                    if neighbor.IND == False:
                        if spaces[nd] == True:
                            spaces[nd] = True
                            counter = counter + 1
                        else:
                            spaces[nd] = False
                            counter = counter + 1
                    else:
                        spaces[nd] = True
                        counter = counter + 1
                else:
                    spaces[nd] = True
                    counter = counter + 1
            else:
                spaces[nd] = True
                counter = counter + 1
        return spaces
            
    def direct(self, world):
        ## Here is where you do all the inbodied math for deciding where to go
        self.getNeighborsIND(world)
        motors = [0,0,0,0]
        spaces = self.checkFillIND()
        ## Fill motors with collective food sense
        for i in range(len(self.tnl)):
            relloc = self.tdl[i]
            motors[relloc] = motors[relloc] + self.tnl[i].CAFgrad
        ## Then do the max math (best for now)
        goodmoves = []
        trumaxmot = 0
        maxmot = max(motors)
        while maxmot >= 0:
            if maxmot >= trumaxmot:
                move = motors.index(maxmot)
                if spaces[move] == True:
                    motors[move] = -10
                else:
                    goodmoves.append(move)
                    motors[move] = -10
                trumaxmot = maxmot
                maxmot = max(motors)
            else:
                goodmoves.append(4)
                break
        move = random.choice(goodmoves)
        return move
    
    ## OTHER MOVEMENT IDEAS (from Josh & Kory)
        ## Have cells emit signals to one another that play into their 
        ## chemotactic decisions
        
        ## Store the actions for each cell as decisions are made, then execute
        ## all at once (instead of moving one cell at a time: keep the current
        ## state of the cells and the state at t+1) --> should implement this
        ## up top too
        
    def movemulti(self, world):
        direct = self.direct(world)
        if direct == 0:
            for cell in self.CAs:
                #print("moved up")
                cell.moveup()
                coords = [(cell.x, cell.y)]
                coords = world.wrapper(coords)
                cell.x = coords[0][0]
                cell.y = coords[0][1]
        elif direct == 1:
            for cell in self.CAs:
                #print("moved down")
                cell.movedown()
                coords = [(cell.x, cell.y)]
                coords = world.wrapper(coords)
                cell.x = coords[0][0]
                cell.y = coords[0][1]
        elif direct == 2:
            for cell in self.CAs:
                #print("moved right")
                cell.moveright()
                coords = [(cell.x, cell.y)]
                coords = world.wrapper(coords)
                cell.x = coords[0][0]
                cell.y = coords[0][1]
        elif direct == 3:
            for cell in self.CAs:
                #print("moved left")
                cell.moveleft()
                coords = [(cell.x, cell.y)]
                coords = world.wrapper(coords)
                cell.x = coords[0][0]
                cell.y = coords[0][1]
        #print(cell)
                
    ####################################
    ##  EATING FUNCTIONS
    ####################################
        
    def multieat(self, world):
        for neighbor in self.tnl:
            if neighbor.CAfood == True:
                posx = neighbor.x
                posy = neighbor.y
                world.tiles[posx][posy].foodtough = world.tiles[posx][posy].foodtough - self.enz
                if world.tiles[posx][posy].foodtough <= 0:
                    size = len(self.CAs)
                    for cell in self.CAs:
                        cell.fill = cell.fill + (world.tiles[posx][posy].foodvalue/size)
                    world.tiles[posx][posy].CAfood = False
                    world.tiles[posx][posy].foodtough = 0.0
                    world.tiles[posx][posy].ftcap = 0.0
        for cell in self.CAs:
            cell.fill = cell.fill - world.foodloss
