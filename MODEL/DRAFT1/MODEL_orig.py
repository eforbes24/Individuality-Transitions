#### AUTOMATA MODEL ####
## Eden Forbes
## Simulation

#### LIBRARIES

#### SCRIPTS
import FUNCTIONS
import OBJECTS

#### DEFINE CONSTANTS ####
## World Dimensions
X = 50
Y = 50

## Chemotaxis
g1 = 0.05
g2 = 0.03
g3 = 0.01

## Simulation Features
timesteps = 100
start_CA = 10
CAfood = 100
INDfood = 0

#### DEFINE WORLD ####
world = FUNCTIONS.generateWorld(X,Y,start_CA,CAfood,INDfood)
CAs = FUNCTIONS.generateCAs(world)
# FUNCTIONS.showWorld(world)

for t in range(timesteps):
    world = FUNCTIONS.step(world, X, Y, g1, g2, g3)
    # FUNCTIONS.showWorld(world)



