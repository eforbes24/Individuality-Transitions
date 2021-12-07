## Q700 ANALYSIS SCRIPT
library(dplyr)
library(ggplot2)

#### POPULATION STUDIES ####
pop_data <- read.csv("MAIN/12_6_21(D8)_POPdata.csv")
colnames(pop_data) <- c("RUN","STEP","NUM_UNI","NUM_MULTI","NUM_CELLS","NUM_FOOD")
pop_data$RUN <- as.character(pop_data$RUN)
pop_data$RUN <- factor(pop_data$RUN, levels = c(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19))

pop_datastep <- pop_data %>% group_by(STEP)
pop_datastep <- pop_datastep %>% summarise(
  mean_uni = mean(NUM_UNI),
  sdv_uni = sd(NUM_UNI),
  mean_multi = mean(NUM_MULTI),
  sdv_multi = sd(NUM_MULTI),
  mean_cells = mean(NUM_CELLS),
  sdv_cells = sd(NUM_CELLS),
  mean_food = mean(NUM_FOOD),
  sdv_food = sd(NUM_FOOD)
)

## UNICELL
pop_data_uni <- pop_data %>% select(RUN,STEP,NUM_UNI)

ggplot(data=pop_data_uni, aes(x=STEP, y = NUM_UNI, ymin = 0, ymax = 50, color=RUN)) +
  geom_line() +
  xlab("Timestep") +
  ylab("Number of Unicellular Agents") +
  theme_bw() + 
  theme_classic()

## MULTICELL
pop_data_multi <- pop_data %>% select(RUN,STEP,NUM_MULTI)

ggplot(data=pop_data_multi, aes(x=STEP, y = NUM_MULTI, ymin = 0, ymax = 50, color=RUN)) +
  geom_line() +
  xlab("Timestep") +
  ylab("Number of Multicellular Agents") +
  theme_bw() + 
  theme_classic()

## CELL
pop_data_cells <- pop_data %>% select(RUN,STEP,NUM_CELLS)

ggplot(data=pop_data_cells, aes(x=STEP, y = NUM_CELLS, ymin = 0, ymax = 50, color=RUN)) +
  geom_line() +
  xlab("Timestep") +
  ylab("Number of Cells (Total Unicellular Agents)") +
  theme_bw() + 
  theme_classic()


## FOOD
pop_data_food <- pop_data %>% select(RUN,STEP,NUM_FOOD)

ggplot(data=pop_data_food, aes(x=STEP, y = NUM_FOOD, ymin = 0, ymax = 50, color=RUN)) +
  geom_line() +
  xlab("Timestep") +
  ylab("Number of Food Sources") +
  theme_bw() + 
  theme_classic()


#### CELL STUDIES #### 

## UNICELL ##
org_data <- read.csv("MAIN/12_6_21(D8)_CAdata.csv")
colnames(org_data) <- c("RUN","TYPE","DEATH","ID","START_STEP","END_STEP","LIFESPAN","INMULTI","CHILDREN")

cell_data <- org_data %>% filter(TYPE == "cell")
cell_data$END_STEP[cell_data$END_STEP==0] <- 500
cell_data <- cell_data %>% 
  select(-LIFESPAN) %>% 
  mutate(LIFESPAN = END_STEP - START_STEP)

## FULL RUN
ggplot(cell_data, aes(x=LIFESPAN, y = INMULTI)) +
  geom_point() +
  geom_smooth(method = "lm", lwd = 2.0) +
  xlim(0, 500) + 
  ylim(0, 500) +
  xlab("Lifespan") +
  ylab("Steps as Part of a Multicellular Agent") +
  theme_bw() + 
  theme_classic()
model <- lm(cell_data$INMULTI~cell_data$LIFESPAN)
summary(model)

cell_death_stats <- cell_data %>% 
  group_by(DEATH) %>% 
  summarise(
    mean_lifespan = mean(LIFESPAN),
    std_lifespan = sd(LIFESPAN),
    mean_inmulti = mean(INMULTI),
    std_inmulti = sd(INMULTI)
  )

ggplot(cell_data, aes(x=DEATH, y=LIFESPAN, fill=DEATH)) +
  geom_boxplot() +
  xlab("Agent Type at Death") +
  ylab("Lifespan") +
  theme_bw() + 
  theme_classic()

## TRIM FIRST 150 STEPS FOR INITIAL DIE OFF FOOD CRATER
cell_data2 <- cell_data %>%
  filter(END_STEP>200)
  
ggplot(cell_data2, aes(x=LIFESPAN, y = INMULTI)) +
  geom_point() +
  geom_smooth(method = "lm", lwd = 2.0) +
  xlim(0, 500) + 
  ylim(0, 500) +
  xlab("Lifespan") +
  ylab("Steps as Part of a Multicellular Agent") +
  theme_bw() + 
  theme_classic()
model2 <- lm(cell_data2$INMULTI~cell_data2$LIFESPAN)
summary(model2)

cell_death_stats2 <- cell_data2 %>% 
  group_by(DEATH) %>% 
  summarise(
    mean_lifespan = mean(LIFESPAN),
    std_lifespan = sd(LIFESPAN),
    mean_inmulti = mean(INMULTI),
    std_inmulti = sd(INMULTI)
  )

ggplot(cell_data2, aes(x=DEATH, y=LIFESPAN, fill=DEATH)) +
  geom_boxplot() +
  xlab("Agent Type at Death") +
  ylab("Lifespan") +
  theme_bw() + 
  theme_classic()

## MULTICELL ##

multi_data <- org_data 
multi_data$END_STEP[multi_data$END_STEP==0] <- 500
multi_data <- multi_data %>% 
  select(-LIFESPAN) %>% 
  mutate(LIFESPAN = END_STEP - START_STEP)

ggplot(multi_data, aes(x=TYPE, y=LIFESPAN, fill=DEATH)) +
  geom_boxplot() +
  xlab("Kind of Agent") +
  ylab("Lifespan") +
  theme_bw() + 
  theme_classic()











