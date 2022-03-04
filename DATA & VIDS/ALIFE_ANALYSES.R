## ALIFE ANALYSIS SCRIPT
library(dplyr)
library(ggplot2)

# [1,1,1]frm223:x2534:y716:6c:fYmY_ZOO_DISTdata.csv
# [1,1,1]frm215:x918:y1322:9c:fYmY_ZOO_DISTdata.csv
# [1,1,1]frm232:x3847:y2837:7c:fYmY_ZOO_DISTdata.csv
# [1,1,1]frm223:x2534:y716:6c:fYmN_ZOO_DISTdata.csv
# [1,1,1]frm215:x918:y1322:9c:fYmN_ZOO_DISTdata.csv
# [1,1,1]frm232:x3847:y2837:7c:fYmN_ZOO_DISTdata.csv

dist_data <- read.csv("MAIN/ZOO/[1,1,1]frm215:x918:y1322:9c:fYmY_ZOO_DISTdata.csv")
dist_data = dist_data[-1,]
colnames(dist_data) <- c("ID","FILL","DIST","STEP")

ggplot(data=dist_data, aes(x=STEP, y = FILL, ymin = 0, ymax = 1, color=factor(ID))) +
  geom_line() +
  labs(x = "Timestep", y = "Automaton Fill", color = "Automaton") +
  theme_bw() + 
  theme_classic() +
  theme(text = element_text(size = 15))

ggplot(data=dist_data, aes(x=STEP, y = DIST, ymin = 0, ymax = 6, color=factor(ID))) +
  geom_line() +
  labs(x = "Timestep", y = "Closest Automaton Distance", color = "Automaton") +
  theme_bw() + 
  theme_classic() +
  theme(text = element_text(size = 15))



