# Load required libraries
install.packages("tidyverse", dependencies = TRUE)
library(tidyverse)
library(janitor)
library(randomForest)
library(caret)
library(ggplot2)
library(patchwork)

data <- read.csv("data.csv",sep=",",header = TRUE)
head(data)

data <- data.frame()

install.packages("tidyverse")
install.packages("Rcpp")

install.packages("devtools")
