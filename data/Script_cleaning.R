library(dplyr)
library(tidyverse)
library(reshape2)
library(stringr)


mat_drug_protein <- read.table("C:/Users/luciapp1/Downloads/mat_drug_protein.txt", quote="\"", comment.char="")

data1 <- mat_drug_protein %>% 
  mutate(row_idx = seq(1:nrow(.))) %>% 
  melt("row_idx") %>% 
  filter(value != 0) %>% 
  group_by(row_idx) %>% 
  mutate(n = n()) %>% 
  filter(n> 1) %>% 
  group_by(row_idx) %>% 
  do(sample_n(.,1)) %>% 
  select(-n) %>% 
  mutate_at(vars("variable"), ~str_replace(.,"V",""))



data2 <- mat_drug_protein %>% 
  mutate(row_idx = seq(1:nrow(.))) %>% 
  melt("row_idx") %>% 
  filter(value != 0) %>% 
  group_by(row_idx) %>% 
  mutate(n = n()) %>% 
  filter(n> 1) %>% 
  anti_join(data1) %>% 
  select(-n) %>% 
  mutate_at(vars("variable"), ~str_replace(.,"V",""))
  
  
  
write.table(as.data.frame(data1), "C:\\Users\\luciapp1\\Downloads\\test2.csv", row.names = FALSE)
write.table(as.data.frame(data2), "C:\\Users\\luciapp1\\Downloads\\train2.csv", row.names = FALSE)

data <- mat_drug_protein %>% 
  mutate(row_idx = seq(1:nrow(.))) %>% 
  melt("row_idx") %>% 
  filter(value != 0) %>% 
  group_by(row_idx) %>% 
  mutate(n = n()) %>% 
  filter(n> 1) %>% 
  group_by(row_idx) 

  
  
  group_by(n) %>% 
  summarise(n2 = n())
