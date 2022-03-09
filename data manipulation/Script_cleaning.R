library(dplyr)
library(tidyverse)
library(reshape2)
library(stringr)

  
mat_drug_protein <- read.table("C:/Users/luciapau/Downloads/mat_protein_drug.txt", quote="\"", comment.char="")
protein <- read.table("C:/Users/luciapau/Downloads/protein.txt", quote="\"", comment.char="")

dup_prot <- protein %>% 
  group_by(V1) %>% 
  summarise(n = n()) #Find occurence of each protein in the list

protein_ex <-protein %>% 
  mutate(row_idx = seq(1:nrow(.))) %>% 
  merge(dup_prot, by = "V1") %>% 
  rename("aux" = "V1") %>% 
  filter(n > 1) #Find duplicated proteins

protein_clean <- protein %>% 
  mutate(row_idx = seq(1:nrow(.))) %>% 
  rename("aux" = "V1") #Generate row Ids to be able to merge the data


data1 <- mat_drug_protein %>% 
  mutate(row_idx = seq(1:nrow(.))) %>%  
  merge(protein_clean, by = c("row_idx")) %>% #Include protein names
  filter(!aux %in% protein_ex$aux) %>%  #Exclude names in the list
  select(-aux) %>%
  mutate(row_idx = seq(1:nrow(.))) %>%
  melt("row_idx") %>% 
  filter(value != 0) %>% 
  group_by(row_idx) %>% 
  mutate(n = n()) %>% 
  filter(n> 1) %>%  #Coerced the proteins to have more than one interaction
  # filter(variable %in% col_dup$variable) %>%
  group_by(row_idx) %>%
  do(sample_n(.,1)) %>% #Random splitting, one interaction per protein is saved for test
  select(-n) %>%
  mutate_at(vars("variable"), ~str_replace(.,"V",""))

#Process replicated, extracting other observations for train

data2 <- mat_drug_protein %>% 
  mutate(row_idx = seq(1:nrow(.))) %>% 
  merge(protein_clean, by = c("row_idx")) %>%
  filter(!aux %in% protein_ex$aux) %>%
  select(-aux) %>%
  mutate(row_idx = seq(1:nrow(.))) %>%
  melt("row_idx") %>% 
  filter(value != 0) %>% 
  group_by(row_idx) %>% 
  mutate(n = n()) %>% 
  filter(n> 1) %>% 
  # filter(variable %in% col_dup$variable) %>% 
  anti_join(data1) %>%  #All IDs except for the ones saved for test
  select(-n) %>% 
  mutate_at(vars("variable"), ~str_replace(.,"V",""))



