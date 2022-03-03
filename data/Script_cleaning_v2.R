library(dplyr)
library(tidyverse)
library(reshape2)
library(stringr)


mat_drug_protein <- read.table("C:/Users/LUCIA/Downloads/mat_drug_protein.txt", quote="\"", comment.char="")


col_dup <- mat_drug_protein %>% 
  mutate(row_idx = seq(1:nrow(.))) %>% 
  melt("row_idx") %>% 
  filter(value != 0) %>% 
  group_by(variable) %>% 
  mutate(n = n()) %>% 
  filter(n> 1)


data1 <- mat_drug_protein %>% 
  mutate(row_idx = seq(1:nrow(.))) %>% 
  melt("row_idx") %>% 
  filter(value != 0) %>% 
  group_by(row_idx) %>% 
  mutate(n = n()) %>% 
  filter(n> 1) %>% 
  # filter(variable %in% col_dup$variable) %>% 
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
  # filter(variable %in% col_dup$variable) %>% 
  anti_join(data1) %>% 
  select(-n) %>% 
  mutate_at(vars("variable"), ~str_replace(.,"V",""))

n_drugs <- data1 %>% distinct(row_idx)



write.table(as.data.frame(data1), "C:\\Users\\LUCIA\\Downloads\\df_test_drug_protein4.csv", row.names = TRUE, quotes = FALSE)
write.table(as.data.frame(data2), "C:\\Users\\LUCIA\\Downloads\\df_train_drug_protein4.csv", row.names = FALSE)


mat_protein_disease <- read.table("C:/Users/LUCIA/Downloads/mat_protein_disease.txt", quote="\"", comment.char="")


col_dup <- mat_protein_disease %>% 
  mutate(row_idx = seq(1:nrow(.))) %>% 
  melt("row_idx") %>% 
  filter(value != 0) %>% 
  group_by(variable) %>% 
  mutate(n = n()) %>% 
  filter(n> 1)

data1 <- mat_protein_disease %>% 
  mutate(row_idx = seq(1:nrow(.))) %>% 
  melt("row_idx") %>% 
  filter(value != 0) %>% 
  group_by(row_idx) %>% 
  mutate(n = n()) %>% 
  filter(n> 1) %>% 
  # filter(variable %in% col_dup$variable) %>% 
  group_by(row_idx) %>% 
  do(sample_n(.,1)) %>% 
  select(-n) %>% 
  mutate_at(vars("variable"), ~str_replace(.,"V",""))



data2 <- mat_protein_disease %>% 
  mutate(row_idx = seq(1:nrow(.))) %>% 
  melt("row_idx") %>% 
  filter(value != 0) %>% 
  group_by(row_idx) %>% 
  mutate(n = n()) %>% 
  filter(n> 1) %>% 
  # filter(variable %in% col_dup$variable) %>% 
  anti_join(data1) %>% 
  select(-n) %>% 
  mutate_at(vars("variable"), ~str_replace(.,"V",""))

n_protein <- data1 %>% distinct(row_idx)



write.table(as.data.frame(data1), "C:\\Users\\LUCIA\\Downloads\\df_test_protein_disease4.csv", row.names = FALSE)
write.table(as.data.frame(data2), "C:\\Users\\LUCIA\\Downloads\\df_train_protein_disease4.csv", row.names = FALSE)


mat_drug_disease <- read.table("C:/Users/LUCIA/Downloads/mat_drug_disease.txt", quote="\"", comment.char="")


col_dup <- mat_drug_disease %>% 
  mutate(row_idx = seq(1:nrow(.))) %>% 
  melt("row_idx") %>% 
  filter(value != 0) %>% 
  group_by(variable) %>% 
  mutate(n = n()) %>% 
  filter(n> 1)

data1 <- mat_drug_disease %>% 
  mutate(row_idx = seq(1:nrow(.))) %>% 
  melt("row_idx") %>% 
  filter(value != 0) %>% 
  group_by(row_idx) %>% 
  mutate(n = n()) %>% 
  filter(n> 1) %>% 
  # filter(variable %in% col_dup$variable) %>% 
  group_by(row_idx) %>% 
  do(sample_n(.,1)) %>% 
  select(-n) %>% 
  mutate_at(vars("variable"), ~str_replace(.,"V",""))



data2 <- mat_drug_disease %>% 
  mutate(row_idx = seq(1:nrow(.))) %>% 
  melt("row_idx") %>% 
  filter(value != 0) %>%
  group_by(row_idx) %>% 
  mutate(n = n()) %>% 
  filter(n> 1) %>% 
  # filter(variable %in% col_dup$variable) %>% 
  anti_join(data1) %>% 
  select(-n) %>% 
  mutate_at(vars("variable"), ~str_replace(.,"V",""))

n_drug_dis <- data1 %>% distinct(row_idx)



write.table(as.data.frame(data1), "C:\\Users\\LUCIA\\Downloads\\df_test_drug_disease4.csv", row.names = FALSE)
write.table(as.data.frame(data2), "C:\\Users\\LUCIA\\Downloads\\df_train_drug_disease4.csv", row.names = FALSE)
