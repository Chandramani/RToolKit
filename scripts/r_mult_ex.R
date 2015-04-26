
rs<-read.table('http://cs.wellesley.edu/~qtw/data/regSuperCensusMod.csv',header=T,sep=',')
rsi<-rs[rs$IndOrg=="IND",]

ri<-rsi[rsi$RegularSuper=="Regular"&rsi$CAmount<=2500&rsi$CAmount>0,]
ri$cf<-clevel3

#Step 1: Use aggregate to create a data frame tallying the total $ contributions grouped by contribution levels



#Step 2: Convert the data frame to an array,  contingency table 


#Step 3: make graph


