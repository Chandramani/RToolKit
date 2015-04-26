
rs<-read.table('http://cs.wellesley.edu/~qtw/data/regSuperCensusMod.csv',header=T,sep=',')
rsi<-rs[rs$IndOrg=="IND",]

ri<-rsi[rsi$RegularSuper=="Regular"&rsi$CAmount<=2500&rsi$CAmount>0,]
ri$cf<-clevel3

#Step 1: Use aggregate to create a data frame tallying the total $ contributions grouped by contribution levels
contsums.df<-aggregate(CAmount ~ Candidate + cf, data=ri,sum)


#Step 2: Convert the data frame to an array,  contingency table 
contsums.tbl<-array(data=contsums.df$CAmount,
            dim=c(length(levels(contsums.df$Candidate)),
                    length(levels(contsums.df$cf))),
            dimnames=list(levels(contsums.df$Candidate),
                    levels(contsums.df$cf)))

#Step 3: 
png('../slides/mult-ex1.png')
mosaicplot(contsums.tbl,main='Candidates versus contribution levels ($)',
		col=rainbow(3))
dev.off()

png('../slides/mult-ex2.png')
dotchart(contsums.tbl,main='Candidates versus contribution levels ($)',xlab='$ contributions')
dev.off()

