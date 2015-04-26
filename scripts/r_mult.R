
#onto campaign contributions example
rs<-read.table('data/regSuperCensusMod.csv',header=T,sep=',')
rsi<-rs[rs$IndOrg=="IND",]
head(rsi)

ct<-table(rsi[,c('Candidate','RegularSuper')])

png('../slides/cc-bar1.png')
barplot(ct,beside=T,col=rainbow(3),main='# of contributions by type and candidate')
legend("topright",rownames(ct),fill=rainbow(3))
dev.off()

png('../slides/cc-mos1.png')
mosaicplot(ct,main='# of contributions by type and candidate')
dev.off()

#here's an alternative using a formula on the data frame
mosaicplot(~Candidate+RegularSuper,data=rsi,main='# of contributions by type and candidate')

#but what if we want to compute a tally on a numeric variable grouped by 2 categorical variables?
#solution: we must first perform a function on the numerical variable
ct.cont<-aggregate(CAmount~Candidate+RegularSuper,data=rsi,sum)
cta<-array(data=ct.cont$CAmount,
            dim=c(length(levels(ct.cont$Candidate)),
                    length(levels(ct.cont$RegularSuper))),
            dimnames=list(levels(ct.cont$Candidate),
                    levels(ct.cont$RegularSuper)))
png('../slides/cc-mos2.png')
mosaicplot(cta,main='$ of contributions by type and candidate')
dev.off()

png('../slides/cc-mos3.png',height=500,width=750,units="px")
par(mfrow = c(1,2))
mosaicplot(ct,main='# of contributions by type and candidate')
mosaicplot(cta,main='$ of contributions by type and candidate')
dev.off()


#OK let's try to make some more Cleveland dot plots.

png('../slides/mult-dot1.png')
dotchart(ct,xlab='total # contributions')
dev.off()

png('../slides/mult-dot2.png')
dotchart(cta,xlab='total $ contributions')
dev.off()


#creating categorical variables
#first let's just study regular contributions and clear out spurious data points
ri<-rsi[rsi$RegularSuper=="Regular"&rsi$CAmount<=2500&rsi$CAmount>0,]
clevel<-cut(ri$CAmount,3)
table(clevel)
clevel2<-cut(ri$CAmount,pretty(ri$CAmount,3))
table(clevel2)
clevel3<-cut(ri$CAmount,breaks=c(0,500,2000,2500),
	labels=c("<$500","$500<=$2000",">$2000"))
table(clevel3)

#This last cut seems best, so let's add it as a factor on the data frame
ri$cf<-clevel3
#now make a contingency table comparing candidate to contribution level
contnums<-table(ri[,c('Candidate','cf'),])
contnums

png('../slides/mult-mos1.png')
mosaicplot(contnums,main='Candidates versus contribution levels (#)',
		col=rainbow(3))
dev.off()

png('../slides/mult-dot6.png')
dotchart(contnums,main='Candidates versus contribution levels (#)',
	xlab='# contributions')
dev.off()


#Now for the exercise



#Let's go over some of the other numerical variables
diversef<-cut(ri$USADiversity,4)
table(diversef)
diverseqf<-cut(ri$USADiversity,quantile(ri$USADiversity,(0:4)/4,na.rm=T))
table(diverseqf)
#Dividing into quartiles offers a more even balance, so add to the data frame
ri$diverseqf<-diverseqf

#create a contingency table on candidate contributions vs. diversity measure
dtab<-table(ri[, c("Candidate", "diverseqf")])
dtab

png('../slides/mult-dot3.png')
dotchart(dtab,main='Candidates versus diversity of contributing states'
	,xlab='# contributions')
dev.off()

png('../slides/mult-dot4.png')
dotchart(prop.table(dtab,1),main='Candidates versus diversity of contributing states',xlab='# contributions')
dev.off()

png('../slides/mult-dot5.png',height=450,width=850,units="px")
par(mfrow = c(1,2))
dotchart(prop.table(dtab,1),main='Candidates vs diversity of contributing states',xlab='# contributions')
dotchart(t(prop.table(dtab,1)),main='Candidates vs diversity of contributing states',xlab='# contributions')
dev.off()
