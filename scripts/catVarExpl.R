
#setwd("C:/Users/tmoore/Documents/qtw/public_html/data/")
BFCases <- read.table(file="http://cs.wellesley.edu/~qtw/data/BirdFluCases.txt", header = TRUE,sep="\t")
names(BFCases)
str(BFCases)

BFDeaths <- read.table(file="http://cs.wellesley.edu/~qtw/data/BirdFluDeaths.txt", header = TRUE,sep="\t")
names(BFDeaths)
str(BFDeaths)

#Let's compare cases by year
Cases  <- rowSums(BFCases[,2:16])
names(Cases) <- BFCases[,1]
Cases
#Let's also compare deaths by year
Deaths <- rowSums(BFDeaths[,2:16])
names(Deaths) <- BFDeaths[,1]
Deaths
#We want to plot Cases and Deaths in a bar chart
Counts <- cbind(Cases, Deaths)
Counts

par(mfrow = c(2,2), mar = c(3, 3, 2, 1))
png('../slides/bar1.png')
barplot(Cases , main = "Bird flu cases")
dev.off()
png('../slides/bar2.png')
barplot(Counts)
dev.off()
png('../slides/bar3.png')
barplot(t(Counts), col = gray(c(0.5,1)))
dev.off()
png('../slides/bar4.png')
barplot(t(Counts), beside = TRUE)
dev.off()

#how about all together
png('../slides/bar5.png')
par(mfrow = c(2,2), mar = c(3, 3, 2, 1))
barplot(Cases , main = "Bird flu cases")
barplot(Counts)
barplot(t(Counts), col = gray(c(0.5,1)))
barplot(t(Counts), beside = TRUE)
dev.off()

#Mosaics
require(stats)
png('../slides/mos1.png')
mosaicplot(Counts)
dev.off()
#OK but does it make sense to compare counts to deaths?  Really we want to see the proportion of cases that lead to death over time
BFProp<-cbind(Survivors=Cases-Deaths,Deaths)
png('../slides/mos2.png')
mosaicplot(BFProp,col=c('green','red'),main="Bird flu survival rates by year")
dev.off()

#Exercise: break up according to country, not year

# Step 2
CCases<-colSums(BFCases[,2:length(BFCases)])
CDeaths<-colSums(BFDeaths[,2:length(BFDeaths)])

# Step 3
BFCountry<-cbind(CCases,CDeaths)
png('../slides/r_cat_ex1_bar.png')
par(las=2,mar=c(6,4,2,1))
barplot(t(BFCountry), beside = T, 
        main="Bird flu cases and deaths by contry",
        ylab="#",col=c('blue','red'))
legend("topright",c("Cases","Deaths"),fill=c('blue','red'))
dev.off()

#Step 4
BFCountryProp<-cbind(Survivors=CCases-CDeaths,CDeaths)
png('../slides/r_cat_ex1_mos.png')
par(las=2,mar=c(1,1,2,1))
mosaicplot(BFCountryProp,col=c("green","red"),main="Bird flu cases and deaths by country")
dev.off()

#Bonus, let's do the mosaic plot in sorted order.
#let's sort by countries the the most bird flu cases to the least
#here's a sorted list of countries
names(sort(CCases,decreasing=T))
#we can then reorder the matrix rows by the sorted names
BFCountryProp<-BFCountryProp[names(sort(CCases,decreasing=T)),]
png('../slides/r_cat_ex1_mos_srt.png')
par(las=2,mar=c(1,1,2,1))
mosaicplot(BFCountryProp,col=c('green','red'),main="Bird flu survival rates by country")
dev.off()


#What if we wanted to plot the results in order of the number of cases
BFCountryProp<-BFCountryProp[names(sort(CCases,decreasing=T)),]
mosaicplot(BFCountryProp,col=c('green','red'),main="Bird flu survival rates by country")

#onto campaign contributions example
rs<-read.table('http://cs.wellesley.edu/~qtw/data/regSuperCensusMod.csv',header=T,sep=',')
rsi<-rs[rs$IndOrg=="IND",]
head(rsi)

#set.seed(1337)  #choose randomly, but in a reproducible way
#rsmall<-rsi[sample(nrow(rsi),500),]
#rsmr<-rsmall[rsmall$RegularSuper=='Regular',]

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


#Break up regular state-by-state contributions by candidateas an exercise
#first a count
r.state<-table(rsi[rsi$RegularSuper=='Regular',c('State','Candidate')])
png('../slides/cc-mos4.png')
mosaicplot(r.state,main='# of contributions by type and state')
dev.off()
#how to sort by bigggest state
#first create a vector of contributions
stateCont<-table(rsi[rsi$RegularSuper=='Regular','State'])
stateCont
r.state.srt<-r.state[names(sort(stateCont,decreasing=T)),]
r.state.srt
mosaicplot(r.state.srt,col=rainbow(3),main='# of contributions by type and state')
png('../slides/cc-mos5.png')
mosaicplot(r.state.srt[1:20,],col=rainbow(3),main='# of contributions (top 20 states sorted by # contributions)')
dev.off()

#now sort by 
regCont<-rsi[rsi$RegularSuper=='Regular',]
stateSumCont<-tapply(regCont$CAmount,regCont$State,sum)
r.state.srtsum<-r.state[names(sort(stateSumCont,decreasing=T)),]
png('../slides/cc-mos6.png')
par(mfrow = c(2,1), mar = c(1, 3, 2, 1),las=2)
mosaicplot(r.state.srt[1:20,],col=rainbow(3),main='# of contributions (top 20 states sorted by # contributions)')
mosaicplot(r.state.srtsum[1:20,],col=rainbow(3),main='# of contributions (top 20 states sorted by $ contributions)')
dev.off()
#Exercise 2: compute a 2-way plot including $ contributions sorted by # and $

#first must compute the sum total of contributions as contingency 

r.state.cont.df<-aggregate(CAmount~State+Candidate,data=regCont,sum)
r.state.cont<-array(data=r.state.cont.df$CAmount,
                    dim=c(length(levels(r.state.cont.df$State)),
                          length(levels(r.state.cont.df$Candidate))),
                    dimnames=list(levels(r.state.cont.df$State),
                                  levels(r.state.cont.df$Candidate)))

r.state.cont.srt<-r.state.cont[names(sort(stateCont,decreasing=T)),]
mosaicplot(r.state.cont.srt[1:20,],col=rainbow(3),main='$ of contributions (top 20 states sorted by # contributions)')

#ct.conts$State<-factor(ct.conts$State[,drop=T])
#we have a problem, for states where there was zero contribution for one or more of the candidates. How to include?
cts<-array(data=ct.conts$CAmount,
           dim=c(length(levels(ct.conts$State)),
                 length(levels(ct.conts$Candidate))),
           dimnames=list(levels(ct.conts$State),
                         levels(ct.conts$Candidate)))


