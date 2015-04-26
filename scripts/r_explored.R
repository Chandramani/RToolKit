
#maybe use the regular contributions (or a random sample) to deal with the skewed data at first.
rs<-read.table('data/regSuperCensusMod.csv',header=T,sep=',')
rsi<-rs[rs$IndOrg=="IND",]

set.seed(1337)  #choose randomly, but in a reproducible way
rsmall<-rsi[sample(nrow(rsi),500),]
rsmr<-rsmall[rsmall$RegularSuper=='Regular',]
summary(rsmr$CAmount)
#for comparison sake let's see if the sample differs by much
summary(rsi$CAmount[rsi$RegularSuper=='Regular'])

#1. boxplot visualizes the summary plus a bit more
png('../slides/images/box1.png') #used to create PNG, not for interactive graphics
boxplot(rsmr$CAmount)
dev.off()
#OK let's fix a few things.  
#first let's exclude negative contributions since they aren't contributions
rsp<-rsmr[rsmr$CAmount>0,]
png('../slides/images/box2.png') 
boxplot(rsp$CAmount)
dev.off()
#Fair enough, let's see how this varies by candidate
png('../slides/images/box3.png') 
boxplot(CAmount~Candidate,data=rsp)
dev.off()
#OK let's make the graph look a bit nicer
png('../slides/images/box4.png') 
boxplot(CAmount~Candidate,data=rsp,
    col="blue",xlab="Candidate",ylab="Contribution Amount",main="Boxplot of Contribution by Candidate")
dev.off()
#yes you can do histograms
png('../slides/images/hist1.png') 
hist(rsp$CAmount)
dev.off()
#but be careful when selecting breaks
png('../slides/images/hist2.png') 
hist(rsp$CAmount,breaks=25)
dev.off()

#a better choice for examining distributions is the dotplot
png('../slides/images/dot1.png') 
dotchart(rsp$CAmount,xlab="Contribution Amount")
dev.off()
#does it vary by candidate?
png('../slides/images/dot2.png')
dotchart(rsp$CAmount,group=rsp$Candidate,xlab="Contribution Amount")
dev.off()

#Now onto CDFs

rap<-rsi[rsi$RegularSuper=='Regular'&rsi$CAmount>0,]

png('../slides/images/cdf1.png')
plot(ecdf(rsp$CAmount),xlab="Contribution Amount")
dev.off()

#after we have looked at the help on ?ecdf, we see it invokes plot.stepfun we now see better parameters
png('../slides/images/cdf1b.png')
plot(ecdf(rsp$CAmount),xlab="Contribution Amount",main="CDF of contribution amts per cand.",col='black',verticals=T,do.points=F,lwd=2)
dev.off()



png('../slides/images/cdf2.png')
plot(ecdf(rsp$CAmount),xlab="Contribution Amount",main="CDF of contribution amts per cand.",col='black',verticals=T,do.points=F,lwd=2)
plot(ecdf(rsp$CAmount[rsp$Candidate=="barack obama"]),xlab="Contribution Amount",col='red',verticals=T,do.points=F,lwd=2,add=T)
plot(ecdf(rsp$CAmount[rsp$Candidate=="mitt romney"]),xlab="Contribution Amount",col='green',verticals=T,do.points=F,lwd=2,add=T)
plot(ecdf(rsp$CAmount[rsp$Candidate=="newt gingrich"]),xlab="Contribution Amount",col='blue',verticals=T,do.points=F,lwd=2,add=T)
legend("bottomright",c("overall","barack obama","mitt romney", "newt gingrich"),col=c("black","red","green","blue"),lty="solid",lwd=2)
dev.off()


#let's try this again with a loop

#same first line
plot(ecdf(rsp$CAmount),xlab="Contribution Amount",main="CDF of contribution amts per cand.",col='black',verticals=T,do.points=F,lwd=2)
#then we need to loop over colors and candidate names
cnames<-levels(rsp$Candidate)
cols<-rainbow(length(cnames))
for(i in 1:length(cnames)) 
    plot(ecdf(rsp$CAmount[rsp$Candidate==cnames[i]]),col=cols[i],verticals=T,do.points=F,lwd=2,add=T)
#now make the legend
legend("bottomright",c("overall",cnames),col=c("black",cols),lty="solid",lwd=2)



#now a rank-order plot
ac<-read.table('data/allcontMod.csv',header=T,sep=',')
acp<-ac[ac$CAmount>0,]
acs<-sort(acp$CAmount, decreasing=T)
aca<-NULL
for(i in 1:length(acs)) aca<-c(aca,sum(acs[1:i])/sum(acs))
png('../slides/images/rank1.png')
plot(y=aca,x=1:length(acs)/length(acs),xlab="Fraction of Total Super PAC Contributions",ylab="Fraction of Total $ Contributed to Super PACs",lty="solid",type='s',cex=1.5)
dev.off()



#make a function
RankOrderPlot <- function(vec, ...) {
vec<-sort(vec, decreasing = T)
vec2<- NULL
for(i in 1:length(vec)) vec2<-c(vec2,sum(vec[1:i])/sum(vec))
plot(y=vec2, x= 1:length(vec)/length(vec),...)
}

RankOrderPlot(acp$CAmount,xlab="Fraction of Total Super PAC Contributions",ylab="Fraction of Total $ Contributed to Super PACs",lty="solid",type='s',cex=1.5)

#Now create a comparison overlay
RankOrderPlot(rsp$CAmount,xlab="Fraction of Total Regular Contributions",ylab="Fraction of Total $ Regular Contributions",lty="solid",type='s',cex=1.5)

RankOrderLines <- function(vec, ...) {
vec<-sort(vec, decreasing = T)
vec2<- NULL
for(i in 1:length(vec)) vec2<-c(vec2,sum(vec[1:i])/sum(vec))
lines(y=vec2, x= 1:length(vec)/length(vec),...)
}

RankOrderPlot(acp$CAmount,xlab="Fraction of Total Regular Contributions",ylab="Fraction of Total $ Contributed",lty="solid",type='s',cex=1.5)
RankOrderLines(rsp$CAmount[rsp$Candidate=="barack obama"],lty="solid",type='s',cex=1.5,col='red')
RankOrderLines(rsp$CAmount[rsp$Candidate=="mitt romney"],lty="solid",type='s',cex=1.5,col='blue')
RankOrderLines(rsp$CAmount[rsp$Candidate=="newt gingrich"],lty="solid",type='s',cex=1.5,col='green')




#what else: create a rank order plot grouped by categorical variable. Which one? State?
acPos<-ac[ac$CAmount>0,]
stCont<-tapply(acPos$CAmount,acPos$State,sum)

stContSrt<-sort(stCont, decreasing=T)
stContFr<-NULL
for(i in 1:length(stContSrt)) stContFr<-c(stContFr,sum(stContSrt[1:i])/sum(stContSrt))

names(stContSrt)
