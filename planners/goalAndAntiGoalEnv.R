
#9 by 9
#goal and anti goal
#reward is b1*distance_goal - b2*dist_antigoal
#point is to try to use inverse LRL to infer b1, b2


#---------------------------------------

library(ggplot2)

getR = function(allPositions,
                b1,b2,
                gX,gY,
                aX,aY){
  
  distGoal = distAnti = rep(NA,dim(allPositions)[1])
  for(i in 1:dim(allPositions)[1]){
    distGoal[i] = sqrt((allPositions[i,"x"]-gX)^2 + 
                         (allPositions[i,"y"]-gY)^2)
    distAnti[i] = sqrt((allPositions[i,"x"]-aX)^2 + 
                         (allPositions[i,"y"]-aY)^2)
  }
  f1 =  1/(1 + distGoal)
  f2 =  1/(1 + distAnti)
  R = b1*f1 + b2*f2
  return(R)
}

getOppositePlan = function(obsPlan){
  totalSteps = dim(obsPlan)[1]
  oppPlan = obsPlan * NA
  oppPlan[1,] = obsPlan[1,]
  for(iT in 2:totalSteps){
    
    obsMove = c(obsPlan[iT,1] - obsPlan[iT-1,1],
                obsPlan[iT,2] - obsPlan[iT-1,2])
    oppMove = -1*obsMove
    oppPlan[iT,] = c(oppPlan[iT-1,1] + oppMove[1],
                     oppPlan[iT-1,2] + oppMove[2],
                     iT-1)
  }
  return(oppPlan)
}

#mainDir = "C:/Users/evan.c.carter/Documents/SIPS_LRL/localGridPlanning/monteCarloPlanSearch/proofOfConcept/"
source("samplePlan.R")
source("update_aL_tL.R")
source("coreLRL.R")

#-------------------------------------------

#get a ground truth trajectory
b1 = -1.5
b2 = 1
nRep = 25

totalSteps = 5 #assumes step 1 is start
x = 0:8
y = 0:8
gX = 0
gY = 0
aX = 3 
aY = 1

allPositions = expand.grid(x,y)
names(allPositions) = c("x","y")
R = getR(allPositions,b1,b2,gX,gY,aX,aY)

#check it
toPlot = cbind(allPositions,R)
ggplot(toPlot,aes(x,y,fill=R)) + 
  theme_bw() + 
  geom_tile(color="black")

#get ground truth trajectory
moves = expand.grid(data.frame(xChange=c(-1,0,1),
                               yChange=c(-1,0,1)))
dontKeep = moves$xChange==0 & moves$yChange==0
moves = moves[!dontKeep,]
obsPlan = matrix(NA,totalSteps,3)
startState = c(4,4,0) 
obsPlan[1,] = startState 
for(iT in 2:totalSteps){
  
  possible = data.frame(x = moves$xChange + obsPlan[iT-1,1],
                        y = moves$yChange + obsPlan[iT-1,2])
  
  possible$currR = getR(possible,b1,b2,gX,gY,aX,aY)
  currMove = moves[which.min(possible$currR),]
  
  obsPlan[iT,] = c(obsPlan[iT-1,1] + currMove$xChange,
                   obsPlan[iT-1,2] + currMove$yChange,
                   iT-1)
}
obsPlan


#---------------------------------------

#what I really want to know is how many times and with how much
# variation do I need to sample the environment
#Do nSim as the number of plans in addition to obs and opp
# that have some randomness added to them
#Do nMod as the number of steps, starting with the last, in the 
# nSim plans that have randomness added
#Note that randomness isn't added using softmax. The softmax
# requires some reward function and none of these modified plans
# have access to a good estimate of the reward function, though
# it could be done by updating R at each step. What would that 
# mean for the default dynamics?

nSim_ = c(1,5,10,50,100)
sigma_ = c(.1,1,10)  #noise in feature sampling 
modSteps_ = 1:(totalSteps-1)  #num steps, starting from last, that get modified 
params = expand.grid(nSim_,sigma_,modSteps_)
names(params) = c("nSim","sigma","modSteps")

b1_hat = matrix(NA,nRep,dim(params)[1]) #ML estimate of b1
b2_hat = matrix(NA,nRep,dim(params)[1]) #ML estimate of b2
maxL = matrix(NA,nRep,dim(params)[1]) #likelihood associated with estimate

candidate_bs = expand.grid(seq(-4,4,.5),seq(-4,4,.5))
names(candidate_bs) = c("b1","b2")
#---------------------------------------

oppPlan = getOppositePlan(obsPlan)

for(iParam in 1:dim(params)[1]){
  
  print(paste0("Running param set ",iParam," of ",dim(params)[1]))
  
  nSim = params$nSim[iParam] 
  sigma = params$sigma[iParam] 
  modSteps= params$modSteps[iParam] 
  
  #start repetition loop
  for(iRep in 1:nRep){
    
    #initialize aL
    aL = data.frame(s=do.call(paste,data.frame(t(startState))),
                    f1=sqrt((startState[1] - gX)^2 + (startState[2] - gY)^2),
                    f2=sqrt((startState[1] - aX)^2 + (startState[2] - aY)^2))
    
    #update aL and tL by sampling obsPlan
    out = samplePlan(obsPlan,
                     moves,
                     gX,gY,aX,aY,
                     sigma,
                     aL,tL=NULL,firstStep=T)
    
    #update aL and tL by sampling oppPlan
    out = samplePlan(oppPlan,
                     moves,
                     gX,gY,aX,aY,
                     sigma,
                     aL = out[[1]],
                     tL = out[[2]],
                     firstStep=F)
    
    
    #begin sampling modified plans based on sim params
    if(nSim>0){
      toReplace = sort(totalSteps - (0:(modSteps-1)),decreasing = F)
      for(iSim in 1:nSim){
        #initialize to-be-modified plan 
        obsPlan_ = obsPlan
        obsPlan_[toReplace,] = NA*obsPlan_[toReplace,]
        #get modification of observed plan
        for(iReplace in 1:length(toReplace)){
          currState = obsPlan_[toReplace[iReplace]-1,]
          possibleStates = cbind(currState[1] + moves[,1],
                                 currState[2] + moves[,2],
                                 currState[3]+1)
          obsPlan_[toReplace[iReplace],] = 
            possibleStates[sample(1:dim(possibleStates)[1],1),]
        }
        #get opposite of modified observed plan
        oppPlan_ = getOppositePlan(obsPlan_)
        #update aL and tL by sampling obsPlan_ and oppPlan_ 
        out = samplePlan(obsPlan_,
                         moves,
                         gX,gY,aX,aY,
                         sigma,
                         aL = out[[1]],
                         tL = out[[2]],
                         firstStep=F)
        out = samplePlan(oppPlan_,
                         moves,
                         gX,gY,aX,aY,
                         sigma,
                         aL = out[[1]],
                         tL = out[[2]],
                         firstStep=F)
      }#end iSim loop
    }#end if()
    
    aL = out[[1]]
    tL = out[[2]]
    
    obserdT = rlist::list.rbind(lapply(strsplit(aL$s," "),as.numeric))[,3]
    
    names(tL) = c("i","j","x","step")
    adjMat = Matrix::sparseMatrix(i = tL$i,
                                  j = tL$j,
                                  x = tL$x,
                                  dims=c(length(aL$s),length(aL$s)))
    
    dT = adjMat * 1/Matrix::rowSums(adjMat)
    dT[is.nan(dT)] = 0
    
    candidate_ls = rep(NA,dim(candidate_bs)[1])
    for(iB in 1:dim(candidate_bs)[1]){
      rA = candidate_bs$b1[iB]*aL$f1 + candidate_bs$b2[iB]*aL$f2
      #normalize rA so that the highest value is 0 and the rest is negative
      rA = (rA - min(rA))/(max(rA)-min(rA))-1
      #get piStar
      lrl = coreLRL(dT,
                    rA = rA,
                    states = aL$s,
                    terminalStates = aL$s[obserdT==max(obserdT)],
                    lambda=1)
      piStar = lrl[[1]]  
      
      obsStates = do.call(paste,data.frame(obsPlan))
      l = rep(NA,length(obsStates)-1)
      for(i in 1:(length(obsStates)-1)){
        l[i] = piStar[which(aL$s==obsStates[i]),which(aL$s==obsStates[i+1])]
      }
      candidate_ls[iB] = prod(l)
    }
    
    #get bs that max candidate_ls
    maxL[iRep,iParam] = candidate_ls[which.max(candidate_ls)]
    b1_hat[iRep,iParam] = candidate_bs[which.max(candidate_ls),"b1"]
    b2_hat[iRep,iParam] = candidate_bs[which.max(candidate_ls),"b2"]
    
    
  }#end iRep loop
}#end iParam loop

save(file=paste0(mainDir,"maxL.R"),maxL)
save(file=paste0(mainDir,"b1_hat.R"),b1_hat)
save(file=paste0(mainDir,"b2_hat.R"),b2_hat)

dim(maxL)
b1_error = b1 - b1_hat
b2_error = b2 - b2_hat

toPlot = data.frame(meanMaxL = colMeans(maxL,na.rm = T),
                    merr_b1 = colMeans(b1_error,na.rm=T),
                    merr_b2 = colMeans(b2_error,na.rm=T),
                    rmse_b1 = sqrt(colMeans(b1_error,na.rm=T)^2),
                    rmse_b2 = sqrt(colMeans(b2_error,na.rm=T)^2))
toPlot = cbind(toPlot,params)

#plot the means
ggplot(toPlot,aes(x=modSteps,y=merr_b1)) + 
  geom_line() + 
  facet_grid(sigma ~ nSim) + 
  geom_hline(yintercept = 0) + 
  theme_bw()
ggplot(toPlot,aes(x=modSteps,y=merr_b2)) + 
  geom_line() + 
  facet_grid(sigma ~ nSim) + 
  geom_hline(yintercept = 0) + 
  theme_bw()
ggplot(toPlot,aes(x=modSteps,y=rmse_b1)) + 
  geom_line() + 
  facet_grid(sigma ~ nSim) + 
  geom_hline(yintercept = 0) + 
  theme_bw()
ggplot(toPlot,aes(x=modSteps,y=rmse_b2)) + 
  geom_line() + 
  facet_grid(sigma ~ nSim) + 
  geom_hline(yintercept = 0) + 
  theme_bw()



#plot with iRep instances represented separately
toPlot = data.frame(b1_err = c(b1_error), #stacks columns
                    b2_err = c(b2_error),
                    iRep=rep(1:100,60),
                    nSim = c(rep(t(params$nSim),100)),
                    sigma = c(rep(t(params$sigma),100)),
                    modSteps = c(rep(t(params$modSteps),100)))

ggplot(toPlot,aes(x=factor(modSteps),y=b1_err)) + 
  #geom_jitter() +
  geom_boxplot() +
  facet_grid(sigma ~ nSim) + 
  theme_bw() + 
  geom_hline(yintercept = 0,color="red")


ggplot(toPlot,aes(x=factor(modSteps),y=b2_err)) + 
  #geom_jitter() +
  geom_boxplot() +
  facet_grid(sigma ~ nSim) + 
  theme_bw() + 
  geom_hline(yintercept = 0,color="red")


toPlot[toPlot$iRep==1 & 
         toPlot$nSim==0 & 
         toPlot$modSteps==1 & 
         toPlot$sigma==1,]



ggplot(toPlot,aes(b1_hat,b2_hat,fill=ml)) + 
  geom_tile(color="black")

ggplot(toPlot,aes(x=sigma,y=b1rse,
                  color=factor(nSim))) + 
  geom_point() + 
  facet_grid(. ~ modSteps)

toPlot[toPlot$b1rse==0,]


#plot estimation error by params
#do an original reward function that makes sense given 
# the reward requirements of 0 being max



