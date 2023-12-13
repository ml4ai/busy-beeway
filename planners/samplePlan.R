

samplePlan = function(plan,
                      moves,
                      gX,gY,aX,aY,
                      sigma,
                      aL,
                      tL=NULL,
                      firstStep=F){
  
  if(length(tL)==0 & !firstStep){
    print("If tL doesn't exist, firstStep is set to T")
    firstStep = T
  }
  
  totalSteps = dim(plan)[1]
  
  #sample obsPlan and add to tL and aL
  for(iT in 2:totalSteps){
    #label current state
    currState = do.call(paste,data.frame(t(plan[iT-1,])))
    #get all possible next states
    possibleStates = cbind(plan[iT-1,1] + moves[,1],
                           plan[iT-1,2] + moves[,2],
                           iT-1)
    #get distance to goal
    dG = sqrt((possibleStates[,1] - gX)^2 + (possibleStates[,2] - gY)^2)
    dA = sqrt((possibleStates[,1] - aX)^2 + (possibleStates[,2] - aY)^2)
    #add noise
    rG = dG + rnorm(length(dG),0,sigma)
    rA = dA + rnorm(length(dA),0,sigma)
    #update aL & tL
    if(iT==2 & firstStep){firstStep=T}else{firstStep=F}
    out_ = update_aL_tL(possibleStates,currState,
                       f = cbind(rG,rA),
                       aL,tL,
                       firstStep = firstStep) 
    aL = out_[[1]]
    tL = out_[[2]]
  }
  
  out = list(aL,tL)
  return(out)
  
}