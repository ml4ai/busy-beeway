
update_aL_tL = function(possibleStates,
                        currState,
                        f,
                        aL,
                        tL,
                        firstStep){
  
  iT = 2+as.numeric(unlist(strsplit(currState," ")))[3]
  
  #name states and add features
  possibleStates = data.frame(s=do.call(paste,data.frame(possibleStates)))
  possibleStates = cbind(possibleStates,f)
  names(possibleStates) = c("s",paste0("f",1:dim(f)[2]))
  #update aL with new states, if any
  aLAdd = ! possibleStates$s %in% aL$s
  aL = rbind(aL,possibleStates[aLAdd,])
  #update features of aL 
  for(iF in 2:dim(aL)[2]){
    aL[aL$s %in% possibleStates$s,iF] = 
      rowMeans(cbind(aL[aL$s %in% possibleStates$s,iF],
                     possibleStates[,iF]))
  }
  #get idx for current and next states
  currIdx = which(aL$s==currState)
  nextIdx = which(aL$s %in% possibleStates$s)
  #if it's the first step, create tL. Otherwise update
  if(firstStep){
    tL = data.frame(currIdx,
                    nextIdx,
                    count = 1,
                    step = iT-1)
  }else{
    tL_ = data.frame(currIdx,
                     nextIdx,
                     count = 1,
                     step = iT-1)
    #which transitions have been seen before?
    tLAdd = tL_$currIdx %in% tL$currIdx & 
      tL_$nextIdx %in% tL$nextIdx
    #update familiar transitions
    tL$count[tLAdd] = tL$count[tLAdd]+1 
    #add the transitions that haven't been logged
    tL = rbind(tL,tL_[!tLAdd,])
  }
  
  out = list(aL,tL)
  
  return(out)
  
}