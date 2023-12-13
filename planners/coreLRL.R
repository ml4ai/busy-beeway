

#Follow core lrl from Piray
#https://github.com/payampiray/LinearRL/blob/master/core_lrl.m




coreLRL = function(dT,
                   rA,
                   states,
                   terminalStates,
                   lambda=1){
  #Inputs
  # dT = one step transition probability under default (from the adjancency mat)
  # rA = the reward vector for all states (-1 for non-terminal)
  # states = list of all state names
  # terminalStates = list of all terminal states
  #Outputs
  # piStar = optimal policy (pii in piray)
  # z = desirability, exp(v), where v is value function (expv in piray)
  # M = the default representation of non-terminal states

  #index the terminal states using the list of terminal state names
  termStateIdX = which(states %in% terminalStates)
  nonTermStateIdx = which(! states %in% terminalStates)
  
  #Inversion
  D = solve(diag(exp(-1*rA/lambda))-dT)
  #M is the subblock of D for only non-terminal states
  M = D[nonTermStateIdx,nonTermStateIdx]
  #P is dT from non-terminal to terminal only
  P = dT[nonTermStateIdx,termStateIdX]
  #exponentiation of r for terminal states
  expr = exp(rA[termStateIdX]/lambda)
  #translate to z
  z_N = M %*% P %*% expr 
  z = rep(0,length(rA))
  z[nonTermStateIdx] = z_N
  z[termStateIdX] = exp(rA[termStateIdX]/lambda)
    
  #get the norm term: default(all x_|x)*z(all x_)
  G = dT %*% z 
  
  piStar = matrix(0,length(states),length(states))
  for(i in 1:length(states)){
    piStar[i,which(dT[i,]>0)] = dT[i,which(dT[i,]>0)]*z[which(dT[i,]>0)]/G[i]
  }
  #matrix version in matlab
  #zg = t(z) / G #transpose breaks it?
  #pii = dT*zg
  
  return(list(piStar,z,M,dT))
  
}

