source("~/busy-beeway/planners/lmdp/utility.R")
source("~/busy-beeway/planners/lmdp/mct_state_space.R")
source("~/busy-beeway/planners/lmdp/value_functions.R")

toss_magic_coin <- function(outcome,s_prob) {
  new_outcome <- rbinom(1,1,s_prob)
  if (outcome == new_outcome) {
    if (new_outcome) {
      new_s_prob <- s_prob - 0.1
    }
    else {
      new_s_prob <- s_prob + 0.1
    }
  }
  else {
    new_s_prob <- 0.5
  }
  list(new_outcome,new_s_prob)
}

run_sim <- function(b0,
                    b1,
                    b2,
                    delT=2,
                    N=10,
                    bets=c(5,10,15,20,25,30,35,40,45,50)) {
  
  p_dat <- create_p_dat(delT)
  vf1 <- create_vf_mct(b0,b1,b2)
  P <- data.frame(bet=0,t=-delT)
  O <- data.frame(outcome=rbinom(1,1,0.5),s_prob=0.5,t=-delT)
  for (i in 1:delT) {
    outcome <- O[i,1]
    s_prob <- O[i,2]
    new_outcome <- rbinom(1,1,s_prob)
    toss <- toss_magic_coin(outcome,s_prob)
    O <- rbind(O,data.frame(outcome=toss[[1]],s_prob=toss[[2]],t=i-delT))
    P <- rbind(P,data.frame(bet=0,t=i-delT))
  }

  for (t in 1:N) {
    new_states <- create_state_space(bets,P,O,delT,p_dat,t)
    v <- vf1(new_states)
    z <- exp(-v)
    max_z <- sample(which(equals_plus(z,max(z))),1)
    P <- rbind(P,data.frame(bet=bets[max_z],t=t))
    toss <- toss_magic_coin(O[which(O$t == t-1),1],O[which(O$t == t-1),2])
    if (equals_plus(toss[[2]],0)) {
      toss[[2]] <- 0
    }
    if (equals_plus(toss[[2]],1)) {
      toss[[2]] <- 1
    }
    O <- rbind(O,data.frame(outcome=toss[[1]],s_prob=toss[[2]],t=t))
  }
  tr <- O$outcome
  tr[which(tr == 0)] <- -1
  
  list(P,O,sum(tr*P$bet))
}