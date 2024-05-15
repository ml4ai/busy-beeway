source("~/busy-beeway/planners/lmdp/utility.R")
source("~/busy-beeway/planners/lmdp/ioc_state_space.R")

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

sim_session <- function(b1,
                        b2,
                        delT=2,
                        N=10) {
  
  p_dat <- create_p_dat(delT)
  vf1 <- create_vf_mct(b1,b2,b2*25)
  P <- data.frame(bet=0,t=-delT)
  O <- data.frame(outcome=rbinom(1,1,0.5),s_prob=0.5,t=-delT)
  bets <- c(5,10,15,20,25)
  for (i in 1:delT) {
    outcome <- O[i,1]
    s_prob <- O[i,2]
    new_outcome <- rbinom(1,1,s_prob)
    toss <- toss_magic_coin(outcome,s_prob)
    O <- rbind(O,data.frame(outcome=toss[[1]],s_prob=toss[[2]],t=i-delT))
    P <- rbind(P,data.frame(bet=0,t=i-delT))
  }

  for (t in 1:N) {
    new_states <- create_state_space_data_mct(bets,P,O,delT,p_dat,t)
    v <- vf1(new_states)
    z <- exp(-v)
    P <- rbind(P,data.frame(bet=bets[which.max(z)],t=t))
    toss <- toss_magic_coin(O[which(O$t == t-1),1],O[which(O$t == t-1),2])
    O <- rbind(O,data.frame(outcome=toss[[1]],s_prob=toss[[2]],t=t))
  }
  tr <- O$outcome
  tr[which(tr == 0)] <- -1
  
  list(P,O,sum(tr*P$bet))
}