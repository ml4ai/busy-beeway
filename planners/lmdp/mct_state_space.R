source("~/busy-beeway/planners/lmdp/utility.R")

create_p_tree <- function(o,s_prob,t_prob,depth,t) {
  if (t == depth) {
    return(list(list(o,t_prob)))
  }
  outcome <- o[length(o)]
  res <- list()
  for (n_o in 0:1) {
    
    if (outcome == n_o) {
      if (n_o) {
        res <- rbind(res,create_p_tree(c(o,n_o),s_prob - 0.1,t_prob*s_prob,depth,t+1))
      }
      else {
        res <- rbind(res,create_p_tree(c(o,n_o),s_prob + 0.1,t_prob*(1.0-s_prob),depth,t+1))
      }
    }
    else {
      if (n_o) {
        res <- rbind(res,create_p_tree(c(o,n_o),0.5,t_prob*s_prob,depth,t+1))
      }
      else {
        res <- rbind(res,create_p_tree(c(o,n_o),0.5,t_prob*(1.0-s_prob),depth,t+1))
      }
    }
  }
  res
}

#For MCT
create_p_dat <- function(delT) {
  head_dat <- list()
  tail_dat <- list()
  depth <- delT + 1
  s_probs <- seq(0,1,by=0.1)
  for (n in s_probs) {
    head_dat <- rbind(head_dat,list(create_p_tree(c(1),n,1,depth,0)))
    tail_dat <- rbind(tail_dat,list(create_p_tree(c(0),n,1,depth,0)))
  }
  list(head_dat,tail_dat)
}

create_state_space <- function(bets,P,O,delT,p_dat,t) {
  o_t <- (t-1) - delT
  O_t <- O[which(O$t == o_t),]
  p_bet <- P[which(P$t > o_t & P$t < t),1]
  if (O_t$outcome) {
    p_dat <- p_dat[[1]]
  }
  else {
    p_dat <- p_dat[[2]]
  }
  
  p_dat <- p_dat[[which(equals_plus(seq(0,1,by=0.1),O_t$s_prob))]]
  b_states <- NULL
  for (b in bets) {
    pos_r <- c()
    pos_p <- c()
    neg_r <- c()
    neg_p <- c()
    for (p in 1:length(p_dat)) {
      tr <- p_dat[[p]][[1]][2:length(p_dat[[p]][[1]])]
      tr[which(tr == 0)] <- -1
      r <- sum(tr * c(p_bet,b))
      if (r < 0) {
        neg_r <- c(neg_r,r)
        neg_p <- c(neg_p,p_dat[[p]][[2]])
      }
      else {
        pos_r <- c(pos_r,r)
        pos_p <- c(pos_p,p_dat[[p]][[2]])
      }
    }
    pos_p <- pos_p/sum(pos_p)
    neg_p <- neg_p/sum(neg_p)
    b_states <- rbind(b_states,data.frame(bet=b,expected_gain=sum(pos_r*pos_p),expected_loss=abs(sum(neg_r*neg_p))))
  }
  b_states
}

create_state_space_data <- function(P,O,bets,delT,p_dat) {
  trans <- NULL
  off_trans <- list()
  for (t in 1:length(which(P$t > 0))) {
    b_states <- create_state_space(bets,P,O,delT,p_dat,t)
    p_bet <- P[which(P$t == t),1]
    p_eg <- b_states[which(b_states$bet == p_bet),2]
    p_el <- b_states[which(b_states$bet == p_bet),3]
    trans <- rbind(trans,data.frame(expected_gain=p_eg,expected_loss=p_el))
    
    off_trans <- rbind(off_trans,list(data.frame(expected_gain=b_states$expected_gain,
                                                 expected_loss=b_states$expected_loss)))
  }
  list(trans,off_trans)
}