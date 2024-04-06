source("~/busy-beeway/planners/lmdp/lmdp.R")

runif_circle <- function(n,R,center = c(0,0)) {
  r <- R * sqrt(runif(n))
  theta <- runif(n,1,360) 
  if (n == 1) {
    res <- c(center[1] + r * cos_plus(theta),center[2] + r * sin_plus(theta))
  }
  else {
    res <- list(center[1] + r * cos_plus_vec(theta),center[2] + r * sin_plus_vec(theta))
  }
  res
}

run_sim_fixed_delay <- function(b1,b2,sessions,t=200,delay=8,gb_length=50,o_prob=0.5,ub_cp=1) {
  orig_goals <- 7
  
  vf1 <- create_vf(b1,b2)
  
  D <- list()
  p <- c(0,0)
  g <- c(0,30)
  gb <- simulate_global_board(gb_length)
  gb_dim <- length(-gb_length:gb_length)
  for (i in 1:sessions) {
    lives <- 3
    O <- simulate_coll_probs(gb_dim,gb_dim,t,o_prob,ub_cp)
    O[1,which(gb[,1] == p[1] & gb[,2] == p[2])] <- 0
    goals <- orig_goals
    repeat {
      d <- L_planner(p,g,O,gb,vf1,delay)
      D <- rbind(D,d)
      O <- simulate_coll_probs(gb_dim,gb_dim,t,o_prob,ub_cp)
      O[1,which(gb[,1] == p[1] & gb[,2] == p[2])] <- 0
      if (!d[[3]]) {
        lives <- lives - 1
        if (lives < 0) {
          print("You Lost!")
          break
        }
        goals <- orig_goals
      }
      else {
        goals <- goals - 1
        if (goals <= 0) {
          print("You Won!")
          break
        }
      }
    }
  }
  D
}

animate_sim <- function(D,i) {
  P <- D[[i,1]]
  O <- D[[i,2]]
  g <- D[[i,4]]
  ggplot(O,aes(x,y))+
    geom_text(aes(label='W'),color="darkred",size=3) +
    geom_point(data=P,aes(x,y),size=3) +
    geom_text(aes(x=g[1],y=g[2],label='G'),color = "green",size=5) +
    scale_x_continuous(limits = c(-50,50)) +
    scale_y_continuous(limits = c(-50,50)) +
    labs(title = 'Time Step: {frame_time}') +
    transition_time(t)
}