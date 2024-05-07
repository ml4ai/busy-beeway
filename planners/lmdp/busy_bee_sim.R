source("~/busy-beeway/planners/lmdp/utility.R")

move_obstacles <- function(O,ospeeds,oprobs) {
  n_obs <- nrow(O)
  spd_samples <- sample(ospeeds/30,n_obs,TRUE,oprobs)
  data.frame(x=O[,1] + spd_samples*cos_plus_vec(O[,3]),
             y=O[,2] + spd_samples*sin_plus_vec(O[,3]),
             h=O[,3],
             t=O[,4]+1,
             id=O[,5])
}

sim_attempt_fixed_delay <- function(p,g,O,obs_st,vf1,pspeed,ospeeds,oprobs) {
  c_O <- O
  new_O <- move_obstacles(old_O,ospeeds,oprobs)
  P <- data.frame(x=p[1],y=p[2],t=0)
  c_P <- P
  coll <- collision
}

run_sim_fixed_delay <- function(b1,
                                b2,
                                sessions,
                                pspeeds = 4.0,
                                ospeeds = c(2.0,4.0,6.0,8.0),
                                oprobs = c(0.25,0.34,0.25,0.15),
                                delT = 8,
                                rho = 1.5,
                                state_samples = 500,
                                n_obs = 50) {
  lives <- 3
  orig_goals <- 7
  vf1 <- create_vf(b1,b2)
  
  D <- list()
  d <- NULL
  goals <- orig_goals
  O_r <- runif_circle(n_obs,50)
  O <- data.frame(x=O_r[[1]],y=O_r[[2]],h=runif(n_obs,0,360),t=rep(0,n_obs),id=1:n_obs)
  repeat {
    p <- runif_circle(1,50)
    if (all(greater_equals_plus(sqrt((p[1] - O[,1])^2 + (p[2] - O[,2])^2),1))) {
      break
    }
  }
  
  repeat {
    g_h <- runif(1,0,360)
    g_r <- rnorm(1,30,1)
    g <- c(p[1] + g_r*cos_plus(g_h),p[2] + g_r*sin_plus(g_h))
    if (lesser_equals_plus(sqrt(g[1]^2 + g[2]^2),50)) {
      break
    }
  }
  
  repeat {
    
  }
}