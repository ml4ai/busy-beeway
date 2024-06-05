source("~/busy-beeway/planners/lmdp/utility.R")
source("~/busy-beeway/planners/lmdp/bb_state_space.R")
source("~/busy-beeway/planners/lmdp/value_functions.R")

move_obstacles <- function(O,ospeeds,oprobs) {
  n_obs <- nrow(O)
  spd_samples <- sample(ospeeds/30,n_obs,TRUE,oprobs)
  data.frame(x=O[,1] + spd_samples*cos_plus_vec(O[,3]),
             y=O[,2] + spd_samples*sin_plus_vec(O[,3]),
             h=O[,3],
             t=O[,4]+1,
             id=O[,5])
}

sim_attempt <- function(p,g,O,obs_st,vf1,delT,rho,pspeed,ospeeds,oprobs,state_samples) {
  c_O <- O
  new_O <- move_obstacles(c_O,ospeeds,oprobs)
  P <- data.frame(x=p[1],y=p[2],t=0)
  c_P <- P
  coll <- collision(c_O[,1],c_O[,2],new_O[,1],new_O[,2],c_P[,1],c_P[,2],0.3)
  if (coll[[1]]) {
    O <- rbind(O,data.frame(x=coll[[2]],y=coll[[3]],h=new_O$h,t=new_O$t,id=new_O$id))
    return(list(P,O,FALSE))
  }
  exceeded_bounds <- which(sqrt(new_O[,1]^2 + new_O[,2]^2) > 50)
  new_O[exceeded_bounds,1] <- -c_O[exceeded_bounds,1]
  new_O[exceeded_bounds,2] <- -c_O[exceeded_bounds,2]
  c_O <- new_O
  O <- rbind(O,c_O)
  t <- 1
  ts <- delT + 1
  repeat {
    if (rbinom(1,1,0.9)) {
      p_dist <- pspeed/30
    }
    else {
      p_dist <- rtruncnorm(1,0,pspeed,pspeed,0.1)/30
    }
    p_dist <- min(point_dist(c_P[,1],c_P[,2],g[1],g[2]),p_dist)
    m_n_samps <- runif_on_circle(state_samples,p_dist,c(c_P[,1],c_P[,2]))
    g_heading <- find_direction(c_P[,1],c_P[,2],g[1],g[2])
    g_samp_x <- c_P[,1] + p_dist*cos_plus(g_heading)
    g_samp_y <- c_P[,2] + p_dist*sin_plus(g_heading)

    m_n_samps[[1]] <- c(m_n_samps[[1]],g_samp_x)
    m_n_samps[[2]] <- c(m_n_samps[[2]],g_samp_y)
    rd_goal_samps <- point_dist_sq(m_n_samps[[1]],m_n_samps[[2]],g[1],g[2])
    rd_goal_samps <- (rd_goal_samps - min(rd_goal_samps))/(max(rd_goal_samps) - min(rd_goal_samps))
    m_n_df <- data.frame(x=m_n_samps[[1]],
                         y=m_n_samps[[2]],
                         rd_goal=rd_goal_samps)
    threat_level_samps <- rep(0,state_samples + 1)
    o_t <- (t-1)-delT
    if (o_t >= 0) {
      for (i in 1:length(threat_level_samps)) {
        threat_level_samps[i] <- compute_threat_level(m_n_samps[[1]][i],
                                                      m_n_samps[[2]][i],
                                                      O[which(O$t == o_t),],
                                                      obs_st[ts,1],
                                                      obs_st[ts,2],
                                                      rho)

      }
    }
  
    m_n_df$threat_level <- threat_level_samps
    
    v <- vf1(m_n_df)
    z <- exp(-v)
    max_z <- sample(which(equals_plus(z,max(z))),1)
    new_P <- data.frame(x=m_n_df[max_z,1],y=m_n_df[max_z,2],t=t)
    coll <- collision(c_P[,1],c_P[,2],new_P[,1],new_P[,2],c_O[,1],c_O[,2],0.3)
    if (coll[[1]]) {
      P <- rbind(P,data.frame(x=new_P[,1],y=new_P[,2],t=new_P$t))
      return(list(P,O,FALSE))
    }
    
    coll <- collision(c_P[,1],c_P[,2],new_P[,1],new_P[,2],g[1],g[2],0.3)
    if (coll[[1]]) {
      P <- rbind(P,data.frame(x=coll[[2]],y=coll[[3]],t=new_P$t))
      return(list(P,O,TRUE))
    }
    
    c_P <- new_P
    P <- rbind(P,c_P)
    
    new_O <- move_obstacles(c_O,ospeeds,oprobs)
    coll <- collision(c_O[,1],c_O[,2],new_O[,1],new_O[,2],c_P[,1],c_P[,2],0.3)
    if (coll[[1]]) {
      O <- rbind(O,data.frame(x=coll[[2]],y=coll[[3]],h=new_O$h,t=new_O$t,id=new_O$id))
      return(list(P,O,FALSE))
    }
    exceeded_bounds <- which(sqrt(new_O[,1]^2 + new_O[,2]^2) > 50)
    new_O[exceeded_bounds,1] <- -c_O[exceeded_bounds,1]
    new_O[exceeded_bounds,2] <- -c_O[exceeded_bounds,2]
    c_O <- new_O
    O <- rbind(O,c_O)
    t <- t + 1
  }
}

create_sim_obs_st <- function(ospeeds,oprobs,n_obs,timesteps) {
  O_r <- runif_circle(n_obs,50)
  O <- data.frame(x=O_r[[1]],y=O_r[[2]],h=runif(n_obs,0,360),t=rep(0,n_obs),id=1:n_obs)
  c_O <- O
  for (i in 1:timesteps) {
    new_O <- move_obstacles(c_O,ospeeds,oprobs)
    c_O <- new_O
    O <- rbind(O,c_O)
  }
  generate_obs_ses(O)
}

sim_session <- function(b0,
                        b1,
                        b2,
                        pspeed = 4.0,
                        ospeeds = c(2.0,4.0,6.0,8.0),
                        oprobs = c(0.25,0.34,0.25,0.15),
                        delT = 8,
                        rho = 1.5,
                        state_samples = 500,
                        n_obs = 50) {
  lives <- 3
  orig_goals <- 7
  vf1 <- create_vf_bb(b0,b1,b2)
  obs_st <- create_sim_obs_st(ospeeds,oprobs,n_obs,30)
  D <- list()
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
    d <- sim_attempt(p,
                     g,
                     O,
                     obs_st,
                     vf1,
                     delT,
                     rho,
                     pspeed,
                     ospeeds,
                     oprobs,
                     state_samples)
    reset_all <- !d[[3]]
    d[[4]] <- g
    D <- rbind(D,d)
    if (reset_all) {
      lives <- lives - 1
      if (lives < 0) {
        return(D)
      }
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
    }
    else {
      goals <- goals - 1
      if (goals <= 0) {
        return(D)
      }
      p <- d[[1]]
      p_t <- p[nrow(p),3]
      p <- c(p[nrow(p),1],p[nrow(p),2])
      O <- d[[2]]
      O <- O[which(O$t == p_t),]
      O <- data.frame(x=O[,1],y=O[,2],h=O[,3],t=rep(0,n_obs),id=1:n_obs)
      repeat {
        g_h <- runif(1,0,360)
        g_r <- rnorm(1,30,1)
        g <- c(p[1] + g_r*cos_plus(g_h),p[2] + g_r*sin_plus(g_h))
        if (lesser_equals_plus(sqrt(g[1]^2 + g[2]^2),50)) {
          break
        }
      }
    }
  }
}

run_sim <- function(b0,
                    b1,
                    b2,
                    sessions = 3,
                    pspeed = 4.0,
                    ospeeds = c(2.0,4.0,6.0,8.0),
                    oprobs = c(0.25,0.34,0.25,0.15),
                    delT = 8,
                    rho = 1.5,
                    state_samples = 500,
                    n_obs = 50) {
  dat <- NULL
  for (s in 1:sessions) {
    dat <- rbind(dat,sim_session(b0,
                                 b1,
                                 b2,
                                 pspeed,
                                 ospeeds,
                                 oprobs,
                                 delT,
                                 rho,
                                 state_samples,
                                 n_obs))
  }
  dat
}