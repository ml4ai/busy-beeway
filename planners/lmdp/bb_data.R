library("ggplot2")
library("dplyr")
library("tidyr")
source("~/busy-beeway/planners/lmdp/bb_state_space.R")

process_session <- function(D,delT=8,rho=3,sample_size=500) {
  dat <- list()
  
  for (i in 1:nrow(D)) {
    p_df <- D[[i,1]]
    O <- D[[i,2]]
    g <- D[[i,4]]
    if (nrow(p_df) > 1) { 
      s_dat <- create_state_space_data(p_df,g,O,delT,rho,sample_size)
      dat <- rbind(dat,list(s_dat))
    }
  }
  
  dat
}

compute_stats <- function(p_df,g,O,diff=1) {
  goal_distance <- point_dist(p_df$posX,p_df$posY,g[1],g[2])
  
  min_obstacle_distance <- rep(0,nrow(p_df))
  for (t in 1:nrow(p_df)) {
    o_t <- O[which(O$t == (t - 1)),]
    obs_distance <- point_dist(o_t$posX,o_t$posY,p_df$posX[t],p_df$posY[t])
    
    min_dist <- min(obs_distance)
    min_obstacle_distance[t] <- min_dist
  }
  if (diff > 0) {
    data.frame(goal_distance_diff=diff(goal_distance,diff),
              min_obstacle_distance_diff=diff(min_obstacle_distance,diff),t=1:(nrow(p_df)-diff))
  }
  else {
    data.frame(goal_distance=goal_distance,
               min_obstacle_distance=min_obstacle_distance,t=1:nrow(p_df))
  }
}

compute_stats_session <- function(D,diff=1) {
  dat <- list()
  
  for (i in 1:nrow(D)) {
    p_df <- D[[i,1]]
    O <- D[[i,2]]
    g <- D[[i,4]]
    s_dat <- compute_stats(p_df,g,O,diff)
    dat <- rbind(dat,list(s_dat))
  }
  dat
}

plot_stats <- function(s_dat) {
  s_dat %>% pivot_longer(cols=-t,names_to="variable",values_to = "value") %>%
    ggplot(aes(x=t,y=value,colour = variable)) +
    geom_path()
}