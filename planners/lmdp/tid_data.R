source("~/busy-beeway/planners/lmdp/tid_state_space.R")

process_session <- function(D,grid,t_dat,grid_dist,delT=2) {
  dat <- list()
  for (i in 1:nrow(D)) {
    P <- D[[i,1]]
    O <- D[[i,2]]
    s_dat <- create_state_space_data(P,O,grid,delT,t_dat,grid_dist)
    dat <- rbind(dat,list(s_dat))
  }
  dat
}