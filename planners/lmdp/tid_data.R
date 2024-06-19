source("~/busy-beeway/planners/lmdp/tid_state_space.R")

process_session <- function(D,grid_length=10,t_samps=1000,delT=2) {
  dat <- list()
  grid <- expand.grid(x=1:grid_length, y=1:grid_length)
  t_dat <- create_t_dat(delT,grid,t_samps)
  grid_dist <- as.matrix(dist(matrix(c(grid$x,grid$y),nrow(grid),2),
                              method="manhattan",
                              diag=TRUE,
                              upper=TRUE))
  for (i in 1:nrow(D)) {
    P <- D[[i]][[1]]
    O <- D[[i]][[2]]
    s_dat <- create_state_space_data(P,O,grid,delT,t_dat,grid_dist)
    dat <- rbind(dat,list(s_dat))
  }
  dat
}