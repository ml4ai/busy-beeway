source("~/busy-beeway/planners/lmdp/ioc_state_space.R")

process_session <- function(D,obs_st,delT=3,sample_size=100,rho=0.3) {
  dat <- list()
  
  for (i in 1:nrow(D)) {
    p_df <- D[[i,1]]
    O <- D[[i,2]]
    g <- D[[i,4]]
    s_dat <- create_state_space_data(p_df,g,O,obs_st,delT,sample_size,rho)
    dat <- rbind(dat,list(s_dat))
  }
  
  dat
}
