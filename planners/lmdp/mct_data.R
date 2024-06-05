source("~/busy-beeway/planners/lmdp/mct_state_space.R")

process_session <- function(d,p_dat,delT=2,bets=c(5,10,15,20,25,30,35,40,45,50)) {
  P <- d[[1]]
  O <- d[[2]]
  s_dat <- create_state_space_data(P,O,bets,delT,p_dat)
  s_dat
}