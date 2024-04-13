source("~/busy-beeway/planners/lmdp/ioc_graph.R")

#step size is in terms of game units and delay is in terms of step size (e.g., delay = 8, delayed by 8 steps)
process_run_fixed_delay <- function(p_pos,g,gb,O,obs_st,delay=8,sample_size=29) {
  dat <- list()
  dp <- create_uniform_default_policy(delay)
  c_px <- gb[gb_tr_idx[1],1] 
  c_py <- gb[gb_tr_idx[1],2]
  c_board <- create_board(c_px,c_py,gx,gy,o_df[which(o_df$t == 0),],obs_st,omin,omax,delay,time_step,pspeed)
  tr_idx <- c(which(equals_plus(c_board[,1],c_px) & equals_plus(c_board[,2],c_py)))
  for (t in 2:length(gb_tr_idx)) {
    if (length(tr_idx) > delay) {
      dat <- rbind(dat,list(c_board,tr_idx,c(gx,gy),dp))
      c_board <- create_board(c_px,c_py,gx,gy,o_df[which(o_df$t == (t-1)),],obs_st,omin,omax,delay,time_step,pspeed)
      tr_idx <- c(which(equals_plus(c_board[,1],c_px) & equals_plus(c_board[,2],c_py)))
    }
    c_px <- gb[gb_tr_idx[t],1] 
    c_py <- gb[gb_tr_idx[t],2]
    tr_idx <- c(tr_idx,which(equals_plus(c_board[,1],c_px) & equals_plus(c_board[,2],c_py)))
  }
  if (length(tr_idx) > 0) {
    dat <- rbind(dat,list(c_board,tr_idx,c(gx,gy),dp))
  }
  dat
}
