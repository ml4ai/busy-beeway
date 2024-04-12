source("~/busy-beeway/planners/lmdp/lmdp.R")

process_simulated_run <- function(p_pos,g,gb,O,delay=8) {
  dat <- list()
  dp <- create_uniform_default_policy(delay)
  current_p <- c(p_pos[1,1],p_pos[1,2])
  c_board <- simulate_board(current_p[1],current_p[2],g[1],g[2],gb,O,1,delay)
  tr_idx <- c(which(equals_plus(c_board[,1],current_p[1]) & equals_plus(c_board[,2],current_p[2])))
  for (t in 2:NROW(p_pos)) {
    if (length(tr_idx) > delay) {
      dat <- rbind(dat,list(c_board,tr_idx,g,dp))
      c_board <- simulate_board(current_p[1],current_p[2],g[1],g[2],gb,O,t,delay)
      tr_idx <- c(which(equals_plus(c_board[,1],current_p[1]) & equals_plus(c_board[,2],current_p[2])))
    }
    current_p <- c(p_pos[t,1],p_pos[t,2])
    tr_idx <- c(tr_idx,which(equals_plus(c_board[,1],current_p[1]) & equals_plus(c_board[,2],current_p[2])))
  }
  if (length(tr_idx) > 0) {
    dat <- rbind(dat,list(c_board,tr_idx,g,dp))
  }
  dat
}

process_simulated_data <- function(D,gb,delay=8) {
  dat <- list()
  for (d in 1:nrow(D)) {
    p_pos <- D[[d,1]]
    O <- D[[d,2]]
    g <- D[[d,4]]
    dat <- rbind(dat,process_simulated_run(p_pos,g,gb,O,delay))
  }
  dat
}

map_global_trajectory <- function(p_df,gb,step_size = 1) {
  half_step <- step_size/2
  c_idx <- which(greater_equals_plus(p_df[1,1],(gb$cols - half_step)) & 
                   lesser_equals_plus(p_df[1,1],(gb$cols + half_step)) &
                   greater_equals_plus(p_df[1,2],(gb$rows - half_step)) & 
                   lesser_equals_plus(p_df[1,2],(gb$rows + half_step)))[1]
  tr_idx <- c(c_idx)
  for (i in 2:nrow(p_df)) {
    idx <- which(greater_equals_plus(p_df[i,1],(gb$cols - half_step)) & 
                   lesser_equals_plus(p_df[i,1],(gb$cols + half_step)) &
                   greater_equals_plus(p_df[i,2],(gb$rows - half_step)) & 
                   lesser_equals_plus(p_df[i,2],(gb$rows + half_step)))[1]
    if (length(idx) > 0) {
      if (idx == c_idx) {
        next
      }
      
      c_idx <- idx
      tr_idx <- c(tr_idx,c_idx)
    }
  }
  tr_idx
}



#step size is in terms of game units and delay is in terms of step size (e.g., delay = 8, delayed by 8 steps)
process_run_fixed_delay <- function(p_pos,g,gb,O,obs_st,omin=4.0,omax=16,delay=8,time_step=1/30,pspeed=4) {
  step_size <- time_step*pspeed
  orig_x <- p_pos[1,1]
  orig_y <- p_pos[1,2]
  p_df <- data.frame(cols=(p_pos[,1]-orig_x),rows=(p_pos[,2]-orig_y),t=p_pos[,3])
  gb_tr_idx <- map_global_trajectory(p_df,gb,step_size)
  print(length(gb_tr_idx))
  o_df <- data.frame(cols=(O[,1]-orig_x),rows=(O[,2]-orig_y),angle=O[,3],t=O[,4])
  gx <- g[1] - orig_x
  gy <- g[2] - orig_y
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

process_obstacle_run_data <- function(o_df,time_step) {
  max_id <- max(o_df$id)
  o_df$seconds <- 0
  df <- NULL
  for (i in 1:max_id) {
    o_i <- o_df[which(o_df$id==i),]
    o_i[,9] <- (o_i[,1] - o_i[,1][1])/1000
    max_s <- max(o_i[,9])
    df <- rbind(df,data.frame(posX=o_i[1,2],posY=o_i[1,3],angle=o_i[1,4],t=0,id=i))
    c_t <- time_step
    t <- 1

    while (c_t <= max_s) {
      l_range <- o_i[which(lesser_equals_plus(o_i[,9],c_t)),]
      u_range <- o_i[which(greater_equals_plus(o_i[,9],c_t)),]
      lb <- l_range[which.max(l_range[,9]),]
      ub <- u_range[which.min(u_range[,9]),]
      if (equals_plus(ub[,9],lb[,9])) {
        x_new <- ub[,2]
        y_new <- ub[,3]
      }
      else {
        rs <- time_step/(ub[,9] - lb[,9])
        irs <- 1-rs
        x_new <- ub[,2]*rs + lb[,2]*irs
        y_new <- ub[,3]*rs + lb[,3]*irs
      }
      df <- rbind(df,data.frame(posX=x_new,posY=y_new,angle=o_i[1,4],t=t,id=i))
      t <- t + 1
      c_t <- c_t + time_step
    }
  }
  df
}