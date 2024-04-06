source("~/busy-beeway/planners/lmdp/lmdp.R")

logsumexp <- function(X) {
  c <- max(X)
  c + log(sum(exp(X-c)))
}

rdunif <- function(n,a,b) {
  rg <- a:b
  sample(rg,n,replace = TRUE,prob=rep(1/length(rg),length(rg)))
}

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

#step size in terms of units (e.g., step_size = 1 means each step is a 1 unit)
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
process_run_fixed_delay <- function(p_pos,g,gb,O,obs_st,omin=4.0,omax=16,delay=8,step_size=1) {
  orig_x <- p_pos[1,1]
  orig_y <- p_pos[1,2]
  p_df <- data.frame(cols=(p_pos[,1]-orig_x),rows=(p_pos[,2]-orig_y),t=p_pos[,3])
  gb_tr_idx <- map_global_trajectory(p_df,gb,step_size)
  gx <- g[1] - orig_x
  gy <- g[2] - orig_y
  dat <- list()
  dp <- create_uniform_default_policy(delay)
  c_px <- gb[gb_tr_idx[1],1] 
  c_py <- gb[gb_tr_idx[1],2]
  o_df <- data.frame(cols=(O[,1]-orig_x),rows=(O[,2]-orig_y),angle=O[,3],t=O[,4],id=O[,5])
  c_board <- create_board(c_px,c_py,gx,gy,o_df,obs_st,omin,omax,delay,step_size)
  tr_idx <- c(which(equals_plus(c_board[,1],c_px) & equals_plus(c_board[,2],c_py)))
  for (t in 2:length(gb_tr_idx)) {
    if (length(tr_idx) > delay) {
      dat <- rbind(dat,list(c_board,tr_idx,c(gx,gy),dp))
      c_board <- create_board(c_px,c_py,gx,gy,o_df,obs_st,omin,omax,delay,step_size)
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

compute_log_rl <- function(dat,b_1,b_2,lambda,plot_gb=FALSE) {
  vf <- create_vf(b_1,b_2)
  rl <- 0
  for (d in 1:nrow(dat)) {
    g <- dat[[d,3]]
    board <- dat[[d,1]]
    tr_idx <- dat[[d,2]]
    dp <- dat[[d,4]]
    v <- vf(board,g)
    if (plot_gb) {
      print(plot_game_board(board,tr=tr_idx,fill_data=2,fill_aux = v))
    } 
    z <- exp(-v)
    a <- rep(0,length(z))
    b <- rep(0,length(z))
    
    a_counts <- as.data.frame(table(tr_idx[2:length(tr_idx)]))
    
    b_counts <- as.data.frame(table(tr_idx[1:(length(tr_idx)-1)]))
    
    a[as.numeric(as.character(a_counts[,1]))] <- a_counts[,2]
    
    b[as.numeric(as.character(b_counts[,1]))] <- b_counts[,2]
    rl <- rl + (sum(a*v) + sum(b*log(dp %*% z)))
  }
  rl + lambda*((b_1-1)^2+(b_2-1)^2)
}

#B2 with respect to B1 (i.e., L(B2)/L(B1))
compute_2M_rl <- function(dat,B1,B2,lambda=1) {
  exp(compute_log_rl(dat,B1[1],B1[2],lambda) - compute_log_rl(dat,B2[1],B2[2],lambda))
}

compute_grad_hess <- function(dat,b_1,b_2,lambda) {
  vf <- create_vf(b_1,b_2)
  b_1_vf <- create_vf(1,0)
  b_2_vf <- create_vf(0,1)
  grad <- c(0,0)
  hess <- matrix(0,2,2)
  
  for (d in 1:nrow(dat)) {
    g <- dat[[d,3]]
    board <- dat[[d,1]]
    tr_idx <- dat[[d,2]]
    dp <- dat[[d,4]]
   
    v <- vf(board,g)
    z <- exp(-v)
    
    b_1_v <- b_1_vf(board,g)
    b_2_v <- b_2_vf(board,g)
 
    u <- matrix(0,length(z),length(z))
    G <- dp %*% z
    for(i in 1:length(z)) {
      u[i,which(dp[i,]>0)] <- dp[i,which(dp[i,]>0)]*z[which(dp[i,]>0)]/G[i]
    }
    
    
    a <- rep(0,length(z))
    b <- rep(0,length(z))
    
    a_counts <- as.data.frame(table(tr_idx[2:length(tr_idx)]))
    
    b_counts <- as.data.frame(table(tr_idx[1:(length(tr_idx)-1)]))
    
    a[as.numeric(as.character(a_counts[,1]))] <- a_counts[,2]
    
    b[as.numeric(as.character(b_counts[,1]))] <- b_counts[,2]
    
    b_1_e <- c(b_1_v %*% t(u))
    b_2_e <- c(b_2_v %*% t(u))
    
    grad[1] <- grad[1] + (sum(a*b_1_v) - sum(b*b_1_e))
    grad[2] <- grad[2] + (sum(a*b_2_v) - sum(b*b_2_e))
    
    hess[1,1] <- hess[1,1] +  sum(b*(c(b_1_v^2 %*% t(u))-b_1_e^2))
    hess[2,2] <- hess[2,2] +  sum(b*(c(b_2_v^2 %*% t(u))-b_2_e^2))
    
    off_diag <- sum(b*(c((b_1_v*b_2_v) %*% t(u))-(b_1_e*b_2_e)))
    
    hess[1,2] <- hess[1,2] + off_diag
    hess[2,1] <- hess[2,1] + off_diag
  }
  grad[1] <- grad[1] + lambda*(2*b_1 - 2)
  grad[2] <- grad[2] + lambda*(2*b_2 - 2)
  
  hess[1,1] <- hess[1,1] + lambda*2
  hess[2,2] <- hess[2,2] + lambda*2
  
  hess[1,2] <- hess[1,2]
  hess[2,1] <- hess[2,1]
  list(grad,hess)
}

#newton's method
newton_2_var <- function(dat,guess = NULL,max_iter=10000,tol=0.0001,lambda=1) {
  if (is.null(guess)) {
    B <- rnorm(2,1,1/sqrt(2*lambda))
  }
  else {
    B <- guess
  }
  nit <- 0
  res <- compute_grad_hess(dat,B[1],B[2],lambda)
  grad <- res[[1]]
  hess <- res[[2]]
  n_direction <- solve(hess,-grad)
  l_sq <- ((grad %*% -n_direction)/2)[1]
  while (nit < max_iter & l_sq >= tol) {
    B <- B + n_direction
    nit <- nit + 1
    res <- compute_grad_hess(dat,B[1],B[2],lambda)
    grad <- res[[1]]
    hess <- res[[2]]
    n_direction <- solve(hess,-grad)
    l_sq <- ((grad %*% -n_direction)/2)[1]
  }
  rl <- compute_log_rl(dat,B[1],B[2],lambda)
  list(B,rl)
}

rjmcmc_run <- function(p_pos,g,gb,O) {
  
}

plot_dist <- function(res) {
  ggplot() + 
    geom_point(res,mapping=aes(x=b1,y=b2,color=w)) + 
    scale_color_gradient(low="blue",high="orange")
}
