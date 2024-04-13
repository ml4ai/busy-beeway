source("~/busy-beeway/planners/lmdp/ioc_graph.R")
source("~/busy-beeway/planners/lmdp/ioc_data.R")

logsumexp <- function(X) {
  c <- max(X)
  c + log(sum(exp(X-c)))
}

rdunif <- function(n,a,b) {
  rg <- a:b
  sample(rg,n,replace = TRUE,prob=rep(1/length(rg),length(rg)))
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
