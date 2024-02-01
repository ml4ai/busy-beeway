source("~/busy-beeway/planners/lmdp/lmdp.R")

logsumexp <- function(X) {
  c <- max(X)
  c + log(sum(exp(X-c)))
}

rdunif <- function(n,a,b) {
  rg <- a:b
  sample(rg,n,replace = TRUE,prob=rep(1/length(rg),length(rg)))
}

recover_trajectory <- function(obs,board,H=8,start_i=1) {
  c_idx <- which(greater_equals_plus(obs[start_i,1],(board$cols - 1/2)) & 
                   lesser_equals_plus(obs[start_i,1],(board$cols + 1/2)) &
                   greater_equals_plus(obs[start_i,2],(board$rows - 1/2)) & 
                   lesser_equals_plus(obs[start_i,2],(board$rows + 1/2)))[1]
  tr_idx <- c(c_idx)
  for (i in (start_i+1):nrow(obs)) {
    idx <- which(greater_equals_plus(obs[i,1],(board$cols - 1/2)) & 
                     lesser_equals_plus(obs[i,1],(board$cols + 1/2)) &
                     greater_equals_plus(obs[i,2],(board$rows - 1/2)) & 
                     lesser_equals_plus(obs[i,2],(board$rows + 1/2)))
    if (length(idx) > 0) {
      if (length(idx) > 1) {
        next
      }
      
      if (idx == c_idx) {
        next
      }
      
      c_idx <- idx
      tr_idx <- c(tr_idx,c_idx)
      if (length(tr_idx) == (H + 1)) {
        res <- tr_idx
        j <- i
      }
      if (length(tr_idx) >= (H + 2)) {
        e_range <- j:i
        new_start <- e_range[which.min((board[res[H+1],1] - obs[e_range,1])^2 + (board[res[H+1],2] - obs[e_range,2])^2)]
        return(list(res,new_start))
      }
    }
    else {
      e_range <- j:i
      new_start <- e_range[which.min((board[res[H+1],1] - obs[e_range,1])^2 + (board[res[H+1],2] - obs[e_range,2])^2)]
      return(list(res,new_start))
    }
  }
  list(tr_idx,nrow(obs))
}

process_data <- function(D,obs_st,omin,omax,pt,prate,pdisp = 2) {
  dat <- list()
  a <- pt - pdisp
  b <- pt + pdisp
  for (d in 1:nrow(D)) {
    start_i <- 1
    P <- D[[d,1]]
    O <- D[[d,2]]
    g <- D[[d,4]]
    repeat {
      repeat {
        H <- rdunif(1,a,b)
        if (H > 0) {
          break
        }
      }
      O_c <- O[which(O$t == P[start_i,3]),]
      board <- create_board(P[start_i,1],P[start_i,2],g[1],g[2],O_c,obs_st,omin,omax,H,prate)
      dp <- create_uniform_default_policy(H)
      X <- recover_trajectory(P,board,H,start_i)
      tr_idx <- X[[1]]
      start_i <- X[[2]]
      dat <- rbind(dat,list(board,tr_idx,g,dp,length(tr_idx)))
      if (start_i == nrow(P)) {
        break
      }
    }
  }
  dat
}

v_log_likelihood <- function(tr_idx,dp,cf,board,g) {
  v <- cf(board,g)
  z <- exp(-v)
  a_sum <- sum(v[tr_idx[2:length(tr_idx)]])
  b_sum <- sum(log(dp[tr_idx[1:(length(tr_idx)-1)],] %*% z))
  -(a_sum + b_sum)
}

l_pf <- function(dat,k=100) {
  start <- Sys.time()
  B1 <- rep(1,k)
  B2 <- runif(k,8,10)
  W <- rep(1,k)
  for (i in 1:k) {
    per <- ((i/k)*100)
    if (per %% 10 == 0) {
      print(sprintf("%i%% done!",as.integer(per)))
    }
    cf <- create_cf(B1[i],B2[i])
    for (d in 1:nrow(dat)) {
      g <- dat[[d,3]]
      board <- dat[[d,1]]
      tr_idx <- dat[[d,2]]
      dp <- dat[[d,4]]
      W[i] <- W[i]+v_log_likelihood(tr_idx,dp,cf,board,g)
    }
  }
  print("Finished!")
  W <- exp(W-logsumexp(W))
  print(Sys.time() - start)
  data.frame(w=W,b1=B1,b2=B2)
}

compute_neg_log_like <- function(dat,b_1,b_2) {
  cf <- create_cf(b_1,b_2)
  neg_log_like <- 0
  for (d in 1:nrow(dat)) {
    g <- dat[[d,3]]
    board <- dat[[d,1]]
    tr_idx <- dat[[d,2]]
    dp <- dat[[d,4]]
    v <- cf(board,g)
    z <- exp(-v)
    
    a_sum <- sum(v[tr_idx[2:length(tr_idx)]])
    
    b_sum <- sum(log(dp[tr_idx[1:(length(tr_idx)-1)],] %*% z))
    neg_log_like <- neg_log_like + (a_sum+b_sum)
  }
  neg_log_like
}

compute_grad_hess <- function(dat,b_1,b_2) {
  cf <- create_cf(b_1,b_2)
  b_1_cf <- create_cf(1,0)
  b_2_cf <- create_cf(0,1)
  grad <- c(0,0)
  hess <- matrix(0,2,2)
  
  for (d in 1:nrow(dat)) {
    g <- dat[[d,3]]
    board <- dat[[d,1]]
    tr_idx <- dat[[d,2]]
    dp <- dat[[d,4]]
   
    v <- cf(board,g)
    z <- exp(-v)
    
    b_1_v <- b_1_cf(board,g)
    b_2_v <- b_2_cf(board,g)
 
    b_1_g_a_sum <- sum(b_1_v[tr_idx[2:length(tr_idx)]])
    b_2_g_a_sum <- sum(b_2_v[tr_idx[2:length(tr_idx)]])
    
    e_v <- dp[tr_idx[1:(length(tr_idx)-1)],] %*% z
    e_v_sq <- e_v^2
    
    b_1_e_v <- dp[tr_idx[1:(length(tr_idx)-1)],] %*% (-b_1_v*z)
    b_2_e_v <- dp[tr_idx[1:(length(tr_idx)-1)],] %*% (-b_2_v*z)
    
    b_1_b_1_e_v <- dp[tr_idx[1:(length(tr_idx)-1)],] %*% ((b_1_v^2)*z)
    b_2_b_2_e_v <- dp[tr_idx[1:(length(tr_idx)-1)],] %*% ((b_2_v^2)*z)
    b_1_b_2_e_v <- dp[tr_idx[1:(length(tr_idx)-1)],] %*% (b_1_v*b_2_v*z)
    
    grad[1] <- grad[1] + b_1_g_a_sum + sum(b_1_e_v/e_v)
    grad[2] <- grad[2] + b_2_g_a_sum + sum(b_2_e_v/e_v)
    
    hess[1,1] <- hess[1,1] + sum((b_1_b_1_e_v*e_v - b_1_e_v^2)/e_v_sq)
    hess[2,2] <- hess[2,2] + sum((b_2_b_2_e_v*e_v - b_2_e_v^2)/e_v_sq)
    
    off_diag_h <- sum((b_1_b_2_e_v*e_v - b_1_e_v*b_2_e_v)/e_v_sq)
    
    hess[1,2] <- hess[1,2] + off_diag_h
    hess[2,1] <- hess[2,1] + off_diag_h
  }
  list(grad,hess)
}

#newton's method
l_mle <- function(dat,guess = NULL,max_iter=100,tol=0.001) {
  if (is.null(guess)) {
    B <- runif(2,0,10)
  }
  else {
    B <- guess
  }
  nit <- 0
  res <- compute_grad_hess(dat,B[1],B[2])
  grad <- res[[1]]
  hess <- res[[2]]
  n_direction <- solve(hess,-grad)
  lambda_sq <- ((grad %*% -n_direction)/2)[1]
  while (nit < max_iter & lambda_sq >= tol) {
    B <- B + n_direction
    nit <- nit + 1
    res <- compute_grad_hess(dat,B[1],B[2])
    grad <- res[[1]]
    hess <- res[[2]]
    n_direction <- solve(hess,-grad)
    lambda_sq <- ((grad %*% -n_direction)/2)[1]
  }
  nll <- compute_neg_log_like(dat,B[1],B[2])
  list(B,nll)
}

#Univariate model
compute_divs <- function(dat,b) {
  cf <- create_cf(1,b)
  b_cf <- create_cf(0,1)
  div <- 0
  second_div <- 0

  for (d in 1:nrow(dat)) {
    g <- dat[[d,3]]
    board <- dat[[d,1]]
    tr_idx <- dat[[d,2]]
    dp <- dat[[d,4]]
    
    v <- cf(board,g)
    z <- exp(-v)
    b_v <- b_cf(board,g)
    
    b_a_sum <- sum(b_v[tr_idx[2:length(tr_idx)]])
    
    e_v <- dp[tr_idx[1:(length(tr_idx)-1)],] %*% z
    e_v_sq <- e_v^2
    
    b_e_v <- dp[tr_idx[1:(length(tr_idx)-1)],] %*% (-b_v*z)
    
    b_sq_e_v <- dp[tr_idx[1:(length(tr_idx)-1)],] %*% ((b_v^2)*z)
    
    div <- div + b_a_sum + sum(b_e_v/e_v)
    
    second_div <- second_div + sum((b_sq_e_v*e_v - b_e_v^2)/e_v_sq)
  }
  list(div,second_div)
}

infer_cost_mle <- function(D,obs_st,omin,omax,pt,prate=1/8,pdisp=2,guess=NULL,max_iter=100,tol=0.001,k=100) {
  if (is.null(guess)) {
    guess <- runif(2,0,10)
  }
  b_1 <- rep(0,k)
  b_2 <- rep(0,k)
  W <- rep(1,k)
  lens <- list()
  for (i in 1:k) {
    dat <- process_data(D,obs_st,omin,omax,pt,prate,pdisp)
    res <- l_mle(dat,guess,max_iter,tol)
    b_1[i] <- res[[1]][1]
    b_2[i] <- res[[1]][2]
    W[i] <- -res[[2]]
    l <- rep(0,nrow(dat))
    for (d in 1:nrow(dat)) {
      l[d] <- dat[[d,5]]
    }
    lens[[i]] <- l
  }
  W <- exp(W-logsumexp(W))
  list(data.frame(w=W,b1=b_1,b2=b_2),lens)
}

#newton's method
l_mle_uni <- function(dat,guess = NULL,max_iter=100,tol=0.001) {
  if (is.null(guess)) {
    b <- runif(1,0,10)
  }
  else {
    b <- guess
  }
  nit <- 0
  res <- compute_divs(dat,b)
  div <- res[[1]]
  second_div <- res[[2]]
  n_direction <- -(div/second_div)
  lambda_sq <- (div * -n_direction)/2
  while (nit < max_iter & lambda_sq >= tol) {
    b <- b + n_direction
    nit <- nit + 1
    res <- compute_divs(dat,b)
    div <- res[[1]]
    second_div <- res[[2]]
    n_direction <- -(div/second_div)
    lambda_sq <- (div * -n_direction)/2
  }
  nll <- compute_neg_log_like(dat,1,b)
  c(b,nll)
}

get_max_W <- function(res) {
  res[which(equals_plus(res$w,max(res$w))),]
}

plot_dist <- function(res) {
  ggplot() + 
    geom_point(res,mapping=aes(x=b1,y=b2,color=w)) + 
    scale_color_gradient(low="blue",high="orange")
}
