source("~/busy-beeway/planners/lmdp/value_functions.R")
source("~/busy-beeway/planners/lmdp/tid_data.R")

compute_log_rl <- function(dat,b_0,b_1,b_2,lambda) {
  vf <- create_vf_tid(b_0,b_1,b_2)
  rl <- 0
  n_sum <- 0 
  for (i in 1:length(dat)) {
    trans <- dat[[i]][[1]]
    n_sum <- n_sum + nrow(trans)
    off_trans <- dat[[i]][[2]]
    
    b_prime <- sum(vf(trans))
    b_count <- 0
    for (n in 1:length(off_trans)) {
      v <- vf(off_trans[[n]])
      z <- exp(-v)
      E <- sum(z)/nrow(off_trans[[n]])
      b_count <- b_count + log(E)
    }
    
    rl <- rl + (b_prime + b_count)
  }
  (1/n_sum)*(rl + lambda*(b_1^2+b_2^2))
}

#B2 with respect to B1 (i.e., L(B2)/L(B1))
compute_2M_rl <- function(dat,B1,B2,lambda=1) {
  exp(compute_log_rl(dat,B1[1],B1[2],B1[3],lambda) - compute_log_rl(dat,B2[1],B2[2],B2[3],lambda))
}

compute_grad_hess <- function(dat,b_0,b_1,b_2,lambda) {
  vf <- create_vf_tid(b_0,b_1,b_2)
  b_0_vf <- create_vf_tid(1,0,0)
  b_1_vf <- create_vf_tid(0,1,0)
  b_2_vf <- create_vf_tid(0,0,1)
  grad <- c(0,0,0)
  hess <- matrix(0,3,3)
  n_sum <- 0
  
  for (i in 1:length(dat)) {
    
    trans <- dat[[i]][[1]]
    off_trans <- dat[[i]][[2]]
    n_sum <- n_sum + nrow(trans)
    b_0_prime <- sum(b_0_vf(trans))
    b_1_prime <- sum(b_1_vf(trans))
    b_2_prime <- sum(b_2_vf(trans))
    
    b_0_count <- 0
    b_1_count <- 0
    b_2_count <- 0
    
    b_0_var <- 0
    b_1_var <- 0
    b_2_var <- 0
    
    b_0_1_cov <- 0
    b_0_2_cov <- 0
    b_1_2_cov <- 0
    for (n in 1:length(off_trans)) {
      v <- vf(off_trans[[n]])
      z <- exp(-v)

      b_0_v <- b_0_vf(off_trans[[n]])
      
      b_1_v <- b_1_vf(off_trans[[n]])
      
      b_2_v <- b_2_vf(off_trans[[n]])
      
      if (any(is.nan(b_0_v))) {
        print(off_trans[[n]])
      }
      
      b_0_v_sq <- b_0_v^2
      b_1_v_sq <- b_1_v^2
      b_2_v_sq <- b_2_v^2
      
      b_0_1_v <- b_0_v * b_1_v
      b_0_2_v <- b_0_v * b_2_v
      b_1_2_v <- b_1_v * b_2_v
      
      pi_bar <- z/sum(z)
      E_0 <- sum(b_0_v*pi_bar)
      E_1 <- sum(b_1_v*pi_bar)
      E_2 <- sum(b_2_v*pi_bar)
     
      b_0_count <- b_0_count + E_0
      b_1_count <- b_1_count + E_1
      b_2_count <- b_2_count + E_2
      
      b_0_var <- b_0_var + (sum(b_0_v_sq*pi_bar) - (E_0)^2)
      b_1_var <- b_1_var + (sum(b_1_v_sq*pi_bar) - (E_1)^2)
      b_2_var <- b_2_var + (sum(b_2_v_sq*pi_bar) - (E_2)^2)
      
      b_0_1_cov <- b_0_1_cov + (sum(b_0_1_v*pi_bar) - (E_0*E_1))
      b_0_2_cov <- b_0_2_cov + (sum(b_0_2_v*pi_bar) - (E_0*E_2))
      b_1_2_cov <- b_1_2_cov + (sum(b_1_2_v*pi_bar) - (E_1*E_2))
    }
    
    
    grad[1] <- grad[1] + (b_0_prime - b_0_count)
    grad[2] <- grad[2] + (b_1_prime - b_1_count)
    grad[3] <- grad[3] + (b_2_prime - b_2_count)
  
    hess[1,1] <- hess[1,1] + b_0_var
    hess[2,2] <- hess[2,2] + b_1_var
    hess[3,3] <- hess[3,3] + b_2_var
    
    hess[1,2] <- hess[1,2] + b_0_1_cov
    hess[1,3] <- hess[1,3] + b_0_2_cov
    hess[2,1] <- hess[2,1] + b_0_1_cov
    hess[2,3] <- hess[2,3] + b_1_2_cov
    hess[3,1] <- hess[3,1] + b_0_2_cov
    hess[3,2] <- hess[3,2] + b_1_2_cov
  }
  
  grad[1] <- (1/n_sum)*(grad[1] + lambda*(2*b_0))
  grad[2] <- (1/n_sum)*(grad[2] + lambda*(2*b_1))
  grad[3] <- (1/n_sum)*(grad[3] + lambda*(2*b_2))
  
  hess[1,1] <- (1/n_sum)*(hess[1,1] + lambda*2)
  hess[2,2] <- (1/n_sum)*(hess[2,2] + lambda*2)
  hess[3,3] <- (1/n_sum)*(hess[3,3] + lambda*2)
  
  hess[1,2] <- (1/n_sum)*(hess[1,2])
  hess[1,3] <- (1/n_sum)*(hess[1,3])
  hess[2,1] <- (1/n_sum)*(hess[2,1])
  hess[2,3] <- (1/n_sum)*(hess[2,3])
  hess[3,1] <- (1/n_sum)*(hess[3,1])
  hess[3,2] <- (1/n_sum)*(hess[3,2])
  list(grad,hess)
}

#newton's method
newton_3_var <- function(dat,guess = NULL,max_iter=10000,tol=0.0001,lambda=1) {
  if (is.null(guess)) {
    if (lambda > 0) {
      B <- rnorm(3,0,1/sqrt(2*lambda))
    }
    else {
      B <- rnorm(3,0,1)
    }
  }
  else {
    B <- guess
  }

  nit <- 0
  res <- compute_grad_hess(dat,B[1],B[2],B[3],lambda)
  grad <- res[[1]]
  hess <- res[[2]]
  
  n_direction <- solve(hess,-grad)
  n_l_sq <- (grad %*% n_direction)[1]
  l_sq <- -n_l_sq/2
  alp <- 0.25
  bta <- 0.5
  t <- 1
  
  while(compute_log_rl(dat,
                       B[1]+t*n_direction[1],
                       B[2]+t*n_direction[2],
                       B[3]+t*n_direction[3],
                       lambda) >
        compute_log_rl(dat,B[1],B[2],B[3],lambda) + alp*t*n_l_sq) {
    t <- bta*t
  }
  
  while (nit < max_iter & l_sq >= tol & !is.nan(l_sq)) {
    B <- B + t*n_direction
    nit <- nit + 1
    res <- compute_grad_hess(dat,B[1],B[2],B[3],lambda)
    grad <- res[[1]]
    hess <- res[[2]]
    n_direction <- solve(hess,-grad)
    n_l_sq <- (grad %*% n_direction)[1]
    l_sq <- -n_l_sq/2
    
    t <- 1
    while(compute_log_rl(dat,
                         B[1]+t*n_direction[1],
                         B[2]+t*n_direction[2],
                         B[3]+t*n_direction[3],
                         lambda) >
          compute_log_rl(dat,B[1],B[2],B[3],lambda) + alp*t*n_l_sq) {
      t <- bta*t
    }
  }
  rl <- compute_log_rl(dat,B[1],B[2],B[3],lambda)
  list(B,rl,l_sq,nit)
}