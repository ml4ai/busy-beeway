source("~/busy-beeway/planners/lmdp/ioc_graph.R")
source("~/busy-beeway/planners/lmdp/ioc_data.R")
library("parallel")
library("parallelly")

logsumexp <- function(X) {
  c <- max(X)
  c + log(sum(exp(X-c)))
}

rdunif <- function(n,a,b) {
  rg <- a:b
  sample(rg,n,replace = TRUE,prob=rep(1/length(rg),length(rg)))
}

compute_log_rl <- function(dat,b_1,b_2,lambda) {
  vf <- create_vf(b_1,b_2)
  rl <- 0
  for (i in 1:length(dat)) {
    trans <- dat[[i]][[1]]
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
  rl + lambda*(b_1^2+b_2^2)
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
  
  for (i in 1:length(dat)) {
  
    trans <- dat[[i]][[1]]
    off_trans <- dat[[i]][[2]]
    
    b_1_prime <- sum(b_1_vf(trans))
    b_2_prime <- sum(b_2_vf(trans))
    
    b_1_count <- 0
    b_2_count <- 0
    
    b_1_var <- 0
    b_2_var <- 0
    
    b_cov <- 0
    for (n in 1:length(off_trans)) {
      v <- vf(off_trans[[n]])
      z <- exp(-v)
      b_1_v <- b_1_vf(off_trans[[n]])
      
      b_2_v <- b_2_vf(off_trans[[n]])
      
      b_1_v_sq <- b_1_v^2
      b_2_v_sq <- b_2_v^2
      
      b_1_2_v <- b_1_v * b_2_v
      
      pi_bar <- z/sum(z)
      E_1 <- sum(b_1_v*pi_bar)
      E_2 <- sum(b_2_v*pi_bar)
      b_1_count <- b_1_count + E_1
      b_2_count <- b_2_count + E_2
      
      b_1_var <- b_1_var + (sum(b_1_v_sq*pi_bar) - (E_1)^2)
      b_2_var <- b_2_var + (sum(b_2_v_sq*pi_bar) - (E_2)^2)
      
      b_cov <- b_cov + (sum(b_1_2_v*pi_bar) - (E_1*E_2))
    }
    
    
    grad[1] <- grad[1] + (b_1_prime - b_1_count)
    grad[2] <- grad[2] + (b_2_prime - b_2_count)
    
    hess[1,1] <- hess[1,1] + b_1_var
    hess[2,2] <- hess[2,2] + b_2_var
    
    hess[1,2] <- hess[1,2] + b_cov
    hess[2,1] <- hess[2,1] + b_cov
  }
  grad[1] <- grad[1] + lambda*(2*b_1)
  grad[2] <- grad[2] + lambda*(2*b_2)
  
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

newton_2_var_loocv_helper <- function(s,
                                      D,
                                      obs_st,
                                      sample_size,
                                      s_delT,
                                      s_rho,
                                      guess,
                                      max_iter,
                                      tol,
                                      lambda) {
  dat <- process_session(D,obs_st,s_delT[s],sample_size,s_rho[s])
  rl_m <- 0
  for (d in 1:length(dat)) {
    d_test <- dat[d]
    d_train <- dat[-d]
    sol <- newton_2_var(d_train,guess,max_iter,tol,lambda)
    rl_m <- rl_m + compute_log_rl(d_test,sol[[1]][1],sol[[1]][2],lambda)
  }
  rl_m <- rl_m/length(dat)
  sol <- newton_2_var(dat,guess,max_iter,tol,lambda)
  data.frame(b1=sol[[1]][1],
             b2=sol[[1]][2],
             delT=s_delT[s],
             rho=s_rho[s],
             training_rl=sol[[2]],
             mean_loocv_rl=rl_m)
}

#newton's method with loocv sampler over delT and rho. 
#Here delT and rho variables set limits of sampling distributions.
#Tries to use as many cores to parallel process by default. 
#Set cores < 2 to not parallel process.
newton_2_var_loocv <- function(D,
                           obs_st,
                           sample_size=100,
                           delT=c(0,10),
                           rho=c(0,10),
                           p_samples=100,
                           guess = NULL,
                           max_iter=10000,
                           tol=0.0001,
                           lambda=1,
                           cores=NULL) {
  
  start_time <- Sys.time()
  sols <- NULL
  s_delT <- sample(delT[1]:delT[2],p_samples,replace=TRUE)
  s_rho <- runif(p_samples,rho[1],rho[2])
  if (is.null(cores)) {
    cores <- detectCores()
  }
  if (supportsMulticore() & cores > 1) {
    res <- mclapply(1:p_samples,
                    newton_2_var_loocv_helper,
                    D=D,
                    obs_st=obs_st,
                    sample_size=sample_size,
                    s_delT=s_delT,
                    s_rho=s_rho,
                    guess=guess,
                    max_iter=max_iter,
                    tol=tol,
                    lambda=lambda,
                    mc.cores = cores)
    sols <- Reduce(rbind,res)
  }
  else {
    for (s in 1:p_samples) {
      dat <- process_session(D,obs_st,s_delT[s],sample_size,s_rho[s])
      rl_m <- 0
      for (d in 1:length(dat)) {
        d_test <- dat[d]
        d_train <- dat[-d]
        sol <- newton_2_var(d_train,guess,max_iter,tol,lambda)
        rl_m <- rl_m + compute_log_rl(d_test,sol[[1]][1],sol[[1]][2],lambda)
      }
      rl_m <- rl_m/length(dat)
      sol <- newton_2_var(dat,guess,max_iter,tol,lambda)
      sols <- rbind(sols,data.frame(b1=sol[[1]][1],
                                    b2=sol[[1]][2],
                                    delT=s_delT[s],
                                    rho=s_rho[s],
                                    training_rl=sol[[2]],
                                    mean_loocv_rl=rl_m))
    }
  }
  end_time <- Sys.time()
  time_diff <- end_time - start_time
  print(time_diff)
  sols
}