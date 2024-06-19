source("~/busy-beeway/planners/lmdp/value_functions.R")
source("~/busy-beeway/planners/lmdp/bb_data.R")
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

compute_log_rl <- function(dat,B,lambda) {
  vf <- create_vf_bb(B)
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
  (1/n_sum)*(rl + lambda*sum(B^2))
}

#B2 with respect to B1 (i.e., L(B2)/L(B1))
compute_2M_rl <- function(dat,B1,B2,lambda=1) {
  exp(compute_log_rl(dat,B1,lambda) - compute_log_rl(dat,B2,lambda))
}

compute_grad_hess <- function(dat,B,lambda) {
  vf <- create_vf_bb(B)
  n_params <- length(B)
  B_n_vf <- apply(diag(1,n_params,n_params),1,create_vf_bb)

  grad <- rep(0,n_params)
  hess <- matrix(0,n_params,n_params)
  n_sum <- 0
  
  for (i in 1:length(dat)) {
    
    trans <- dat[[i]][[1]]
    off_trans <- dat[[i]][[2]]
    n_sum <- n_sum + nrow(trans)
    B_n_prime <- c()
    for (b in 1:n_params) {
      B_n_prime <- c(B_n_prime,sum(B_n_vf[[b]](trans)))
    }
    
    B_n_count <- rep(0,n_params)
    
    B_var <- matrix(0,n_params,n_params)
    
    for (n in 1:length(off_trans)) {
      v <- vf(off_trans[[n]])
      z <- exp(-v)
      pi_bar <- z/sum(z)

      for (b in 1:n_params) {
        b_b_v <- B_n_vf[[b]](off_trans[[n]])
        E_b <- sum(b_b_v*pi_bar)
        for (a in 1:n_params) {
          if (a == b) {
            B_var[b,a] <- B_var[b,a] + (sum((b_b_v^2)*pi_bar) - E_b^2)
          }
          else {
            b_a_v <- B_n_vf[[a]](off_trans[[n]])
            E_a <- sum(b_a_v*pi_bar)
            B_var[b,a] <- B_var[b,a] + (sum((b_b_v*b_a_v)*pi_bar) - (E_b*E_a))
          }
        }
        B_n_count[b] <- B_n_count[b] + E_b
      }
    }
    
    grad <- grad + (B_n_prime - B_n_count)
    
    hess <- hess + B_var
  }
  grad <- (1/n_sum)*(grad + lambda*(2*B))
  
  hess <- (1/n_sum)*(hess + diag(lambda*2,n_params,n_params))

  list(grad,hess)
}

#newton's method
newtons_method <- function(dat,n_params,guess = NULL,max_iter=10000,tol=0.0001,lambda=1) {
  if (is.null(guess)) {
    if (lambda > 0) {
      B <- rnorm(n_params,0,1/sqrt(2*lambda))
    }
    else {
      B <- rnorm(n_params,0,1)
    }
  }
  else {
    B <- guess
  }
  nit <- 0
  res <- compute_grad_hess(dat,B,lambda)
  grad <- res[[1]]
  hess <- res[[2]]
  
  n_direction <- solve(hess,-grad)
  n_l_sq <- (grad %*% n_direction)[1]
  l_sq <- -n_l_sq/2
  alp <- 0.25
  bta <- 0.5
  t <- 1
  
  while(compute_log_rl(dat,B+t*n_direction,lambda) > compute_log_rl(dat,B,lambda) + alp*t*n_l_sq) {
    t <- bta*t
  }
  
  while (nit < max_iter & l_sq >= tol & !is.nan(l_sq)) {
    B <- B + t*n_direction
    nit <- nit + 1
    res <- compute_grad_hess(dat,B,lambda)
    grad <- res[[1]]
    hess <- res[[2]]
    n_direction <- solve(hess,-grad)
    n_l_sq <- (grad %*% n_direction)[1]
    l_sq <- -n_l_sq/2
    
    t <- 1
    while(compute_log_rl(dat,B+t*n_direction,lambda) > compute_log_rl(dat,B,lambda) + alp*t*n_l_sq) {
      t <- bta*t
    }
  }
  rl <- compute_log_rl(dat,B,lambda)
  list(B,rl,l_sq,nit)
}

newton_loocv_helper <- function(s,
                                n_params,
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
    sol <- newtons_method(d_train,n_params,guess,max_iter,tol,lambda)
    rl_m <- rl_m + compute_log_rl(d_test,sol[[1]],lambda)
  }
  rl_m <- rl_m/length(dat)
  sol <- newtons_method(dat,n_params,guess,max_iter,tol,lambda)
  df <- data.frame(t(sol[[1]]))
  df$delT <- s_delT[s]
  df$rho <- s_rho[s]
  df$training_rl <- sol[[2]]
  df$mean_loocv_rl <- rl_m
  df
}

#newton's method with loocv sampler over delT and rho. 
#Here delT and rho variables set limits of sampling distributions.
#Tries to use as many cores to parallel process by default. 
#Set cores < 2 to not parallel process.
newton_loocv <- function(D,
                         n_params,
                         obs_st,
                         sample_size=500,
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
  s_rho <- sample(seq(rho[1],rho[2],by=0.5),p_samples,replace=TRUE)
  if (is.null(cores)) {
    cores <- detectCores()
  }
  if (supportsMulticore() & cores > 1) {
    res <- mclapply(1:p_samples,
                    newton_loocv_helper,
                    D=D,
                    n_params=n_params,
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
        sol <- newtons_method(d_train,n_params,guess,max_iter,tol,lambda)
        rl_m <- rl_m + compute_log_rl(d_test,sol[[1]],lambda)
      }
      rl_m <- rl_m/length(dat)
      sol <- newtons_method(dat,n_params,guess,max_iter,tol,lambda)
      df <- data.frame(t(sol[[1]]))
      df$delT <- s_delT[s]
      df$rho <- s_rho[s]
      df$training_rl <- sol[[2]]
      df$mean_loocv_rl <- rl_m
      sols <- rbind(sols,df)
    }
  }
  end_time <- Sys.time()
  time_diff <- end_time - start_time
  print(time_diff)
  sols
}