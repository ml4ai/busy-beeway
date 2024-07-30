source("~/busy-beeway/planners/lmdp/value_functions.R")
source("~/busy-beeway/planners/lmdp/bb_data.R")
source("~/busy-beeway/data/data_processing.R")
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
  
  b_iter <- max_iter
  b_nit <- 0
  while(compute_log_rl(dat,B+t*n_direction,lambda) > (compute_log_rl(dat,B,lambda) + alp*t*n_l_sq) &
        b_nit < b_iter) {
    b_nit <- b_nit + 1
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
    b_nit <- 0
    while(compute_log_rl(dat,B+t*n_direction,lambda) > (compute_log_rl(dat,B,lambda) + alp*t*n_l_sq) &
          b_nit < b_iter) {
      b_nit <- b_nit + 1
      t <- bta*t
    }
  }
  rl <- compute_log_rl(dat,B,lambda)
  list(B,rl,l_sq,nit)
}

newton_loocv_helper <- function(s,
                                n_params,
                                lvl,
                                pth,
                                control,
                                sample_size,
                                p_pairs,
                                guess,
                                max_iter,
                                tol,
                                lambda) {
  delT <- p_pairs[s,1]
  rho <- p_pairs[s,2]
  skip <- p_pairs[s,3]
  if (lvl == 0) {
    D <- load_BB_data(pth,skip,control)
  }
  else {
    D <- load_lvl_data(lvl,pth,skip,control)
  }
  dat <- process_session(D,delT,rho,sample_size)
  if (length(dat) > 1) {
    rl <- c()
    for (d in 1:length(dat)) {
      d_test <- dat[d]
      d_train <- dat[-d]
      sol <- tryCatch({
        newtons_method(d_train,n_params,guess,max_iter,tol,lambda)
      }, error = function(e) {
        NULL
      })
      if (!is.null(sol)) {
        rl <- c(rl,exp(compute_log_rl(d_test,sol[[1]],lambda)))
      }
    }
    rl_m <- mean(rl)
    rl_sd <- sd(rl)
  }
  else {
    rl_m <- NA
    rl_sd <- NA
  }
  sol <- tryCatch({
    newtons_method(dat,n_params,guess,max_iter,tol,lambda)
  }, error = function(e) {
    NULL
  })
  if (!is.null(sol)) {
    if (n_params == 1) {
      df <- data.frame(X1=sol[[1]])
    }
    else {
      df <- data.frame(t(sol[[1]]))
    }
    df$delT <- delT
    df$rho <- rho
    df$skip <- skip
    df$training_rl <- exp(sol[[2]])
    df$mean_loocv_rl <- rl_m
    df$sd_loocv_rl <- rl_sd
  }
  else {
    df <- data.frame(t(rep(NA,n_params)))
    if (n_params == 1) {
      df <- data.frame(X1=NA)
    }
    else {
      df <- data.frame(t(rep(NA,n_params)))
    }
    df$delT <- delT
    df$rho <- rho
    df$skip <- skip
    df$training_rl <- NA
    df$mean_loocv_rl <- NA
    df$sd_loocv_rl <- NA
  }
  df
}

#newton's method with loocv sampler over delT and rho. 
#Here delT and rho variables set limits of sampling distributions.
#Tries to use as many cores to parallel process by default. 
#Set cores < 2 to not parallel process.
newton_loocv <- function(lvl,
                         pth,
                         control,
                         n_params,
                         sample_size=1000,
                         skip=c(0,10),
                         delT=c(0,10),
                         rho=c(0,10),
                         guess = NULL,
                         max_iter=10000,
                         tol=0.0001,
                         lambda=1,
                         cores=NULL) {
  
  start_time <- Sys.time()
  sols <- NULL
  s_delT <- delT[1]:delT[2]
  s_rho <- rho[1]:rho[2]
  s_skip <- skip[1]:skip[2]
  p_pairs <- expand.grid(delT=s_delT,rho=s_rho,skip=s_skip)
  if (is.null(cores)) {
    cores <- detectCores()
  }
  if (supportsMulticore() & cores > 1) {
    res <- mclapply(1:nrow(p_pairs),
                    newton_loocv_helper,
                    n_params=n_params,
                    lvl=lvl,
                    pth=pth,
                    control=control,
                    sample_size=sample_size,
                    p_pairs=p_pairs,
                    guess=guess,
                    max_iter=max_iter,
                    tol=tol,
                    lambda=lambda,
                    mc.cores = cores)
    sols <- Reduce(rbind,res)
  }
  else {
    for (s in 1:nrow(p_pairs)) {
      delT <- p_pairs[s,1]
      rho <- p_pairs[s,2]
      skip <- p_pairs[s,3]
      if (lvl == 0) {
        D <- load_BB_data(pth,skip,control)
      }
      else {
        D <- load_lvl_data(lvl,pth,skip,control)
      }
      dat <- process_session(D,delT,rho,sample_size)
      if (length(dat) > 1) {
        rl <- c()
        for (d in 1:length(dat)) {
          d_test <- dat[d]
          d_train <- dat[-d]
          sol <- tryCatch({
            return(newtons_method(d_train,n_params,guess,max_iter,tol,lambda))
          }, error = function(e) {
            return(NULL)
          })
          if (!is.null(sol)) {
            rl <- c(rl,exp(compute_log_rl(d_test,sol[[1]],lambda)))
          }
        }
        rl_m <- mean(rl)
        rl_sd <- sd(rl)
      }
      else {
        rl_m <- NA
        rl_sd <- NA
      }
      df <- tryCatch({
        sol <- newtons_method(dat,n_params,guess,max_iter,tol,lambda)
        df <- data.frame(t(sol[[1]]))
        df$delT <- delT
        df$rho <- rho
        df$skip <- skip
        df$training_rl <- exp(sol[[2]])
        df$mean_loocv_rl <- rl_m
        df$sd_loocv_rl <- rl_sd
        return(df)
        
      }, error = function(e) {
        df <- data.frame(t(rep(NA,n_params)))
        df$delT <- delT
        df$rho <- rho
        df$skip <- skip
        df$training_rl <- NA
        df$mean_loocv_rl <- NA
        df$sd_loocv_rl <- NA
        return(df)
      })
      sols <- rbind(sols,df)
    }
  }
  end_time <- Sys.time()
  time_diff <- end_time - start_time
  print(time_diff)
  sols
}