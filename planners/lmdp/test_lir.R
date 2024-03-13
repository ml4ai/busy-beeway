source("~/busy-beeway/planners/lmdp/lmdp.R")

compute_grad <- function(st,v,dp,lambda) {
  z <- exp(-v)
  u <- matrix(0,length(z),length(z))
  G <- dp %*% z
  for(i in 1:length(z)) {
    u[i,which(dp[i,]>0)] <- dp[i,which(dp[i,]>0)]*z[which(dp[i,]>0)]/G[i]
  }
  
  
  a <- rep(0,length(z))
  b <- rep(0,length(z))
  
  a_counts <- as.data.frame(table(st[2:length(st)]))
  
  b_counts <- as.data.frame(table(st[1:(length(st)-1)]))
  
  a[as.numeric(as.character(a_counts[,1]))] <- a_counts[,2]
  
  b[as.numeric(as.character(b_counts[,1]))] <- b_counts[,2]
  
  grad <- a - c(b %*% u) + (2*lambda*(v-1))
  grad
}

compute_NLL <- function(st,v,dp,lambda) {
  z <- exp(-v)
  a <- rep(0,length(z))
  b <- rep(0,length(z))
  
  a_counts <- as.data.frame(table(st[2:length(st)]))
  
  b_counts <- as.data.frame(table(st[1:(length(st)-1)]))
  
  a[as.numeric(as.character(a_counts[,1]))] <- a_counts[,2]
  
  b[as.numeric(as.character(b_counts[,1]))] <- b_counts[,2]
  
  sum(a*v) + sum(b*dp %*% z) + lambda*(sum((v-1)^2))
}

v_gd <- function(st,grid_length,lambda=1,guess = NULL,max_iter=100,tol=0.001) {
  dp <- create_uniform_default_policy(grid_length)
  if (is.null(guess)) {
    v <- runif(nrow(dp))
  }
  else {
    v <- guess
  }
  
  nit <- 0
  N <- length(st) - 1
  grad <- (1/N)*compute_grad(st,v,dp,lambda)
  m <- sum(grad^2)
  con <- sqrt(m)
  
  c_0 <- 0.25
  tau <- 0.5
  t_0 <- -c_0*m
  a <- 2
  while (((1/N)*compute_NLL(st,v,dp,lambda) - (1/N)*compute_NLL(st,v-a*grad,dp,lambda)) < a*t_0) {
    a <- tau*a
  }
  while (nit < max_iter & con > tol) {
    v <- v - a*grad
    nit <- nit + 1
    grad <- (1/N)*compute_grad(st,v,dp,lambda)
    m <- sum(grad^2)
    con <- sqrt(m)
    
    t_0 <- -c_0*m
    a <- 2
    while (((1/N)*compute_NLL(st,v,dp,lambda) - (1/N)*compute_NLL(st,v-a*grad,dp,lambda)) < a*t_0) {
      a <- tau*a
    }
  }
  list(v,(1/N)*compute_NLL(st,v,dp,lambda))
}

compute_costs <- function(v) {
  grid_length <- round((0.5)*(sqrt(2*length(v)-1)-1))
  dp <- create_uniform_default_policy(grid_length)
  z <- exp(-v)
  c(v + log(dp %*% z))
}

compute_oc <- function(v) {
  grid_length <- round((0.5)*(sqrt(2*length(v)-1)-1))
  dp <- create_uniform_default_policy(grid_length)
  z <- exp(-v)
  u <- matrix(0,length(z),length(z))
  G <- dp %*% z
  for(i in 1:length(z)) {
    u[i,which(dp[i,]>0)] <- dp[i,which(dp[i,]>0)]*z[which(dp[i,]>0)]/G[i]
  }
  u
}