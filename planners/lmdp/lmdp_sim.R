source("~/busy-beeway/planners/lmdp/lmdp.R")

runif_circle <- function(n,R,center = c(0,0)) {
  r <- R * sqrt(runif(n))
  theta <- runif(n) * 2 * pi 
  if (n == 1) {
    res <- c(center[1] + r * cos_plus(theta),center[2] + r * sin_plus(theta))
  }
  else {
    res <- list(center[1] + r * cos_plus_vec(theta),center[2] + r * sin_plus_vec(theta))
  }
  res
}

run_sim <- function() {
  lives <- 3
  orig_goals <- 7
  pspeed <- 8
  pt <- 1
  ospeeds <- c(4.0,8.0,12.0,16.0)
  oprobs <- c(0.25,0.34,0.25,0.15)
  ptr <- 1/30
  st <- 1/4
  max_time <- 1000
  k <- 50000
  eps <- Inf
  tol <- 0.3
  b1 <- .1
  b2 <- .9
  
  cf1 <- create_cf(b1,b2)
  
  obs_st <- generate_ses(ospeeds,oprobs,k,st,ptr,pt,pspeed)
  
  D <- list()
  d <- NULL
  reset_all <- TRUE
  lives <- lives + 1
  goals <- orig_goals - 1
  while (lives >= 0 & goals > 0) {
    if (reset_all) {
      lives <- lives - 1
      goals <- orig_goals - 1
      p <- runif_circle(1,50)
      O_r <- runif_circle(50,50)
      O <- data.frame(x=O_r[[1]],y=O_r[[2]],h=runif(50,1,360))
      repeat {
        p <- runif_circle(1,50)
        if (all(greater_equals_plus(sqrt((p[1] - O[,1])^2 + (p[2] - O[,2])^2),1))) {
          break
        }
      }
      repeat {
        g_h <- runif(1,1,360)
        g_r <- rnorm(1,30,1)
        g <- c(p[1] + g_r*cos_plus(g_h),p[2] + g_r*sin_plus(g_h))
        if (lesser_equals_plus(sqrt(g[1]^2 + g[2]^2),50)) {
          break
        }
      }
    }
    else {
      goals <- goals - 1
      p <- d[[1]]
      p_t <- p[nrow(p),3]
      p <- c(p[nrow(p),1],p[nrow(p),2])
      O <- d[[2]]
      O <- O[which(O$t == p_t),]
      O <- data.frame(x=O[,1],y=O[,2],h=O[,3])
      repeat {
        g_h <- runif(1,1,360)
        g_r <- rnorm(1,30,1)
        g <- c(p[1] + g_r*cos_plus(g_h),p[2] + g_r*sin_plus(g_h))
        if (lesser_equals_plus(sqrt(g[1]^2 + g[2]^2),50)) {
          break
        }
      }
    }
    d <- LSES_planner(p,g,O,obs_st,eps,cf1,pspeed,ospeeds,oprobs,pt,max_time,ptr,tol)
    reset_all <- !d[[3]]
    d[[4]] <- g
    D <- rbind(D,d)
  }
  D
}

animate_sim <- function(D,g) {
  P <- D[[1]]
  O <- D[[2]]
  ggplot(O,aes(x,y))+
    geom_text(aes(label='W'),color="darkred",size=3) +
    geom_point(data=P,aes(x,y),size=3) +
    geom_text(aes(x=g[1],y=g[2],label='G'),color = "green",size=5) +
    scale_x_continuous(limits = c(-50,50)) +
    scale_y_continuous(limits = c(-50,50)) +
    labs(title = 'Time Step: {frame_time}') +
    transition_time(t)
}