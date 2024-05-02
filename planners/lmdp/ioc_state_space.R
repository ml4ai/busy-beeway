library("igraph")
source("~/busy-beeway/planners/lmdp/utility.R")

compute_threat_level <- function(px,py,O,mu,sig,tol=0.3) {
  Ux <- px - O[,1]
  Uy <- py - O[,2]
  
  Vx <- O[,1] + cos_plus_vec(O[,3])
  Vy <- O[,2] + sin_plus_vec(O[,3])
  
  UV <- Ux*Vx + Uy*Vy
  
  U_1x <- UV*Vx
  U_1y <- UV*Vy
  
  U_2x <- Ux - U_1x
  U_2y <- Uy - U_1y
  
  d <- sqrt(U_2x^2 + U_2y^2)
  
  int_idx <- which(d < tol)
  
  if (length(int_idx) > 0) {
    print("test")
  }
  
  m <- sqrt(tol^2-d[int_idx]^2)
  
  W_1x <- U_1x[int_idx] + m*Vx[int_idx]
  
  W_1y <- U_1y[int_idx] + m*Vy[int_idx]
  
  W_2x <- U_1x[int_idx] - m*Vx[int_idx]
  
  W_2y <- U_1y[int_idx] - m*Vy[int_idx]
  
  P_1x <- O[int_idx,1] + W_1x
  
  P_1y <- O[int_idx,2] + W_1y
  
  P_2x <- O[int_idx,1] + W_2x
  
  P_2y <- O[int_idx,2] + W_2y
  
  dist1 <- sqrt(W_2x^2 + W_2y^2)
  
  dist2 <- sqrt(W_1x^2 + W_1y^2)
  
  p <- rep(0,length(dist1))
  
  lidx <- which(dist1 < dist2)
  uidx <- which(dist1 > dist2)
  
  p[lidx] <- (pnorm(dist2[lidx],mu,sig) - pnorm(dist1[lidx],mu,sig))
  
  p[uidx] <- (pnorm(dist1[uidx],mu,sig) - pnorm(dist2[uidx],mu,sig))
  
  sum(p)
}

create_state_graph <- function(size,sample_size=29) {
  n <- sample_size + 1
  g <- make_star(n,mode="out")
  for (i in 1:(size-1)) {
    g <- g + make_star(n,mode="out")
    g <- add_edges(g,c(n*(i - 1) + 1,n*(i) + 1))
  }
  g <- add_vertices(g,1)
  g <- add_edges(g,c(n*(size-1) + 1,n*(size) + 1))
  g
}

create_passive_dynamics <- function(size,sample_size=29) {
  g <- create_state_graph(size,sample_size)
  m <- as_adjacency_matrix(g,sparse=FALSE)
  m <- m/rowSums(m)
  diag(m)[which(is.nan(diag(m)))] <- 1.0
  m[which(is.nan(m))] <- 0.0
  m
}

create_state_space <- function(p_df,gx,gy,O,obs_st,size,sample_size=29,tol=0.3) {
  g <- create_state_graph(size,sample_size)
  n <- sample_size + 1
  t <- 0
  c_radius <- point_dist(p_df[1,1],p_df[1,2],p_df[2,1],p_df[2,2])
  for (i in 1:length(V(g))) {
    if (i == (n*(t) + 1)) {
      V(g)$x[i] <- p_df[t+1,1]
      V(g)$y[i] <- p_df[t+1,2]
      V(g)$rd_goal[i] <- (V(g)$x[i] - gx)^2 + (V(g)$y[i] - gy)^2
      if (t == 0) {
        V(g)$threat_level[i] <- 0
      }
      else {
        V(g)$threat_level[i] <- compute_threat_level(V(g)$x[i],V(g)$y[i],O,obs_st[t,1],obs_st[t,2],tol)
      }
    }
    else {
      samp <- runif_circle(1,c_radius,center=c(p_df[t+1,1],p_df[t+1,2]))
      V(g)$x[i] <- samp[1]
      V(g)$y[i] <- samp[2]
      V(g)$rd_goal[i] <- (V(g)$x[i] - gx)^2 + (V(g)$y[i] - gy)^2
      V(g)$threat_level[i] <- compute_threat_level(V(g)$x[i],V(g)$y[i],O,obs_st[t+1,1],obs_st[t+1,2],tol)
    }
    
    if (i == (n*(t + 1))) {
      t <- t + 1
      if (t <= size) {
        c_radius <- point_dist(p_df[t,1],p_df[t,2],p_df[t+1,1],p_df[t+1,2])
      }
    }
  }
  V(g)$rd_goal <- (V(g)$rd_goal - min(V(g)$rd_goal))/(max(V(g)$rd_goal) - min(V(g)$rd_goal))
  g
}

generate_obs_ses <- function(O) {
  max_t <- max(O$t)
  max_id <- max(O$id)
  df <- NULL
  for (l in 1:max_t) {
    d <- c()
    for (i in 1:max_id) {
      ids <- which(O$id == i)
      d <- c(d,sqrt(diff(O[ids,1],lag=l)^2 + diff(O[ids,2],lag=l)^2))
    }
    df <- rbind(df,data.frame(m=mean(d),sd=sd(d),t=l,ss=length(d)))
  }
  df
}

merge_obs_st <- function(obs_st1,obs_st2) {
  max_t <- max(max(obs_st1$t),max(obs_st2$t))
  df <- NULL
  for (t in 1:max_t) {
    id1 <- which(obs_st1$t == t)
    id2 <- which(obs_st2$t == t)
    
    if (length(id1) > 0 & length(id2) == 0) {
      df <- rbind(df,obs_st1[id1,])
    }
    else if (length(id1) == 0 & length(id2) > 0) {
      df <- rbind(df,obs_st2[id2,])
    }
    else {
      n <- obs_st1[id1,4]
      x_bar <- obs_st1[id1,1]
      m <- obs_st2[id2,4]
      y_bar <- obs_st2[id2,1]
      
      z_ss <- n + m
      
      z_bar <- (n*x_bar + m*y_bar)/z_ss
      
      x_var <- (obs_st1[id1,2])^2
      y_var <- (obs_st2[id2,2])^2
      
      z_var <- (((n - 1)*x_var + (m - 1)*y_var)/(z_ss - 1)) + 
        ((n*m*(x_bar - y_bar)^2)/(z_ss*(z_ss - 1)))
      
      df <- rbind(df,data.frame(m=z_bar,sd=sqrt(z_var),t=t,ss=z_ss))
    }
  }
  df
}