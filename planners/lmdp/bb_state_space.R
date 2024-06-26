source("~/busy-beeway/planners/lmdp/utility.R")

#For Busy Beeway
compute_threat_level <- function(px,py,O,mu,sig,rho=0.3) {
  not_inside <- point_dist(px,py,O[,1],O[,2]) > rho
  
  inside <- !not_inside
  
  O_outside <- O[which(not_inside),]
  
  O_inside <- O[which(inside),]
  
  Ux <- px - O_outside[,1]
  Uy <- py - O_outside[,2]
  
  Vx <- O_outside[,1] + cos_plus_vec(O_outside[,3])
  Vy <- O_outside[,2] + sin_plus_vec(O_outside[,3])
  
  UV <- Ux*Vx + Uy*Vy
  
  U_1x <- UV*Vx
  U_1y <- UV*Vy
  
  U_2x <- Ux - U_1x
  U_2y <- Uy - U_1y
  
  d <- sqrt(U_2x^2 + U_2y^2)
  
  int_idx <- which(d < rho)
  
  m <- sqrt(rho^2-d[int_idx]^2)
  
  W_1x <- U_1x[int_idx] + m*Vx[int_idx]
  
  W_1y <- U_1y[int_idx] + m*Vy[int_idx]
  
  W_2x <- U_1x[int_idx] - m*Vx[int_idx]
  
  W_2y <- U_1y[int_idx] - m*Vy[int_idx]
  
  P_1x <- O_outside[int_idx,1] + W_1x
  
  P_1y <- O_outside[int_idx,2] + W_1y
  
  P_2x <- O_outside[int_idx,1] + W_2x
  
  P_2y <- O_outside[int_idx,2] + W_2y
  
  dist1 <- sqrt(W_2x^2 + W_2y^2)
  
  dist2 <- sqrt(W_1x^2 + W_1y^2)
  
  p_outside <- rep(0,length(dist1))
  
  lidx <- which(dist1 < dist2)
  uidx <- which(dist1 > dist2)
  
  p_outside[lidx] <- (pnorm(dist2[lidx],mu,sig) - pnorm(dist1[lidx],mu,sig))
  
  p_outside[uidx] <- (pnorm(dist1[uidx],mu,sig) - pnorm(dist2[uidx],mu,sig))
  
  a <- O_inside[,1] - px
  b <- O_inside[,2] - py
  
  n <- a*cos_plus_vec(O_inside[,3]) + b*sin_plus_vec(O_inside[,3])
  
  dist <- -n + sqrt(n^2 - a^2 - b^2 + rho^2)
  
  p_inside <- (pnorm(dist,mu,sig) - pnorm(rep(0,length(dist)),mu,sig))
  
  p <- sum(p_outside) + sum(p_inside)
  
  p
}

create_state_space_data <- function(p_df,g,O,obs_st,delT=3,sample_size=29,rho=0.3) {
  trans <- NULL
  off_trans <- list()
  ts <- delT + 1
  rd_goal_vec <- c()
  threat_level_vec <- c()
  for (t in 2:nrow(p_df)) {
    p_dist_sq <- point_dist_sq(p_df[t,1],p_df[t,2],p_df[t-1,1],p_df[t-1,2])
    p_dist <- sqrt(p_dist_sq)
    g_heading <- find_direction(p_df[t-1,1],p_df[t-1,2],g[1],g[2])
    g_vec_x <- p_dist*cos_plus(g_heading)
    g_vec_y <- p_dist*sin_plus(g_heading)
    g_samp_x <- p_df[t-1,1] + g_vec_x
    g_samp_y <- p_df[t-1,2] + g_vec_y
    if (equals_plus(point_dist(p_df[t,1],p_df[t,2],g_samp_x,g_samp_y),0)) {
      m_n_samps <- runif_on_circle(sample_size - 1,
                                   p_dist,
                                   center=c(p_df[t-1,1],p_df[t-1,2]))
      m_n_samps[[1]] <- c(m_n_samps[[1]],p_df[t,1])
      m_n_samps[[2]] <- c(m_n_samps[[2]],p_df[t,2])
    }
    else {
      m_n_samps <- runif_on_circle(sample_size - 2,
                                   p_dist,
                                   center=c(p_df[t-1,1],p_df[t-1,2]))
      m_n_samps[[1]] <- c(m_n_samps[[1]],g_samp_x)
      m_n_samps[[2]] <- c(m_n_samps[[2]],g_samp_y)
      m_n_samps[[1]] <- c(m_n_samps[[1]],p_df[t,1])
      m_n_samps[[2]] <- c(m_n_samps[[2]],p_df[t,2])
    }
    vec_xs <- m_n_samps[[1]] - p_df[t-1,1]
    vec_ys <- m_n_samps[[2]] - p_df[t-1,2]
    goal_heading <- (g_vec_x * vec_xs + g_vec_y * vec_ys)/p_dist_sq
    goal_heading[which(goal_heading > 1)] <- 1
    goal_heading[which(goal_heading < -1)] <- -1
    r_goal_heading <- 1 - goal_heading
    r_goal_heading_tr <- r_goal_heading[sample_size]
    rd_goal_samps <- point_dist_sq(m_n_samps[[1]],m_n_samps[[2]],g[1],g[2])
    rd_goal_samps <- (rd_goal_samps - min(rd_goal_samps))/(max(rd_goal_samps) - min(rd_goal_samps))
    
    rd_goal_tr <- rd_goal_samps[sample_size]
    threat_level_samps <- rep(0,sample_size)
    o_t <- (t-2)-delT
    if (o_t >= 0) {
      for (i in 1:sample_size) {
        threat_level_samps[i] <- compute_threat_level(m_n_samps[[1]][i],
                                                      m_n_samps[[2]][i],
                                                      O[which(O$t == o_t),],
                                                      obs_st[ts,1],
                                                      obs_st[ts,2],
                                                      rho)
      }
    }
    threat_level_tr <- threat_level_samps[sample_size]
    trans <- rbind(trans, data.frame(r_goal_heading=r_goal_heading_tr,rd_goal=rd_goal_tr, threat_level=threat_level_tr))
    off_trans <- rbind(off_trans,list(data.frame(r_goal_heading=r_goal_heading,rd_goal=rd_goal_samps,threat_level=threat_level_samps)))
  }
  list(trans,off_trans)
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

generate_obs_ses <- function(O) {
  if (is.data.frame(O)) {
    max_t <- max(O$t)
    max_id <- max(O$id)
    df <- NULL
    for (l in 1:max_t) {
      d <- c()
      for (i in 1:max_id) {
        ids <- which(O$id == i)
        d <- c(d,sqrt(diff(O[ids,1],lag=l)^2 + diff(O[ids,2],lag=l)^2))
      }
      df <- rbind(df,data.frame(m=mean(d),sd=sd(d),t=l,ss=as.numeric(length(d))))
    }
  }
  else {
    df <- generate_obs_ses(O[[1]])
    for (i in 2:length(O)) {
      c_df <- generate_obs_ses(O[[i]])
      df <- merge_obs_st(df,c_df)
    }
  }
  df
}