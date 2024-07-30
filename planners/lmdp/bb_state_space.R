source("~/busy-beeway/planners/lmdp/utility.R")

create_state_space_data <- function(p_df,g,O,delT=3,rho=3,sample_size=29) {
  trans <- NULL
  off_trans <- list()
  ts <- delT + 1
  for (t in 2:nrow(p_df)) {
    p_dist <- point_dist(p_df[t,1],p_df[t,2],p_df[t-1,1],p_df[t-1,2])
    g_heading <- find_direction(p_df[t-1,1],p_df[t-1,2],g[1],g[2])
    g_samp_x <- p_df[t-1,1] + p_dist*cos_plus(g_heading)
    g_samp_y <- p_df[t-1,2] + p_dist*sin_plus(g_heading)
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
    
    goal_distance <- point_dist(m_n_samps[[1]],m_n_samps[[2]],g[1],g[2])
    goal_distance_tr <- goal_distance[sample_size] 
    
    min_obstacle_distance <- rep(-1,sample_size)
    min_obstacle_bee_heading <- rep(-1,sample_size)
    o_t <- (t-2)-delT
    if (o_t >= 0) {
      O_t <- O[which(O$t == o_t),]

      for (i in 1:sample_size) {
        r_ids <- which(ray_intersects(p_df[t-1,1],
                                        p_df[t-1,2],
                                        vec_xs[i],
                                        vec_ys[i],
                                        O_t[,1],
                                        O_t[,2],
                                        cos_plus_vec(O_t[,3]),
                                        sin_plus_vec(O_t[,3])))
        if (length(r_ids) != 0) {
          O_r <- O_t[r_ids,]
          obs_mags <- point_dist(p_df[t-1,1],p_df[t-1,2],O_r[,1],O_r[,2])
          b_o_vec_xs <- O_r[,1] - p_df[t-1,1]
          b_o_vec_ys <- O_r[,2] - p_df[t-1,2]
          o_vec_xs <- cos_plus_vec(O_r[,3])
          o_vec_ys <- sin_plus_vec(O_r[,3])
          obs_dists_sq <- point_dist_sq(m_n_samps[[1]][i],
                                        m_n_samps[[2]][i],
                                        O_r[,1],
                                        O_r[,2])
          min_o <- which.min(obs_dists_sq)
          o_b_dist <- sqrt(obs_dists_sq[min_o])
          if (lesser_equals_plus(o_b_dist,rho)) {
            min_obstacle_distance[i] <- o_b_dist
            o_b_vec_x <- m_n_samps[[1]][i] - O_r[min_o,1]

            o_b_vec_y <- m_n_samps[[2]][i] - O_r[min_o,2]

            min_obstacle_bee_heading[i] <- (o_vec_xs[min_o] * o_b_vec_x + 
                                              o_vec_ys[min_o] * o_b_vec_y)/o_b_dist
          }
        }
      }
    }
    
    min_obstacle_distance_tr <- min_obstacle_distance[sample_size]
    min_obstacle_bee_heading_tr <- min_obstacle_bee_heading[sample_size]

    trans <- rbind(trans, data.frame(goal_distance=goal_distance_tr,
                                     min_obstacle_distance=min_obstacle_distance_tr,
                                     min_obstacle_bee_heading=min_obstacle_bee_heading_tr))
 
    off_trans <- rbind(off_trans,list(data.frame(goal_distance=goal_distance,
                                                 min_obstacle_distance=min_obstacle_distance,
                                                 min_obstacle_bee_heading=min_obstacle_bee_heading)))
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