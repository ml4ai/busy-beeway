library("ggplot2")
library("gganimate")
library("truncnorm")
source("~/busy-beeway/planners/lmdp/utility.R")

create_board <- function(px,py,gx,gy,O,obs_st,omin,omax,grid_length,time_step,pspeed) {
  step_size <- time_step*pspeed
  gd <- expand.grid(cols=(-grid_length:grid_length)*step_size + px,
                    rows=(-grid_length:grid_length)*step_size + py)
  cp <- matrix(0,nrow(gd),grid_length)
  half_step <- (step_size/2)
  gd$sqdistgoal <- (gd$cols - gx)^2 + (gd$rows - gy)^2
  e <- (grid_length*step_size + half_step)
  uxb <- e + px
  uyb <- e + py
  lxb <- -e + px
  lyb <- -e + py
  ne_corner <- c(uxb,uyb)
  se_corner <- c(uxb,lyb)
  sw_corner <- c(lxb,lyb)
  nw_corner <- c(lxb,uyb)
  
  outside_n <- greater_equals_plus(O[,1],lxb) & lesser_equals_plus(O[,1],uxb) & O[,2] > uyb
  
  outside_ne <- O[,1] > uxb & O[,2] > uyb
  
  outside_e <- O[,1] > uxb & greater_equals_plus(O[,2],lyb) & lesser_equals_plus(O[,2],uyb)
  
  outside_se <- O[,1] > uxb & O[,2] < lyb
  
  outside_s <- greater_equals_plus(O[,1],lxb) & lesser_equals_plus(O[,1],uxb) & O[,2] < lyb
  
  outside_sw <- O[,1] < lxb & O[,2] < lyb
  
  outside_w <- O[,1] < lxb & greater_equals_plus(O[,2],lyb) & lesser_equals_plus(O[,2],uyb)
  
  outside_nw <- O[,1] < lxb & O[,2] > uyb
  
  inside <- greater_equals_plus(O[,1],lxb) & 
    lesser_equals_plus(O[,1],uxb) & 
    greater_equals_plus(O[,2],lyb) & 
    lesser_equals_plus(O[,2],uyb)
  O$O_max_x <- 0
  O$O_max_y <- 0
  
  for (t in 1:grid_length) {
    cmax <- omax*t*time_step
    cmin <- omin*t*time_step
    O[,5] <- O[,1] + cmax*cos_plus_vec(O[,3])
    O[,6] <- O[,2] + cmax*sin_plus_vec(O[,3])
    
    ob_inside <- O[which(inside),]
    inter_n <- intersects(O[,1],O[,2],O[,5],O[,6],nw_corner[1],nw_corner[2],ne_corner[1],ne_corner[2])
    
    inter_e <- intersects(O[,1],O[,2],O[,5],O[,6],se_corner[1],se_corner[2],ne_corner[1],ne_corner[2])
    
    inter_s <- intersects(O[,1],O[,2],O[,5],O[,6],sw_corner[1],sw_corner[2],se_corner[1],se_corner[2])
    
    inter_w <- intersects(O[,1],O[,2],O[,5],O[,6],sw_corner[1],sw_corner[2],nw_corner[1],nw_corner[2])
    
    valid_inter_n <- O[which((outside_n | outside_nw | outside_ne) & inter_n),]
    
    valid_inter_e <- O[which((outside_e | (outside_ne & !inter_n) | (outside_se & !inter_s)) & inter_e),]
    
    valid_inter_s <- O[which((outside_s | outside_sw | outside_se) & inter_s),]
    
    valid_inter_w <- O[which((outside_w | (outside_nw & !inter_n) | (outside_sw & !inter_s)) & inter_w),]
    
    if (nrow(ob_inside) != 0) {
      for (i in 1:nrow(ob_inside)) {
        cell_id <- which(greater_equals_plus(ob_inside[i,1],(gd$cols - half_step)) & 
                           lesser_equals_plus(ob_inside[i,1],(gd$cols + half_step)) &
                           greater_equals_plus(ob_inside[i,2],(gd$rows - half_step)) & 
                           lesser_equals_plus(ob_inside[i,2],(gd$rows + half_step)))[1]
        X <- gd[cell_id,1]
        Y <- gd[cell_id,2]
        dirX <- ob_inside[i,5] - ob_inside[i,1]
        dirY <- ob_inside[i,6] - ob_inside[i,2]
        s_dirX <- sign(dirX)
        s_dirY <- sign(dirY)
        
        stepX <- s_dirX*step_size
        stepY <- s_dirY*step_size
        tDeltaX <- step_size/s_dirX
        tDeltaY <- step_size/s_dirY
        tMaxX <- (X + (stepX*1/2) - ob_inside[i,1])/dirX
        
        tMaxY <- (Y + (stepY*1/2) - ob_inside[i,2])/dirY
        prev_t <- 0
        repeat {
          if (tMaxX < tMaxY) {
            id <- which(equals_plus(gd$cols,X) & equals_plus(gd$rows,Y))
            d1 <- sqrt(prev_t^2*(dirX^2 + dirY^2))
            d2 <- sqrt(tMaxX^2*(dirX^2 + dirY^2))
            p <- gen_coll_prob(d1,d2,obs_st[t,1],obs_st[t,2],cmin,cmax)
            prev_t <- tMaxX
            cp[id,t] <- cp[id,t] + p - (cp[id,t]*p)
            tMaxX <- tMaxX + tDeltaX
            X <- X + stepX
          }
          else {
            id <- which(equals_plus(gd$cols,X) & equals_plus(gd$rows,Y))
            d1 <- sqrt(prev_t^2*(dirX^2 + dirY^2))
            d2 <- sqrt(tMaxY^2*(dirX^2 + dirY^2))
            p <- gen_coll_prob(d1,d2,obs_st[t,1],obs_st[t,2],cmin,cmax)
            prev_t <- tMaxY
            cp[id,t] <- cp[id,t] + p - (cp[id,t]*p)
            tMaxY <- tMaxY + tDeltaY
            Y <- Y + stepY
          }

          if (!any(equals_plus(gd$cols,X) & equals_plus(gd$rows,Y)) | (tMaxX > 1 & tMaxY > 1)) {
            break
          }
        }
      }
    }
    
    if (nrow(valid_inter_n) != 0) {
      for (i in 1:nrow(valid_inter_n)) {
        dirX <- valid_inter_n[i,5] - valid_inter_n[i,1]
        dirY <- valid_inter_n[i,6] - valid_inter_n[i,2]
        t_inter <- (uyb - valid_inter_n[i,2])/dirY
        inter_origin <- c(valid_inter_n[i,1] + t_inter*dirX,valid_inter_n[i,2] + t_inter*dirY)
        cell_id <- which(greater_equals_plus(inter_origin[1],(gd$cols - half_step)) & 
                           lesser_equals_plus(inter_origin[1],(gd$cols + half_step)) &
                           greater_equals_plus(inter_origin[2],(gd$rows - half_step)) & 
                           lesser_equals_plus(inter_origin[2],(gd$rows + half_step)))[1]
        
        X <- gd[cell_id,1]
        Y <- gd[cell_id,2]
        s_dirX <- sign(dirX)
        s_dirY <- sign(dirY)
        
        stepX <- s_dirX*step_size
        stepY <- s_dirY*step_size
        tDeltaX <- step_size/s_dirX
        tDeltaY <- step_size/s_dirY
        tMaxX <- (X + (stepX*1/2) - valid_inter_n[i,1])/dirX
        
        tMaxY <- (Y + (stepY*1/2) - valid_inter_n[i,2])/dirY
        prev_t <- t_inter
        repeat {
          if (tMaxX < tMaxY) {
            id <- which(equals_plus(gd$cols,X) & equals_plus(gd$rows,Y))
            d1 <- sqrt(prev_t^2*(dirX^2 + dirY^2))
            d2 <- sqrt(tMaxX^2*(dirX^2 + dirY^2))
            p <- gen_coll_prob(d1,d2,obs_st[t,1],obs_st[t,2],cmin,cmax)
            prev_t <- tMaxX
            cp[id,t] <- cp[id,t] + p - (cp[id,t]*p)
            tMaxX <- tMaxX + tDeltaX
            X <- X + stepX
          }
          else {
            id <- which(equals_plus(gd$cols,X) & equals_plus(gd$rows,Y))
            d1 <- sqrt(prev_t^2*(dirX^2 + dirY^2))
            d2 <- sqrt(tMaxY^2*(dirX^2 + dirY^2))
            p <- gen_coll_prob(d1,d2,obs_st[t,1],obs_st[t,2],cmin,cmax)
            prev_t <- tMaxY
            cp[id,t] <- cp[id,t] + p - (cp[id,t]*p)
            tMaxY <- tMaxY + tDeltaY
            Y <- Y + stepY
          }

          if (!any(equals_plus(gd$cols,X) & equals_plus(gd$rows,Y)) | (tMaxX > 1 & tMaxY > 1)) {
            break
          }
        }
      }
    }
    
    if (nrow(valid_inter_e) != 0) {
      for (i in 1:nrow(valid_inter_e)) {
        dirX <- valid_inter_e[i,5] - valid_inter_e[i,1]
        dirY <- valid_inter_e[i,6] - valid_inter_e[i,2]
        t_inter <- (uxb - valid_inter_e[i,1])/dirX
        inter_origin <- c(valid_inter_e[i,1] + t_inter*dirX,valid_inter_e[i,2] + t_inter*dirY)
        cell_id <- which(greater_equals_plus(inter_origin[1],(gd$cols - half_step)) & 
                           lesser_equals_plus(inter_origin[1],(gd$cols + half_step)) &
                           greater_equals_plus(inter_origin[2],(gd$rows - half_step)) & 
                           lesser_equals_plus(inter_origin[2],(gd$rows + half_step)))[1]
        
        X <- gd[cell_id,1]
        Y <- gd[cell_id,2]
        s_dirX <- sign(dirX)
        s_dirY <- sign(dirY)
        
        stepX <- s_dirX*step_size
        stepY <- s_dirY*step_size
        tDeltaX <- step_size/s_dirX
        tDeltaY <- step_size/s_dirY
        tMaxX <- (X + (stepX*1/2) - valid_inter_e[i,1])/dirX
        
        tMaxY <- (Y + (stepY*1/2) - valid_inter_e[i,2])/dirY
        prev_t <- t_inter
        repeat {
          if (tMaxX < tMaxY) {
            id <- which(equals_plus(gd$cols,X) & equals_plus(gd$rows,Y))
            d1 <- sqrt(prev_t^2*(dirX^2 + dirY^2))
            d2 <- sqrt(tMaxX^2*(dirX^2 + dirY^2))
            p <- gen_coll_prob(d1,d2,obs_st[t,1],obs_st[t,2],cmin,cmax)
            prev_t <- tMaxX
            cp[id,t] <- cp[id,t] + p - (cp[id,t]*p)
            tMaxX <- tMaxX + tDeltaX
            X <- X + stepX
          }
          else {
            id <- which(equals_plus(gd$cols,X) & equals_plus(gd$rows,Y))
            d1 <- sqrt(prev_t^2*(dirX^2 + dirY^2))
            d2 <- sqrt(tMaxY^2*(dirX^2 + dirY^2))
            p <- gen_coll_prob(d1,d2,obs_st[t,1],obs_st[t,2],cmin,cmax)
            prev_t <- tMaxY
            cp[id,t] <- cp[id,t] + p - (cp[id,t]*p)
            tMaxY <- tMaxY + tDeltaY
            Y <- Y + stepY
          }

          if (!any(equals_plus(gd$cols,X) & equals_plus(gd$rows,Y)) | (tMaxX > 1 & tMaxY > 1)) {
            break
          }
        }
      }
    }
    
    if (nrow(valid_inter_s) != 0) {
      for (i in 1:nrow(valid_inter_s)) {
        dirX <- valid_inter_s[i,5] - valid_inter_s[i,1]
        dirY <- valid_inter_s[i,6] - valid_inter_s[i,2]
        t_inter <- (lyb - valid_inter_s[i,2])/dirY
        inter_origin <- c(valid_inter_s[i,1] + t_inter*dirX,valid_inter_s[i,2] + t_inter*dirY)
        cell_id <- which(greater_equals_plus(inter_origin[1],(gd$cols - half_step)) & 
                           lesser_equals_plus(inter_origin[1],(gd$cols + half_step)) &
                           greater_equals_plus(inter_origin[2],(gd$rows - half_step)) & 
                           lesser_equals_plus(inter_origin[2],(gd$rows + half_step)))[1]
        
        X <- gd[cell_id,1]
        Y <- gd[cell_id,2]
        s_dirX <- sign(dirX)
        s_dirY <- sign(dirY)
        
        stepX <- s_dirX*step_size
        stepY <- s_dirY*step_size
        tDeltaX <- step_size/s_dirX
        tDeltaY <- step_size/s_dirY
        tMaxX <- (X + (stepX*1/2) - valid_inter_s[i,1])/dirX
        
        tMaxY <- (Y + (stepY*1/2) - valid_inter_s[i,2])/dirY
        prev_t <- t_inter
        repeat {
          if (tMaxX < tMaxY) {
            id <- which(equals_plus(gd$cols,X) & equals_plus(gd$rows,Y))
            d1 <- sqrt(prev_t^2*(dirX^2 + dirY^2))
            d2 <- sqrt(tMaxX^2*(dirX^2 + dirY^2))
            p <- gen_coll_prob(d1,d2,obs_st[t,1],obs_st[t,2],cmin,cmax)
            prev_t <- tMaxX
            cp[id,t] <- cp[id,t] + p - (cp[id,t]*p)
            tMaxX <- tMaxX + tDeltaX
            X <- X + stepX
          }
          else {
            id <- which(equals_plus(gd$cols,X) & equals_plus(gd$rows,Y))
            d1 <- sqrt(prev_t^2*(dirX^2 + dirY^2))
            d2 <- sqrt(tMaxY^2*(dirX^2 + dirY^2))
            p <- gen_coll_prob(d1,d2,obs_st[t,1],obs_st[t,2],cmin,cmax)
            prev_t <- tMaxY
            cp[id,t] <- cp[id,t] + p - (cp[id,t]*p)
            tMaxY <- tMaxY + tDeltaY
            Y <- Y + stepY
          }

          if (!any(equals_plus(gd$cols,X) & equals_plus(gd$rows,Y)) | (tMaxX > 1 & tMaxY > 1)) {
            break
          }
        }
      }
    }
    
    if (nrow(valid_inter_w) != 0) {
      for (i in 1:nrow(valid_inter_w)) {
        dirX <- valid_inter_w[i,5] - valid_inter_w[i,1]
        dirY <- valid_inter_w[i,6] - valid_inter_w[i,2]
        t_inter <- (lxb - valid_inter_w[i,1])/dirX
        inter_origin <- c(valid_inter_w[i,1] + t_inter*dirX,valid_inter_w[i,2] + t_inter*dirY)
        cell_id <- which(greater_equals_plus(inter_origin[1],(gd$cols - half_step)) & 
                           lesser_equals_plus(inter_origin[1],(gd$cols + half_step)) &
                           greater_equals_plus(inter_origin[2],(gd$rows - half_step)) & 
                           lesser_equals_plus(inter_origin[2],(gd$rows + half_step)))[1]
        
        X <- gd[cell_id,1]
        Y <- gd[cell_id,2]
        s_dirX <- sign(dirX)
        s_dirY <- sign(dirY)
        
        stepX <- s_dirX*step_size
        stepY <- s_dirY*step_size
        tDeltaX <- step_size/s_dirX
        tDeltaY <- step_size/s_dirY
        tMaxX <- (X + (stepX*1/2) - valid_inter_w[i,1])/dirX
        
        tMaxY <- (Y + (stepY*1/2) - valid_inter_w[i,2])/dirY
        prev_t <- t_inter
        repeat {
          if (tMaxX < tMaxY) {
            id <- which(equals_plus(gd$cols,X) & equals_plus(gd$rows,Y))
            d1 <- sqrt(prev_t^2*(dirX^2 + dirY^2))
            d2 <- sqrt(tMaxX^2*(dirX^2 + dirY^2))
            p <- gen_coll_prob(d1,d2,obs_st[t,1],obs_st[t,2],cmin,cmax)
            prev_t <- tMaxX
            cp[id,t] <- cp[id,t] + p - (cp[id,t]*p)
            tMaxX <- tMaxX + tDeltaX
            X <- X + stepX
          }
          else {
            id <- which(equals_plus(gd$cols,X) & equals_plus(gd$rows,Y))
            d1 <- sqrt(prev_t^2*(dirX^2 + dirY^2))
            d2 <- sqrt(tMaxY^2*(dirX^2 + dirY^2))
            p <- gen_coll_prob(d1,d2,obs_st[t,1],obs_st[t,2],cmin,cmax)
            prev_t <- tMaxY
            cp[id,t] <- cp[id,t] + p - (cp[id,t]*p)
            tMaxY <- tMaxY + tDeltaY
            Y <- Y + stepY
          }

          if (!any(equals_plus(gd$cols,X) & equals_plus(gd$rows,Y)) | (tMaxX > 1 & tMaxY > 1)) {
            break
          }
        }
      }
    }
  }

  gd$collprobs <- 1-apply(1-cp,1,prod)
  gd
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

gen_coll_prob <- function(d1,d2,m,s,mind=0,maxd=16) {
  s <- ptruncnorm(d2,mind,maxd,m,s) - ptruncnorm(d1,mind,maxd,m,s)
  s
}

plot_game_board <- function(states,tr=NULL,fill_data=0,fill_aux=NULL) {
  c_idx <- (nrow(states) + 1)/2
  player <- data.frame(x=states[c_idx,1],y=states[c_idx,2])
  if (fill_data == 0) {
    if (is.null(tr)) {
      g <- ggplot() + 
        theme_bw() + 
        geom_tile(states,mapping=aes(x=cols,y=rows,fill=collprob),colour="white") +
        scale_fill_gradient(low="black",high="orange") +
        geom_point(player,mapping=aes(x,y),size=.02*nrow(states),color="white")
    }
    else {
      g <- ggplot() + 
        theme_bw() + 
        geom_tile(states,mapping=aes(x=cols,y=rows,fill=collprob),colour="white") +
        scale_fill_gradient(low="black",high="orange") +
        geom_point(player,mapping=aes(x,y),size=.02*nrow(states),color="white") 
      for (i in 2:length(tr)) {
        df <- data.frame(x1 = states[tr[i-1],1],x2 = states[tr[i],1],y1 = states[tr[i-1],2], y2 = states[tr[i],2])
        g <- g + geom_segment(df,mapping=aes(x = x1,y = y1, xend = x2, yend = y2),
                              arrow=arrow(length=unit(0.15, "inches")))
      }
    }
  }
  else if (fill_data == 1) {
    if (is.null(tr)) {
      g <- ggplot() + 
        theme_bw() + 
        geom_tile(states,mapping=aes(x=cols,y=rows,fill=sqdistgoal),colour="white") +
        scale_fill_gradient(low="green",high="black") +
        geom_point(player,mapping=aes(x,y),size=.02*nrow(states),color="white")
    }
    else {
      g <- ggplot() + 
        theme_bw() + 
        geom_tile(states,mapping=aes(x=cols,y=rows,fill=sqdistgoal),colour="white") +
        scale_fill_gradient(low="green",high="black") +
        geom_point(player,mapping=aes(x,y),size=.02*nrow(states),color="white") 
      for (i in 2:length(tr)) {
        df <- data.frame(x1 = states[tr[i-1],1],x2 = states[tr[i],1],y1 = states[tr[i-1],2], y2 = states[tr[i],2])
        g <- g + geom_segment(df,mapping=aes(x = x1,y = y1, xend = x2, yend = y2),
                              arrow=arrow(length=unit(0.15, "inches")))
      }
    }
  }
  else {
    if (is.null(tr)) {
      g <- ggplot() + 
        theme_bw() + 
        geom_tile(states,mapping=aes(x=cols,y=rows,fill=fill_aux),colour="white") +
        scale_fill_gradient(low="blue",high="red") +
        geom_point(player,mapping=aes(x,y),size=.02*nrow(states),color="white")
    }
    else {
      g <- ggplot() + 
        theme_bw() + 
        geom_tile(states,mapping=aes(x=cols,y=rows,fill=fill_aux),colour="white") +
        scale_fill_gradient(low="blue",high="red") +
        geom_point(player,mapping=aes(x,y),size=.02*nrow(states),color="white") 
      for (i in 2:length(tr)) {
        df <- data.frame(x1 = states[tr[i-1],1],x2 = states[tr[i],1],y1 = states[tr[i-1],2], y2 = states[tr[i],2])
        g <- g + geom_segment(df,mapping=aes(x = x1,y = y1, xend = x2, yend = y2),
                              arrow=arrow(length=unit(0.15, "inches")))
      }
    }
  }
  g
}

create_vf <- function(b1,b2) {
  valfunc <- function(states,g) {
    val <- b1*((states$sqdistgoal - min(states$sqdistgoal))/(max(states$sqdistgoal) - min(states$sqdistgoal))) + 
      b2*states$collprob
    g_id <- which(greater_equals_plus(g[1],(states$cols - 1/2)) & lesser_equals_plus(g[1],(states$cols + 1/2)) &
                    greater_equals_plus(g[2],(states$rows - 1/2)) & lesser_equals_plus(g[2],(states$rows + 1/2)))
    if (length(g_id) != 0) {
      val[g_id[1]] <- 0
    }
    val
  }
  valfunc
}

create_uniform_default_policy_from_grid <- function(states,g_id = NULL) {
  adj_mat <- apply(states, 1,function(pt)
    (abs(pt["rows"] - states$rows) <= 1 & abs(pt["cols"] - states$cols) <= 1))
  diag(adj_mat) <- 0
  adj_mat[g_id,] <- 0
  adj_mat[g_id,g_id] <- 1
  adj_mat <- adj_mat/rowSums(adj_mat)
  adj_mat
}

create_uniform_default_policy <- function(grid_length,g_id = NULL) {
  gd <- expand.grid(cols=-grid_length:grid_length,
                    rows=-grid_length:grid_length)
  create_uniform_default_policy_from_grid(gd,g_id)
}

max_trajectory <- function(init,t_mat,states,H=8) {
  initial <- which(equals_plus(states$cols,init[1]) & equals_plus(states$rows,init[2]))
  T_1 <- matrix(0,nrow(states),H)
  T_2 <- matrix(0,nrow(states),H)
  T_1[initial,1] <- 1
  for (j in 2:H) {
    for (i in 1:nrow(states)) {
      p <- T_1[,j-1]*t_mat[,i]
      T_1[i,j] <- max(p)
      T_2[i,j] <- which.max(p)
    }
  }
  z <- rep(0,H)
  trX <- rep(0,H)
  trY <- rep(0,H)
  z[H] <- which.max(T_1[,H])
  trX[H] <- states[z[H],1]
  trY[H] <- states[z[H],2]
  for (j in H:2) {
    z[j-1] <- T_2[z[j],j]
    trX[j-1] <- states[z[j-1],1]
    trY[j-1] <- states[z[j-1],2]
  }
  data.frame(x=trX,y=trY)
}

LRL <- function(dp,valfunc,states,g) {
  v <- valfunc(states,g)
  z <- exp(-v) 
  u <- matrix(0,length(z),length(z))
  G <- dp %*% z
  for(i in 1:length(z)){
    u[i,which(dp[i,]>0)] <- dp[i,which(dp[i,]>0)]*z[which(dp[i,]>0)]/G[i]
  }
  return(list(u,z))
}

ob_moves <- function(O,ptr,speeds,probs) {
  speed <- sample(speeds,1,prob=probs)
  O[,1] <- O[,1] + (ptr*speed*cos_plus_vec(O[,3]))
  O[,2] <- O[,2] + (ptr*speed*sin_plus_vec(O[,3]))
  O
}

pointCollide <- function(x1,y1,x2,y2,tol) {
  (x1-x2)^2 + (y1-y2)^2 < (2*tol)^2
}

collision <- function(prevX,prevY,curX,curY,px,py,tol = 0.3) {
  CP <- closestpointonline(prevX,prevY,curX,curY,px,py)
  cpX <- CP[[1]]
  cpY <- CP[[2]]
  list(any(pointCollide(cpX,cpY,px,py,tol)),cpX,cpY)
}

closestpointonline <- function(ax,ay,bx,by,px,py) {
  apx <- px - ax
  apy <- py - ay
  abx <- bx - ax
  aby <- by - ay
  
  ab2 <- abx^2 + aby^2
  
  apab <- apx*abx + apy*aby
  
  t <- apab/ab2
  t[which(is.na(t))] <- 0
  t[which(t < 0)] <- 0
  t[which(t > 1)] <- 1
  list(ax + abx*t,ay+aby*t)
}

find_direction <- function(x1,y1,x2,y2) {
  y <- y2 - y1
  x <- x2 - x1
  if (equals_plus(y,0) & x > 0) {
    h <- 360
  }
  else if (greater_equals_plus(y,0)) {
    h <- atan2(y,x)*180/pi
  }
  else if (y < 0) {
    h <- (atan2(y,x) + 2*pi)*180/pi
  }
  else {
    h <- NaN
  }
  h
}

simulate_coll_probs <- function(x_dim,y_dim,t,ob_prob=0.2,ub=0.5) {
  grid_size <- x_dim*y_dim
  total <- grid_size*t
  matrix(rbinom(total,1,ob_prob)*runif(total,0,ub),t,grid_size)
}

simulate_global_board <- function(grid_length,step_size=1) {
  gd <- expand.grid(cols=(-grid_length:grid_length)*step_size,
                    rows=(-grid_length:grid_length)*step_size)
  gd
}

simulate_board <- function(px,py,gx,gy,gb,O,t,grid_length,step_size=1) {
  gd <- expand.grid(cols=(-grid_length:grid_length)*step_size + px,
                    rows=(-grid_length:grid_length)*step_size + py)
  gd <- transform(merge(
    x = gd,
    y = cbind(rownames = rownames(gb), gb),by=c("cols","rows")
  ), row.names = rownames, rownames = NULL)
  gd$index <- as.numeric(row.names(gd))
  r_ids <- order(gd$index)
  gd <- gd[r_ids,]
  gd$index <- NULL
  row.names(gd) <- NULL
  gd$sqdistgoal <- (gd$cols - gx)^2 + (gd$rows - gy)^2
  gd$collprobs <- O[t,r_ids]
  gd
}

simulate_collision <- function(px,py,gb,O,t) {
  id <- which(gb$cols == px & gb$rows == py)
  rbinom(1,1,prob=O[t,id])
}

#Simplified Version
L_planner <- function(p,
                      g,
                      O,
                      gb,
                      valf,
                      delay=8) {
  p_pos <- data.frame(x=p[1],y=p[2],t=0)
  dp <- create_uniform_default_policy(delay)
  current_p <- p
  idx <- 1
  tr <- NULL
  if (current_p[1] == g[1] & current_p[2] == g[2]) {
    return(list(p_pos,O,TRUE,g))
  }
  for (t in 1:NROW(O)) {
    if (idx > NROW(tr)) {
      c_board <- simulate_board(current_p[1],current_p[2],g[1],g[2],gb,O,t,delay)
      res <- LRL(dp,valf,c_board,g)
      tr <- max_trajectory(current_p,res[[1]],c_board)
      idx <- 2
    }
    current_p <- c(tr[idx,1],tr[idx,2])
    p_pos <- rbind(p_pos,data.frame(x=current_p[1],y=current_p[2],t=t))
    if (simulate_collision(current_p[1],current_p[2],gb,O,t)) {
      return(list(p_pos,O,FALSE,g))
    }
    if (current_p[1] == g[1] & current_p[2] == g[2]) {
      return(list(p_pos,O,TRUE,g))
    }
    idx <- idx + 1
  }
  print("Out of time!")
  return(list(p_pos,O,FALSE,g))
}