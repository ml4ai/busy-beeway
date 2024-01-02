library("ggplot2")
library("gganimate")

cos_plus <- function(degrees) {
  if (equals_plus(degrees, 90) | equals_plus(degrees, 270)) {
    return(0)
  }
  cos(degrees*(pi/180))
}

sin_plus <- function(degrees) {
  if (equals_plus(degrees, 360) | equals_plus(degrees, 180)) {
    return(0)
  }
  sin(degrees*(pi/180))
}

cos_plus_vec <- function(degrees) {
  res <- rep(0,length(degrees))
  id <- which(!equals_plus(degrees,90) & !equals_plus(degrees,270))
  res[id] <- cos(degrees[id]*(pi/180))
}

sin_plus_vec <- function(degrees) {
  res <- rep(0,length(degrees))
  id <- which(!equals_plus(degrees,360) & !equals_plus(degrees,180))
  res[id] <- sin(degrees[id]*(pi/180))
}

equals_plus <- function(x,y,tol=sqrt(.Machine$double.eps)) {
  abs(x-y) <= tol
}

greater_equals_plus <- function(x,y,tol=sqrt(.Machine$double.eps)) {
  (x > y) | equals_plus(x,y,tol)
}

lesser_equals_plus <- function(x,y,tol=sqrt(.Machine$double.eps)) {
  (x < y) | equals_plus(x,y,tol)
}

create_random_board <- function(t=4,g=30,ob=10,cp) {
  heading <- c("N","E","S","W")
  ob_headings <- sample(heading,ob,replace=TRUE)
  gd <- expand.grid(cols=-t:t,rows=-t:t)
  obstacles <- gd[sample(which(gd$cols != 0 & gd$rows != 0),ob),c(1,2)]
  obstacles$heading <- ob_headings
  gh <- runif(1,1,360)
  goal <- c(0,0) + c(g*cos_plus(gh),g*sin_plus(gh))
  gd$sqdistgoal <- (gd$cols - goal[1])^2 + (gd$rows - goal[2])^2
  gd$collprob <- 0.0
  for (o in 1:nrow(obstacles)) {
    transf <- switch(obstacles[o,3],
                     N=c(0,1),E=c(1,0),S=c(0,-1),W=c(-1,0))
    current_p <- c(obstacles[o,1],obstacles[o,2])
    for (p in cp) {
      gd[which(gd$cols == current_p[1] & gd$rows == current_p[2]),4] <- 
        gd[which(gd$cols == current_p[1] & gd$rows == current_p[2]),4] + 
        p - gd[which(gd$cols == current_p[1] & gd$rows == current_p[2]),4]*p
      current_p <- c(current_p[1] + transf[1], current_p[2] + transf[2])
      if (!any(gd$cols == current_p[1] & gd$rows == current_p[2])) {
        break
      }
    }
  }
  gd
}

ccw <- function(ax,ay,bx,by,cx,cy) {
  (bx - ax)*(cy - ay) - (by - ay)*(cx - ax)
}

intersects <- function(ax,ay,bx,by,cx,cy,dx,dy) {
  ccw(ax,ay,bx,by,cx,cy)*ccw(ax,ay,bx,by,dx,dy) < 0 & ccw(cx,cy,dx,dy,ax,ay)*ccw(cx,cy,dx,dy,bx,by) < 0
}

frac <- function(x) {
  if (x > 0) {
    x - floor(x)
  }
  else {
    1 - x + floor(x)
  }
}

point_dist <- function(x1,y1,x2,y2) {
  sqrt((x2 - x1)^2 + (y2 - y1)^2)
}

create_board <- function(px,py,gx,gy,O,obs_st,omin,omax,pspeed,pt=1) {
  grid_length <- pspeed*pt
  gd <- expand.grid(cols=-grid_length:grid_length + px,
                    rows=-grid_length:grid_length + py)
  cp <- matrix(0,nrow(gd),grid_length)
  gd$sqdistgoal <- (gd$cols - gx)^2 + (gd$rows - gy)^2
  e <- (grid_length + 1/2)
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
    cmax <- omax*(t/grid_length)
    cmin <- omin*(t/grid_length)
    O[,4] <- O[,1] + cmax*cos_plus_vec(O[,3])
    O[,5] <- O[,2] + cmax*sin_plus_vec(O[,3])
    
    ob_inside <- O[which(inside),]
    inter_n <- intersects(O[,1],O[,2],O[,4],O[,5],nw_corner[1],nw_corner[2],ne_corner[1],ne_corner[2])
    
    inter_e <- intersects(O[,1],O[,2],O[,4],O[,5],se_corner[1],se_corner[2],ne_corner[1],ne_corner[2])
    
    inter_s <- intersects(O[,1],O[,2],O[,4],O[,5],sw_corner[1],sw_corner[2],se_corner[1],se_corner[2])
    
    inter_w <- intersects(O[,1],O[,2],O[,4],O[,5],sw_corner[1],sw_corner[2],nw_corner[1],nw_corner[2])
    
    valid_inter_n <- O[which((outside_n | outside_nw | outside_ne) & inter_n),]
    
    valid_inter_e <- O[which((outside_e | (outside_ne & !inter_n) | (outside_se & !inter_s)) & inter_e),]
    
    valid_inter_s <- O[which((outside_s | outside_sw | outside_se) & inter_s),]
    
    valid_inter_w <- O[which((outside_w | (outside_nw & !inter_n) | (outside_sw & !inter_s)) & inter_w),]
    
    if (nrow(ob_inside) != 0) {
      for (i in 1:nrow(ob_inside)) {
        cell_id <- which(greater_equals_plus(ob_inside[i,1],(gd$cols - 1/2)) & 
                           lesser_equals_plus(ob_inside[i,1],(gd$cols + 1/2)) &
                           greater_equals_plus(ob_inside[i,2],(gd$rows - 1/2)) & 
                           lesser_equals_plus(ob_inside[i,2],(gd$rows + 1/2)))[1]
        X <- gd[cell_id,1]
        Y <- gd[cell_id,2]
        dirX <- ob_inside[i,4] - ob_inside[i,1]
        dirY <- ob_inside[i,5] - ob_inside[i,2]
        stepX <- sign(dirX)
        stepY <- sign(dirY)
        tDeltaX <- 1/(stepX*dirX)
        tDeltaY <- 1/(stepY*dirY)
        tMaxX <- (X + (stepX*1/2) - ob_inside[i,1])/dirX
        
        tMaxY <- (Y + (stepY*1/2) - ob_inside[i,2])/dirY
        prev_t <- 0
        while (TRUE) {
          if (tMaxX < tMaxY) {
            id <- which(equals_plus(gd$cols,X) & equals_plus(gd$rows,Y))
            d1 <- sqrt(prev_t^2*(dirX^2 + dirY^2))
            d2 <- sqrt(tMaxX^2*(dirX^2 + dirY^2))
            p <- gen_coll_prob(d1,d2,obs_st[[t,1]],obs_st[[t,2]],cmin,cmax)
            prev_t <- tMaxX
            cp[id,t] <- cp[id,t] + p - cp[id,t]*p
            tMaxX <- tMaxX + tDeltaX
            X <- X + stepX
          }
          else {
            id <- which(equals_plus(gd$cols,X) & equals_plus(gd$rows,Y))
            d1 <- sqrt(prev_t^2*(dirX^2 + dirY^2))
            d2 <- sqrt(tMaxY^2*(dirX^2 + dirY^2))
            p <- gen_coll_prob(d1,d2,obs_st[[t,1]],obs_st[[t,2]],cmin,cmax)
            prev_t <- tMaxY
            cp[id,t] <- cp[id,t] + p - cp[id,t]*p
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
        dirX <- valid_inter_n[i,4] - valid_inter_n[i,1]
        dirY <- valid_inter_n[i,5] - valid_inter_n[i,2]
        t_inter <- (uyb - valid_inter_n[i,2])/dirY
        inter_origin <- c(valid_inter_n[i,1] + t_inter*dirX,valid_inter_n[i,2] + t_inter*dirY)
        cell_id <- which(greater_equals_plus(inter_origin[1],(gd$cols - 1/2)) & 
                           lesser_equals_plus(inter_origin[1],(gd$cols + 1/2)) &
                           greater_equals_plus(inter_origin[2],(gd$rows - 1/2)) & 
                           lesser_equals_plus(inter_origin[2],(gd$rows + 1/2)))[1]
        
        X <- gd[cell_id,1]
        Y <- gd[cell_id,2]
        stepX <- sign(dirX)
        stepY <- sign(dirY)
        tDeltaX <- 1/(stepX*dirX)
        tDeltaY <- 1/(stepY*dirY)
        tMaxX <- (X + (stepX*1/2) - valid_inter_n[i,1])/dirX
        
        tMaxY <- (Y + (stepY*1/2) - valid_inter_n[i,2])/dirY
        prev_t <- t_inter
        while (TRUE) {
          if (tMaxX < tMaxY) {
            id <- which(equals_plus(gd$cols,X) & equals_plus(gd$rows,Y))
            d1 <- sqrt(prev_t^2*(dirX^2 + dirY^2))
            d2 <- sqrt(tMaxX^2*(dirX^2 + dirY^2))
            p <- gen_coll_prob(d1,d2,obs_st[[t,1]],obs_st[[t,2]],cmin,cmax)
            prev_t <- tMaxX
            cp[id,t] <- cp[id,t] + p - cp[id,t]*p
            tMaxX <- tMaxX + tDeltaX
            X <- X + stepX
          }
          else {
            id <- which(equals_plus(gd$cols,X) & equals_plus(gd$rows,Y))
            d1 <- sqrt(prev_t^2*(dirX^2 + dirY^2))
            d2 <- sqrt(tMaxY^2*(dirX^2 + dirY^2))
            p <- gen_coll_prob(d1,d2,obs_st[[t,1]],obs_st[[t,2]],cmin,cmax)
            prev_t <- tMaxY
            cp[id,t] <- cp[id,t] + p - cp[id,t]*p
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
        dirX <- valid_inter_e[i,4] - valid_inter_e[i,1]
        dirY <- valid_inter_e[i,5] - valid_inter_e[i,2]
        t_inter <- (uxb - valid_inter_e[i,1])/dirX
        inter_origin <- c(valid_inter_e[i,1] + t_inter*dirX,valid_inter_e[i,2] + t_inter*dirY)
        cell_id <- which(greater_equals_plus(inter_origin[1],(gd$cols - 1/2)) & 
                           lesser_equals_plus(inter_origin[1],(gd$cols + 1/2)) &
                           greater_equals_plus(inter_origin[2],(gd$rows - 1/2)) & 
                           lesser_equals_plus(inter_origin[2],(gd$rows + 1/2)))[1]
        
        X <- gd[cell_id,1]
        Y <- gd[cell_id,2]
        stepX <- sign(dirX)
        stepY <- sign(dirY)
        tDeltaX <- 1/(stepX*dirX)
        tDeltaY <- 1/(stepY*dirY)
        tMaxX <- (X + (stepX*1/2) - valid_inter_e[i,1])/dirX
        
        tMaxY <- (Y + (stepY*1/2) - valid_inter_e[i,2])/dirY
        prev_t <- t_inter
        while (TRUE) {
          if (tMaxX < tMaxY) {
            id <- which(equals_plus(gd$cols,X) & equals_plus(gd$rows,Y))
            d1 <- sqrt(prev_t^2*(dirX^2 + dirY^2))
            d2 <- sqrt(tMaxX^2*(dirX^2 + dirY^2))
            p <- gen_coll_prob(d1,d2,obs_st[[t,1]],obs_st[[t,2]],cmin,cmax)
            prev_t <- tMaxX
            cp[id,t] <- cp[id,t] + p - cp[id,t]*p
            tMaxX <- tMaxX + tDeltaX
            X <- X + stepX
          }
          else {
            id <- which(equals_plus(gd$cols,X) & equals_plus(gd$rows,Y))
            d1 <- sqrt(prev_t^2*(dirX^2 + dirY^2))
            d2 <- sqrt(tMaxY^2*(dirX^2 + dirY^2))
            p <- gen_coll_prob(d1,d2,obs_st[[t,1]],obs_st[[t,2]],cmin,cmax)
            prev_t <- tMaxY
            cp[id,t] <- cp[id,t] + p - cp[id,t]*p
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
        dirX <- valid_inter_s[i,4] - valid_inter_s[i,1]
        dirY <- valid_inter_s[i,5] - valid_inter_s[i,2]
        t_inter <- (lyb - valid_inter_s[i,2])/dirY
        inter_origin <- c(valid_inter_s[i,1] + t_inter*dirX,valid_inter_s[i,2] + t_inter*dirY)
        cell_id <- which(greater_equals_plus(inter_origin[1],(gd$cols - 1/2)) & 
                           lesser_equals_plus(inter_origin[1],(gd$cols + 1/2)) &
                           greater_equals_plus(inter_origin[2],(gd$rows - 1/2)) & 
                           lesser_equals_plus(inter_origin[2],(gd$rows + 1/2)))[1]
        
        X <- gd[cell_id,1]
        Y <- gd[cell_id,2]
        stepX <- sign(dirX)
        stepY <- sign(dirY)
        tDeltaX <- 1/(stepX*dirX)
        tDeltaY <- 1/(stepY*dirY)
        tMaxX <- (X + (stepX*1/2) - valid_inter_s[i,1])/dirX
        
        tMaxY <- (Y + (stepY*1/2) - valid_inter_s[i,2])/dirY
        prev_t <- t_inter
        while (TRUE) {
          if (tMaxX < tMaxY) {
            id <- which(equals_plus(gd$cols,X) & equals_plus(gd$rows,Y))
            d1 <- sqrt(prev_t^2*(dirX^2 + dirY^2))
            d2 <- sqrt(tMaxX^2*(dirX^2 + dirY^2))
            p <- gen_coll_prob(d1,d2,obs_st[[t,1]],obs_st[[t,2]],cmin,cmax)
            prev_t <- tMaxX
            cp[id,t] <- cp[id,t] + p - cp[id,t]*p
            tMaxX <- tMaxX + tDeltaX
            X <- X + stepX
          }
          else {
            id <- which(equals_plus(gd$cols,X) & equals_plus(gd$rows,Y))
            d1 <- sqrt(prev_t^2*(dirX^2 + dirY^2))
            d2 <- sqrt(tMaxY^2*(dirX^2 + dirY^2))
            p <- gen_coll_prob(d1,d2,obs_st[[t,1]],obs_st[[t,2]],cmin,cmax)
            prev_t <- tMaxY
            cp[id,t] <- cp[id,t] + p - cp[id,t]*p
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
        dirX <- valid_inter_w[i,4] - valid_inter_w[i,1]
        dirY <- valid_inter_w[i,5] - valid_inter_w[i,2]
        t_inter <- (lxb - valid_inter_w[i,1])/dirX
        inter_origin <- c(valid_inter_w[i,1] + t_inter*dirX,valid_inter_w[i,2] + t_inter*dirY)
        cell_id <- which(greater_equals_plus(inter_origin[1],(gd$cols - 1/2)) & 
                           lesser_equals_plus(inter_origin[1],(gd$cols + 1/2)) &
                           greater_equals_plus(inter_origin[2],(gd$rows - 1/2)) & 
                           lesser_equals_plus(inter_origin[2],(gd$rows + 1/2)))[1]

        X <- gd[cell_id,1]
        Y <- gd[cell_id,2]
        stepX <- sign(dirX)
        stepY <- sign(dirY)
        tDeltaX <- 1/(stepX*dirX)
        tDeltaY <- 1/(stepY*dirY)
        tMaxX <- (X + (stepX*1/2) - valid_inter_w[i,1])/dirX
        
        tMaxY <- (Y + (stepY*1/2) - valid_inter_w[i,2])/dirY
        prev_t <- t_inter
        while (TRUE) {
          if (tMaxX < tMaxY) {
            id <- which(equals_plus(gd$cols,X) & equals_plus(gd$rows,Y))
            d1 <- sqrt(prev_t^2*(dirX^2 + dirY^2))
            d2 <- sqrt(tMaxX^2*(dirX^2 + dirY^2))
            p <- gen_coll_prob(d1,d2,obs_st[[t,1]],obs_st[[t,2]],cmin,cmax)
            prev_t <- tMaxX
            cp[id,t] <- cp[id,t] + p - cp[id,t]*p
            tMaxX <- tMaxX + tDeltaX
            X <- X + stepX
          }
          else {
            id <- which(equals_plus(gd$cols,X) & equals_plus(gd$rows,Y))
            d1 <- sqrt(prev_t^2*(dirX^2 + dirY^2))
            d2 <- sqrt(tMaxY^2*(dirX^2 + dirY^2))
            p <- gen_coll_prob(d1,d2,obs_st[[t,1]],obs_st[[t,2]],cmin,cmax)
            prev_t <- tMaxY
            cp[id,t] <- cp[id,t] + p - cp[id,t]*p
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
  gd$collprobs <- rowSums(cp)
  gd
}

generate_obs_ses <- function(ospeeds=c(4.0,8.0,12.0,16.0),oprobs=c(0.25,0.34,0.25,0.15),k=50000,st=1/4,ptr=1/30,t=1) {
  s <- c()
  n <- 1
  for (j in 1:k) {
    speed <- sample(ospeeds,1,TRUE,oprobs)*(ptr*st)
    a <- 0
    for (i in seq((ptr*st),t,by=(ptr*st))) {
      if (i == n*ptr) {
        speed <- sample(ospeeds,1,TRUE,oprobs)*(ptr*st)
      }
      a <- a + speed
    }
    s <- c(s,a)
  }
  c(mean(s),sd(s))
}

generate_ses <- function(ospeeds=c(4.0,8.0,12.0,16.0),oprobs=c(0.25,0.34,0.25,0.15),k=50000,st=1/4,ptr=1/30,pt=1,pspeed=8) {
  res <- list()
  for (i in 1:(pspeed*pt)) {
    res <- rbind(res,generate_obs_ses(ospeeds,oprobs,k,st,ptr,i/(pspeed*pt)))
  }
  res
}

get_coll_probs <- function(m,s) {
  p <- c(ptruncnorm(1,0,16,m,s) - ptruncnorm(0,0,16,m,s))
  for (i in seq(3,17,by=2)) {
    p <- c(p,ptruncnorm(i,0,16,m,s) - ptruncnorm(i-2,0,16,m,s))
  }
  p
}

gen_coll_prob <- function(d1,d2,m,s,mind=0,maxd=16) {
  ptruncnorm(d2,mind,maxd,m,s) - ptruncnorm(d1,mind,maxd,m,s)
}

plot_game_board <- function(states,tr=NULL,fill_data=0,fill_aux=NULL) {
  player <- data.frame(x=0,y=0)
  if (fill_data == 0) {
    if (is.null(tr)) {
      g <- ggplot() + 
        theme_bw() + 
        geom_tile(states,mapping=aes(x=cols,y=rows,fill=collprob),colour="white") +
        scale_fill_gradient(low="black",high="orange") +
        geom_point(player,mapping=aes(x,y),size=45/max(states$cols),color="white")
    }
    else {
      g <- ggplot() + 
        theme_bw() + 
        geom_tile(states,mapping=aes(x=cols,y=rows,fill=collprob),colour="white") +
        scale_fill_gradient(low="black",high="orange") +
        geom_point(player,mapping=aes(x,y),size=45/max(states$cols),color="white") 
      for (i in 2:nrow(tr)) {
        df <- data.frame(x1 = tr[i-1,1],x2 = tr[i,1],y1 = tr[i-1,2], y2 = tr[i,2])
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
        geom_point(player,mapping=aes(x,y),size=45/max(states$cols),color="white")
    }
    else {
      g <- ggplot() + 
        theme_bw() + 
        geom_tile(states,mapping=aes(x=cols,y=rows,fill=sqdistgoal),colour="white") +
        scale_fill_gradient(low="green",high="black") +
        geom_point(player,mapping=aes(x,y),size=45/max(states$cols),color="white") 
      for (i in 2:nrow(tr)) {
        df <- data.frame(x1 = tr[i-1,1],x2 = tr[i,1],y1 = tr[i-1,2], y2 = tr[i,2])
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
        geom_point(player,mapping=aes(x,y),size=45/max(states$cols),color="white")
    }
    else {
      g <- ggplot() + 
        theme_bw() + 
        geom_tile(states,mapping=aes(x=cols,y=rows,fill=fill_aux),colour="white") +
        scale_fill_gradient(low="blue",high="red") +
        geom_point(player,mapping=aes(x,y),size=45/max(states$cols),color="white") 
      for (i in 2:nrow(tr)) {
        df <- data.frame(x1 = tr[i-1,1],x2 = tr[i,1],y1 = tr[i-1,2], y2 = tr[i,2])
        g <- g + geom_segment(df,mapping=aes(x = x1,y = y1, xend = x2, yend = y2),
                              arrow=arrow(length=unit(0.15, "inches")))
      }
    }
  }
  g
}

create_cf <- function(b1,b2) {
  costfunc <- function(states,g) {
    cost <- b1*(states$sqdistgoal - min(states$sqdistgoal))/(max(states$sqdistgoal) - min(states$sqdistgoal)) + 
      b2*states$collprob
    g_id <- which(greater_equals_plus(g[1],(states$cols - 1/2)) & lesser_equals_plus(g[1],(states$cols + 1/2)) &
                    greater_equals_plus(g[2],(states$rows - 1/2)) & lesser_equals_plus(g[2],(states$rows + 1/2)))
    if (length(g_id) != 0) {
      cost[g_id[1]] <- 0
    }
    cost
  }
  costfunc
}

create_uniform_default_policy_from_grid <- function(states,g_id = NULL) {
  adj_mat <- apply(states, 1,function(pt)
    (pt["cols"] == states$cols & abs(pt["rows"] - states$rows) == 1) |
    (abs(pt["cols"] - states$cols) == 1 & pt["rows"] == states$rows))
  diag(adj_mat) <- 0
  adj_mat[g_id,] <- 0
  adj_mat[g_id,g_id] <- 1
  adj_mat <- adj_mat/rowSums(adj_mat)
  adj_mat
}

create_uniform_default_policy <- function(pspeed,pt,g_id = NULL) {
  grid_length <- pspeed*pt
  gd <- expand.grid(cols=-grid_length:grid_length,
                    rows=-grid_length:grid_length)
  create_uniform_default_policy_from_grid(gd,g_id)
}

sample_trajectory_FH <- function(init,t_mats,states) {
  trX <- c(init[1])
  trY <- c(init[2])
  idxs <- 1:nrow(states)
  for (t_mat in t_mats) {
    ci <- which(equals_plus(states$cols,trX[length(trX)]) & equals_plus(states$rows,trY[length(trY)]))
    ti <- sample(idxs,1,prob=t_mat[ci,])
    trX <- c(trX,states[ti,1])
    trY <- c(trY,states[ti,2])
  }
  data.frame(x=trX,y=trY)
}  

max_trajectory_FH <- function(init,t_mats,states) {
  initial <- which(equals_plus(states$cols,init[1]) & equals_plus(states$rows,init[2]))
  res <- traj_searcher(t_mats,initial,0)
  trX <- c(init[1])
  trY <- c(init[2])
  for (t in res[[1]]) {
    trX <- c(trX,states[t,1])
    trY <- c(trY,states[t,2])
  }
  data.frame(x=trX,y=trY)
}

traj_searcher <- function(t_mats,id,nl,n=1) {
  if (n > length(t_mats)) {
    return(list(c(),nl))
  }
  argm <- NULL
  m <- Inf
  for (i in which(t_mats[[n]][id,] > 0)) {
    res <- traj_searcher(t_mats,i,nl-log(t_mats[[n]][id,i]),n + 1)
    if (res[[2]] < m) {
      argm <- c(i,res[[1]])
      m <- res[[2]]
    }
  }
  return(list(argm,m))
}

tr_likelihood <- function(tr_idx,U) {
  L <- 1
  if (length(tr_idx) == 1) {
    return(1)
  }
  if (length(tr_idx) < 1) {
    return(0)
  }
  for (t in 1:(length(tr_idx) - 1)) {
    L <- L * U[[t]][tr_idx[t],tr_idx[t+1]]
  }
  return(L)
}

total_cost <- function(dp,costfunc,states,tr,g) {
  cost <- costfunc(states,g)
  ct <- cost[which(equals_plus(states$cols,tr[nrow(tr),1]) & equals_plus(states$rows,tr[nrow(tr),2]))]
  for (t in 1:(nrow(tr) - 1)) {
    idx <- which(equals_plus(states$cols,tr[t,1]) & equals_plus(states$rows,tr[t,2]))
    idxp <- which(equals_plus(states$cols,tr[t + 1,1]) & equals_plus(states$rows,tr[t + 1,2]))
    ct <- ct + (cost[idx] - log(dp[idx,idxp]))
  }
  ct
}

sample_total_costs <- function(init,t_mats,dp,costfunc,states,g,s=5000) {
  samps <- c()
  for (t in 1:s) {
    tr <- sample_trajectory_FH(init,t_mats,states)
    samps <- c(samps,total_cost(dp,costfunc,states,tr,g))
  }
  samps
}

LRL <- function(dp,costfunc,states,g,H = 8) {
  q <- costfunc(states,g)
  z_last <- exp(-q) 
  u_last <- matrix(0,length(q),length(q))
  G_last <- dp %*% z_last
  for(i in 1:length(q)){
    u_last[i,which(dp[i,]>0)] <- dp[i,which(dp[i,]>0)]*z_last[which(dp[i,]>0)]/G_last[i]
  }
  u <- list(u_last)
  z <- list(z_last)
  if (H == 1) {
    return(list(u,z))
  }
  for (t in (H-1):1) {
    z_last <- diag(exp(-q)) %*% dp %*% z_last
    u_last <- matrix(0,length(q),length(q))
    G_last <- dp %*% z_last
    for(i in 1:length(q)){
      u_last[i,which(dp[i,]>0)] <- dp[i,which(dp[i,]>0)]*z_last[which(dp[i,]>0)]/G_last[i]
    }
    u <- append(list(u_last),u)
    z <- append(list(z_last),z)
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

checkFutureTr <- function(c_cost,g,tr,board,dp,costf,eps) {
  t_cost <- total_cost(dp,costf,board,tr,g)
  if (t_cost - c_cost > eps) {
    return(TRUE)
  }
  return(FALSE)
}

LSES_planner <- function(p,
                         g,
                         O,
                         obs_st,
                         eps,
                         costf,
                         pspeed,
                         ospeeds,
                         oprobs,
                         pt=1,
                         max_time=500,
                         ptr=1/30,
                         tol=0.3) {
  omax <- max(ospeeds*pt)
  omin <- min(ospeeds*pt)
  prate <- pspeed*ptr
  O_pos <- data.frame(x=O[,1],y=O[,2],h=O[,3],t=0)
  p_pos <- data.frame(x=p[1],y=p[2],t=0)
  O_old <- O
  O <- ob_moves(O,ptr,ospeeds,oprobs)
  coll <- collision(O_old[,1],O_old[,2],O[,1],O[,2],p[1],p[2],tol)
  if (coll[[1]]) {
    print("Collision!")
    O_pos <- rbind(O_pos,data.frame(x=coll[[2]],y=coll[[3]],h=O[,3],t=1))
    p_pos <- rbind(p_pos,data.frame(x=p[1],y=p[2],t=1))
    return(list(p_pos,O_pos,FALSE))
  }
  exceeded_bounds <- which(sqrt(O[,1]^2 + O[,2]^2) > 50)
  O[exceeded_bounds,1] <- -O_old[exceeded_bounds,1]
  O[exceeded_bounds,2] <- -O_old[exceeded_bounds,2]
  O_pos <- rbind(O_pos,data.frame(x=O[,1],y=O[,2],h=O[,3],t=1))
  replan <- TRUE
  dp <- NULL
  current_p <- p
  t <- 1
  board <- create_board(current_p[1],current_p[2],g[1],g[2],O,obs_st,omin,omax,pspeed,pt)
  while (t < max_time - 1) {
    if (replan) {
      c_board <- board
      if (is.null(dp)) {
        dp <- create_uniform_default_policy(pspeed,pt)
      }
      g_id <- which(greater_equals_plus(g[1],(c_board$cols - 1/2)) & 
                      lesser_equals_plus(g[1],(c_board$cols + 1/2)) &
                      greater_equals_plus(g[2],(c_board$rows - 1/2)) & 
                      lesser_equals_plus(g[2],(c_board$rows + 1/2)))
      if (length(g_id) != 0) {
        dp <- create_uniform_default_policy(pspeed,pt,g_id[1]) 
      }
      res <- LRL(dp,costf,c_board,g,pspeed*pt)
      tr <- max_trajectory_FH(current_p,res[[1]],c_board)
      idx <- 2
      replan <- FALSE
    }
    dx <- tr[idx,1] - current_p[1]
    dy <- tr[idx,2] - current_p[2]
    if (dx > 0) {
      h <- 360
    }
    else if (dx < 0) {
      h <- 180
    }
    else {
      if (dy > 0) {
        h <- 90
      }
      else {
        h <- 270
      }
    }
    if (length(g_id) != 0) {
      if (equals_plus(tr[idx,1],c_board[g_id[1],1]) & equals_plus(tr[idx,2],c_board[g_id[1],2])) {
        h <- find_direction(current_p[1],current_p[2],g[1],g[2])
        g_dist <- point_dist(current_p[1],current_p[2],g[1],g[2])
        if (g_dist < prate) {
          prev_p <- current_p
          current_p <- g
          coll <- collision(prev_p[1],prev_p[2],current_p[1],current_p[2],O[,1],O[,2],tol)
          if (coll[[1]]) {
            print("Collision!")
            p_pos <- rbind(p_pos,data.frame(x=coll[[2]],y=coll[[3]],t=t))
            return(list(p_pos,O_pos,FALSE))
          }
          coll <- collision(prev_p[1],prev_p[2],current_p[1],current_p[2],g[1],g[2],tol)
          if (coll[[1]]) {
            print("Goal!")
            p_pos <- rbind(p_pos,data.frame(x=coll[[2]],y=coll[[3]],t=t))
            return(list(p_pos,O_pos,TRUE))
          }
          print("Error! Should have collided or reached goal!")
          return(list(p_pos,O_pos,FALSE))
        }
        for (i in 1:(g_dist %/% prate)) {
          prev_p <- current_p
          current_p[1] <- prev_p[1] + prate*cos_plus(h)
          current_p[2] <- prev_p[2] + prate*sin_plus(h)
          coll <- collision(prev_p[1],prev_p[2],current_p[1],current_p[2],O[,1],O[,2],tol)
          if (coll[[1]]) {
            print("Collision!")
            p_pos <- rbind(p_pos,data.frame(x=coll[[2]],y=coll[[3]],t=t))
            return(list(p_pos,O_pos,FALSE))
          }
          coll <- collision(prev_p[1],prev_p[2],current_p[1],current_p[2],g[1],g[2],tol)
          if (coll[[1]]) {
            print("Goal!")
            p_pos <- rbind(p_pos,data.frame(x=coll[[2]],y=coll[[3]],t=t))
            return(list(p_pos,O_pos,TRUE))
          }
          p_pos <- rbind(p_pos,data.frame(x=current_p[1],y=current_p[2],t=t))
          t <- t + 1
          O_old <- O
          O <- ob_moves(O,ptr,ospeeds,oprobs)
          coll <- collision(O_old[,1],O_old[,2],O[,1],O[,2],current_p[1],current_p[2],tol)
          if (coll[[1]]) {
            print("Collision!")
            O_pos <- rbind(O_pos,data.frame(x=coll[[2]],y=coll[[3]],h=O[,3],t=t))
            p_pos <- rbind(p_pos,data.frame(x=current_p[1],y=current_p[2],t=t))
            return(list(p_pos,O_pos,FALSE))
          }
          exceeded_bounds <- which(sqrt(O[,1]^2 + O[,2]^2) > 50)
          O[exceeded_bounds,1] <- -O_old[exceeded_bounds,1]
          O[exceeded_bounds,2] <- -O_old[exceeded_bounds,2]
          O_pos <- rbind(O_pos,data.frame(x=O[,1],y=O[,2],h=O[,3],t=t))
        }
        prev_p <- current_p
        current_p <- g
        coll <- collision(prev_p[1],prev_p[2],current_p[1],current_p[2],O[,1],O[,2],tol)
        if (coll[[1]]) {
          print("Collision!")
          p_pos <- rbind(p_pos,data.frame(x=coll[[2]],y=coll[[3]],t=t))
          return(list(p_pos,O_pos,FALSE))
        }
        coll <- collision(prev_p[1],prev_p[2],current_p[1],current_p[2],g[1],g[2],tol)
        if (coll[[1]]) {
          print("Goal!")
          p_pos <- rbind(p_pos,data.frame(x=coll[[2]],y=coll[[3]],t=t))
          return(list(p_pos,O_pos,TRUE))
        }
        print("Error! Should have collided or reached goal!")
        return(list(p_pos,O_pos,FALSE))
      }
    }
    for (i in 1:(1 %/% prate)) {
      prev_p <- current_p
      current_p[1] <- prev_p[1] + prate*cos_plus(h)
      current_p[2] <- prev_p[2] + prate*sin_plus(h)
      coll <- collision(prev_p[1],prev_p[2],current_p[1],current_p[2],O[,1],O[,2],tol)
      if (coll[[1]]) {
        print("Collision!")
        p_pos <- rbind(p_pos,data.frame(x=coll[[2]],y=coll[[3]],t=t))
        return(list(p_pos,O_pos,FALSE))
      }
      coll <- collision(prev_p[1],prev_p[2],current_p[1],current_p[2],g[1],g[2],tol)
      if (coll[[1]]) {
        print("Goal!")
        p_pos <- rbind(p_pos,data.frame(x=coll[[2]],y=coll[[3]],t=t))
        return(list(p_pos,O_pos,TRUE))
      }
      p_pos <- rbind(p_pos,data.frame(x=current_p[1],y=current_p[2],t=t))
      t <- t + 1
      O_old <- O
      O <- ob_moves(O,ptr,ospeeds,oprobs)
      coll <- collision(O_old[,1],O_old[,2],O[,1],O[,2],current_p[1],current_p[2],tol)
      if (coll[[1]]) {
        print("Collision!")
        O_pos <- rbind(O_pos,data.frame(x=coll[[2]],y=coll[[3]],h=O[,3],t=t))
        p_pos <- rbind(p_pos,data.frame(x=current_p[1],y=current_p[2],t=t))
        return(list(p_pos,O_pos,FALSE))
      }
      exceeded_bounds <- which(sqrt(O[,1]^2 + O[,2]^2) > 50)
      O[exceeded_bounds,1] <- -O_old[exceeded_bounds,1]
      O[exceeded_bounds,2] <- -O_old[exceeded_bounds,2]
      O_pos <- rbind(O_pos,data.frame(x=O[,1],y=O[,2],h=O[,3],t=t))
    }
    if (!equals_plus((1 / prate),(1 %/% prate))) {
      prev_p <- current_p
      current_p <- c(tr[idx,1],tr[idx,2])
      coll <- collision(prev_p[1],prev_p[2],current_p[1],current_p[2],O[,1],O[,2],tol)
      if (coll[[1]]) {
        print("Collision!")
        p_pos <- rbind(p_pos,data.frame(x=coll[[2]],y=coll[[3]],t=t))
        return(list(p_pos,O_pos,FALSE))
      }
      coll <- collision(prev_p[1],prev_p[2],current_p[1],current_p[2],g[1],g[2],tol)
      if (coll[[1]]) {
        print("Goal!")
        p_pos <- rbind(p_pos,data.frame(x=coll[[2]],y=coll[[3]],t=t))
        return(list(p_pos,O_pos,TRUE))
      }
      p_pos <- rbind(p_pos,data.frame(x=current_p[1],y=current_p[2],t=t))
      t <- t + 1
      O_old <- O
      O <- ob_moves(O,ptr,ospeeds,oprobs)
      coll <- collision(O_old[,1],O_old[,2],O[,1],O[,2],current_p[1],current_p[2],tol)
      if (coll[[1]]) {
        print("Collision!")
        O_pos <- rbind(O_pos,data.frame(x=coll[[2]],y=coll[[3]],h=O[,3],t=t))
        p_pos <- rbind(p_pos,data.frame(x=current_p[1],y=current_p[2],t=t))
        return(list(p_pos,O_pos,FALSE))
      }
      exceeded_bounds <- which(sqrt(O[,1]^2 + O[,2]^2) > 50)
      O[exceeded_bounds,1] <- -O_old[exceeded_bounds,1]
      O[exceeded_bounds,2] <- -O_old[exceeded_bounds,2]
      O_pos <- rbind(O_pos,data.frame(x=O[,1],y=O[,2],h=O[,3],t=t))
    }
    else {
      current_p <- c(tr[idx,1],tr[idx,2])
    }
    idx <- idx + 1
    if (idx > nrow(tr)) {
      replan <- TRUE
    }
    else {
      c_cost <- total_cost(dp,costf,c_board,tr[(idx-1):nrow(tr),],g)
      board <- create_board(tr[idx,1],tr[idx,2],g[1],g[2],O,obs_st,omin,omax,pspeed,pt)
      replan <- checkFutureTr(c_cost,g,tr[(idx-1):nrow(tr),],board,dp,costf,eps)
    }
  }
  print("Ran out of time!")
  return(list(p_pos,O_pos,FALSE))
}