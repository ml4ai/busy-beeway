library("ggplot2")
library("gganimate")
library("truncnorm")

min_max_rescale <- function(v,a=0,b=1) {
  v_min <- min(v)
  v_max <- max(v)
  if (equals_plus(v_max - v_min,0)) {
    v <- rep(a,length(v))
  }
  else {
    v <- a + (((v - v_min)*(b-a))/(v_max - v_min))
  }
  v
}

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

ccw <- function(ax,ay,bx,by,cx,cy) {
  (bx - ax)*(cy - ay) - (by - ay)*(cx - ax)
}

intersects <- function(ax,ay,bx,by,cx,cy,dx,dy) {
  ccw(ax,ay,bx,by,cx,cy)*ccw(ax,ay,bx,by,dx,dy) < 0 & ccw(cx,cy,dx,dy,ax,ay)*ccw(cx,cy,dx,dy,bx,by) < 0
}

ray_intersects <- function(as_x,as_y,ad_x,ad_y,bs_x,bs_y,bd_x,bd_y) {
  dx <- bs_x - as_x
  dy <- bs_y - as_y
  
  det <- bd_x * ad_y - bd_y * ad_x
  u_num <- dy * bd_x - dx * bd_y
  v_num <- dy * ad_x - dx * ad_y
  normal_int <- greater_equals_plus(u_num * det,0) & greater_equals_plus(v_num * det,0) & !equals_plus(det,0)
  coincide <- equals_plus(u_num,0) & equals_plus(v_num,0) & equals_plus(det,0)
  collide <- greater_equals_plus((bs_x - as_x)/ad_x,0) & equals_plus((bs_x - as_x)/ad_x,(bs_y - as_y)/ad_y)
  normal_int | (coincide & collide)
}

frac <- function(x) {
  if (x > 0) {
    x - floor(x)
  }
  else {
    1 - x + floor(x)
  }
}

point_dist_sq <- function(x1,y1,x2,y2) {
  (x2 - x1)^2 + (y2 - y1)^2
}

point_dist <- function(x1,y1,x2,y2) {
  sqrt(point_dist_sq(x1,y1,x2,y2))
}

# Randomly samples in a circle
runif_circle <- function(n,R,center = c(0,0)) {
  r <- R * sqrt(runif(n))
  theta <- runif(n,0,360) 
  if (n == 1) {
    res <- c(center[1] + r * cos_plus(theta),center[2] + r * sin_plus(theta))
  }
  else {
    res <- list(center[1] + r * cos_plus_vec(theta),center[2] + r * sin_plus_vec(theta))
  }
  res
}

# Randomly samples on the circumference of a circle
runif_on_circle <- function(n,r,center = c(0,0)) {
  theta <- runif(n,0,360) 
  if (n == 1) {
    res <- c(center[1] + r * cos_plus(theta),center[2] + r * sin_plus(theta))
  }
  else {
    res <- list(center[1] + r * cos_plus_vec(theta),center[2] + r * sin_plus_vec(theta))
  }
  res
}

gen_coll_prob <- function(d1,d2,m,s,mind=0,maxd=16) {
  s <- ptruncnorm(d2,mind,maxd,m,s) - ptruncnorm(d1,mind,maxd,m,s)
  s
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

#2*tol because both have radius of 0.3
pointCollide <- function(x1,y1,x2,y2,tol) {
  (x1-x2)^2 + (y1-y2)^2 < (2*tol)^2
}

collision <- function(prevX,prevY,curX,curY,px,py,tol = 0.3) {
  CP <- closestpointonline(prevX,prevY,curX,curY,px,py)
  cpX <- CP[[1]]
  cpY <- CP[[2]]
  list(any(pointCollide(cpX,cpY,px,py,tol)),cpX,cpY)
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

plot_trajectory <- function(dat) {
  ggplot(data=dat) + geom_path(mapping=aes(x=posX,y=posY))
}

plot_weight_dist <- function(res) {
  ggplot() + 
    geom_point(res,mapping=aes(x=b1,y=b2,color=rl)) + 
    scale_color_gradient(low="orange",high="blue")
}

plot_tuning_dist <- function(res) {
  ggplot() + 
    geom_point(res,mapping=aes(x=delT,y=rho,color=rl)) + 
    scale_color_gradient(low="orange",high="blue")
}

animate_sim <- function(D,i) {
  P <- D[[i,1]]
  O <- D[[i,2]]
  g <- data.frame(x=D[[i,4]][1],y=D[[i,4]][2])
  ggplot(O,aes(x,y))+
    geom_text(aes(label='W'),color="darkred",size=3) +
    geom_point(data=P,aes(x,y),size=3) +
    geom_text(data=g,aes(x,y,label='G'),color = "green",size=5) +
    scale_x_continuous(limits = c(-50,50)) +
    scale_y_continuous(limits = c(-50,50)) +
    labs(title = 'Time Step: {frame_time}') +
    transition_time(t)
}