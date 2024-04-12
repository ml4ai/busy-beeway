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

runif_circle <- function(n,R,center = c(0,0)) {
  r <- R * sqrt(runif(n))
  theta <- runif(n,1,360) 
  if (n == 1) {
    res <- c(center[1] + r * cos_plus(theta),center[2] + r * sin_plus(theta))
  }
  else {
    res <- list(center[1] + r * cos_plus_vec(theta),center[2] + r * sin_plus_vec(theta))
  }
  res
}