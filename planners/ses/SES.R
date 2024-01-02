library("data.tree")
library("truncnorm")
library("ggplot2")

#sine and cosine helper functions
cos_plus <- function(degrees) {
  if (isTRUE(all.equal(degrees, 90)) | isTRUE(all.equal(degrees, 270))) {
    return(0)
  }
  cos(degrees*(pi/180))
}

sin_plus <- function(degrees) {
  if (isTRUE(all.equal(degrees, 360)) | isTRUE(all.equal(degrees, 180))) {
    return(0)
  }
  sin(degrees*(pi/180))
}

updateObstacleDynamics <- function(h,sts,speeds,probs){
  speed <- sts*sample(speeds,1,prob=probs)
  dX <- speed*cos_plus(h)
  dY <- speed*sin_plus(h)
  c(dX,dY)
}

pdist <- function(x1,y1,x2,y2) {
  sqrt((x2 - x1)^2 + (y2 - y1)^2)
}

#point and vec
pdistVec <- function(x1,y1,X,Y) {
  sqrt((X - x1)^2 + (Y - y1)^2)
}

processEnsemble <- function(X,m) {
  x <- c()
  for (i in 1:length(X)) {
    x <- append(x,X[[i]][[m]][1])
  }
  list(mean(x),sd(x))
}

SES_plot <- function(XE,t,s=5000) {
  mu <- XE[[t,2]]
  sd <- XE[[t,3]]
  x <- rnorm(s,mu,sd)
  y <- rep(0,s)
  h <- dnorm(x,mu,sd)
  D <- data.frame(x=x,y=y,h=h)
  g <- ggplot(D,aes(x,y,color=h)) + geom_point() + scale_color_continuous(low="blue",high="red")
  print(g)
}

SES <- function(sth,sts,soe,ptr,speeds,probs) {
  X <- list()
  for (i in 0:soe) {
    X[[i+1]] <- list()
    X[[i+1]][[1]] <- c(0,0)
    x_bar <- X[[i+1]][[1]]
    k <- 1
    for (t in seq(0,sth, by=sts)) {
      f <- updateObstacleDynamics(0,sts,speeds,probs)
      x_bar <- x_bar + ptr*f
      if (isTRUE(all.equal(t,k*ptr))) {
        X[[i+1]][[k+1]] <- x_bar
        k <- k + 1
      }
    }
  }
  XE <- list()
  for (m in 0:(sth/ptr)) {
    XE <- rbind(XE,c(m*ptr,processEnsemble(X,m + 1)))
  }
  XE
}

find_direction <- function(x1,y1,x2,y2) {
  y <- y2 - y1
  x <- x2 - x1
  if (isTRUE(all.equal(y,0)) & x > 0) {
    h <- 360
  }
  else if (y > 0 | isTRUE(all.equal(y,0))) {
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

getAction <- function(cX,cY,tgX,tgY,speed) {
  h <- find_direction(cX,cY,tgX,tgY)
  dX <- speed*cos_plus(h)
  dY <- speed*sin_plus(h)
  c(dX,dY)
}

#vector, val
allequalvec <- function(X,x) {
  y <- c()
  for (i in X) {
     y <- append(y, isTRUE(all.equal(i,X)))
  }
  y
}

#vector, vector
allequalvecs <- function(X,Y) {
  z <- c()
  for (i in 1:length(X)) {
    z <- append(z,isTRUE(all.equal(X[i],Y[i])))
  }
  z
}
getCollisionProb <- function(px,py,x,y,XE,OX,OY,OH,m,tol=0.3) {
  E <- XE[which(XE[,1] == m),]
  a <- (x - px)
  b <- (y - py)
  abdistsq <- a^2 + b^2
  pox <- (px - OX)
  poy <- (py - OY)
  coh <- cos(OH*(pi/180))
  soh <- sin(OH*(pi/180))
  
  denom1 <- a*coh + b*soh
  denom2 <- b*coh - a*soh
  nom <- 2*tol*sqrt(abdistsq)
  
  s1 <- (a*pox + b*poy)/denom1
  s2 <- (b*pox - a*poy)/denom2
  ub1 <- c()
  lb1 <- c()
  ub1[which(denom1 > 0)] <- ((abdistsq/denom1) + s1)[which(denom1 > 0)]
  lb1[which(denom1 > 0)] <- s1[which(denom1 > 0)]
  ub1[which(denom1 < 0)] <- s1[which(denom1 < 0)]
  lb1[which(denom1 < 0)] <- ((abdistsq/denom1) + s1)[which(denom1 <0)]
  
  sb <- ((nom/denom2) + s2)
  nsb <- ((-nom/denom2) + s2)
  ub2 <- c()
  lb2 <- c()
  
  ub2[which(denom2 > 0)] <- sb[which(denom2 > 0)]
  ub2[which(denom2 < 0)] <- nsb[which(denom2 < 0)]
  
  lb2[which(denom2 > 0)] <- nsb[which(denom2 > 0)]
  lb2[which(denom2 < 0)] <- sb[which(denom2 < 0)]
  
  ub2[which(ub2 > ub1)] <- ub1[which(ub2 > ub1)]
  ub2[which(ub2 < lb1)] <- lb1[which(ub2 < lb1)]
  
  lb2[which(lb2 < lb1)] <- lb1[which(lb2 < lb1)]
  lb2[which(lb2 > ub1)] <- ub1[which(lb2 > ub1)]
  
  sum(pnorm(ub2,E[[2]],E[[3]]) - pnorm(lb2,E[[2]],E[[3]]))
}

growGoalTree <- function(pX,pY,tgX,tgY,OX,OY,OH,pspeed,XE,ptr,Praccept,tol=0.3) {
  pcX <- pX
  pcY <- pY
  cX <- pX
  cY <- pY
  k <- 1
  Tree <- Node$new(0)
  Tree$x <- cX
  Tree$y <- cY
  Tree$t <- 0
  Tree$p <- 0
  growFullTree <- F
  while (!collision(pcX,pcY,cX,cY,tgX,tgY,tol)[[1]]) {
    GA <- getAction(cX,cY,tgX,tgY,pspeed)
    pcX <- cX
    pcY <- cY
    cX <- cX + ptr*GA[1]
    cY <- cY + ptr*GA[2]
    collProb <- getCollisionProb(pcX,pcY,cX,cY,XE,OX,OY,OH,k*ptr,tol)
    if (collProb < Praccept | isTRUE(all.equal(collProb,Praccept))) {
      Tree <- Tree$AddChild(Tree$totalCount)
      Tree$x <- cX
      Tree$y <- cY
      Tree$t <- k
      Tree$p <- collProb
      k <- k + 1
    }
    else {
      growFullTree <- T
      Tree <- Tree$root
      return(list(growFullTree,Tree))
    }
  }
  Tree <- Tree$root
  list(growFullTree,Tree)
}

nearest_vertex <- function(x,y,Tree,mTree = NULL,minD = Inf) {
  d <- pdist(x,y,Tree$x,Tree$y)
  if (d < minD) {
    mTree <- Tree
    minD <- d
  }
  if (Tree$isLeaf) {
    return(list(mTree,minD))
  }
  for (i in Tree$children) {
    R <- nearest_vertex(x,y,i,mTree,minD)
    if (R[[2]] < minD) {
      mTree <- R[[1]]
      minD <- R[[2]]
    }
  }
  return(list(mTree,minD))
}

growFullTree <- function(Tree,pX,pY,tgX,tgY,OX,OY,OH,pspeed,XE,K,ptr,gb,tol){
  for (k in 1:K) {
    while(T) {
      if (runif(1) < gb) {
        qrandX <- tgX
        qrandY <- tgY
      }
      else {
        R <- pdist(pX,pY,tgX,tgY)
        r <- R*sqrt(runif(1))
        theta <- runif(1)*2*pi
        qrandX <- pX + r * cos(theta)
        qrandY <- pY + r * sin(theta)
      }
      near <- nearest_vertex(qrandX,qrandY,Tree)
      qnearX <- near[[1]]$x
      qnearY <- near[[1]]$y
      t <- near[[1]]$t
      if ((t + 1)*ptr > XE[[length(XE[,1]),1]]) {
        break
      }
      GA <- getAction(qnearX,qnearY,qrandX,qrandY,pspeed)
      qnewX <- qnearX + ptr*GA[1]
      qnewY <- qnearY + ptr*GA[2]
      collProb <- getCollisionProb(qnearX,qnearY,qnewX,qnewY,XE,OX,OY,OH,(t+1)*ptr,tol)
      if (collProb < Praccept | isTRUE(all.equal(collProb,Praccept))) {
        n <- near[[1]]$AddChild(Tree$totalCount)
        n$x <- qnewX
        n$y <- qnewY
        n$t <- t + 1
        n$p <- collProb
        break
      }
    }
  }
  Tree
}

getMinWPath <- function(Tree,tgX,tgY,greed,safety,p) {
  p <- p + Tree$p
  if (Tree$isLeaf) {
    if ((Tree$level - 1) > safety | isTRUE(all.equal((Tree$level - 1), safety))) {
      w <- (greed*pdist(Tree$x,Tree$y,tgX,tgY)) + (p/(Tree$level - 1))
      return(list(Tree,w))
    }
    return(list(NULL,-1))
  }
  mT <- NULL
  minW <- Inf
  for (c in Tree$children) {
    R <- getMinWPath(c,tgX,tgY,greed,safety,p)
    if (!is.null(R[[1]])) {
      if (R[[2]] < minW) {
        minW <- R[[2]]
        mT <- R[[1]]
      }
    }
  }
  if (is.null(mT)) {
    return(list(NULL,-1))
  }
  return(list(mT,minW))
}

getZeroCollPath <- function(Tree) {
  if (Tree$isLeaf) {
    return(Tree)
  }
  
  mT <- NULL
  maxL <- -Inf
  for (c in Tree$children) {
    if (isTRUE(all.equal(c$p,0))) {
      R <- getZeroCollPath(c)
      if ((R$level - 1) > maxL) {
        maxL <- R$level - 1
        mT <- R
      }
    }
  }
  if (is.null(mT)) {
    return(Tree)
  }
  return(mT)
}

getPathFromTree <- function(Tree,tgX,tgY,greed,safety) {
  RW <- getMinWPath(Tree,tgX,tgY,greed,safety,0)
  R <- RW[[1]]
  if (is.null(R)) {
    R <- getZeroCollPath(Tree)
  }
  PS <- R$path
  if (isTRUE(all.equal(length(PS),1))) {
    return(NULL)
  }
  PS <- PS[2:length(PS)]
  ph <- list()
  for (p in PS) {
    Tree <- Tree[[p]]
    ph <- append(ph,list(c(Tree$x,Tree$y)))
  }
  ph
}

ob_moves <- function(oX,oY,oH,ptr,speeds,probs) {
  speed <- sample(speeds,1,prob=probs)
  oX <- oX + (ptr*speed*cos(oH*(pi/180)))
  oY <- oY + (ptr*speed*sin(oH*(pi/180)))
  list(oX,oY)
}

checkFutureNodes <- function(pX,pY,P,t,XE,OX,OY,OH,ptr,Praccept,safety,tol) {
  if (t > length(P)) {
    return(T)
  }
  if (length(P[t:length(P)]) < safety) {
    return(T)
  }
  prevx <- pX
  prevy <- pY
  ct <- 1
  for (p in P[t:length(P)]) { 
    collProb <- getCollisionProb(prevx,prevy,p[1],p[2],XE,OX,OY,OH,ct*ptr,tol)
    if (collProb > Praccept) {
      return(T)
    }
    prevx <- p[1]
    prevy <- p[2]
    ct <- ct + 1
  }
  F
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

SES_planning <- function(XE,
                         pX,
                         pY,
                         gX,
                         gY,
                         OX,
                         OY,
                         OH,
                         pspeed,
                         ospeeds,
                         oprobs,
                         ptr,
                         sth,
                         Praccept=0.01,
                         max_time=500,
                         K=1500,
                         gb=0.01,
                         greed=0.001,
                         safety=10,
                         tol=0.3) {
  O <- data.frame(x=OX,y=OY,t=0)
  PD <- data.frame(x=pX,y=pY,t=0)
  reGrowTree <- T
  new_ob <- ob_moves(OX,OY,OH,ptr,ospeeds,oprobs)
  prevOX <- OX
  prevOY <- OY
  OX <- new_ob[[1]]
  OY <- new_ob[[2]]
  coll <- collision(prevOX,prevOY,OX,OY,pX,pY,tol)
  if (coll[[1]]) {
    print("Collision!")
    O <- rbind(O,data.frame(x=coll[[2]],y=coll[[3]],t=1))
    PD <- rbind(PD,data.frame(x=pX,y=pY,t=1))
    return(list(PD,O))
  }
  OX[which(sqrt(OX^2 + OY^2) > 50)] <- -OX[which(sqrt(OX^2 + OY^2) > 50)]
  OY[which(sqrt(OX^2 + OY^2) > 50)] <- -OY[which(sqrt(OX^2 + OY^2) > 50)]
  O <- rbind(O,data.frame(x=OX,y=OY,t=1))
  i <- 1
  for (t in 1:(max_time - 1)) {
    if (reGrowTree) {
      tlp <- t
      GP <- getAction(pX,pY,gX,gY,pspeed)
      tgX <- pX + sth*GP[1]
      tgY <- pY + sth*GP[2]
      GFTT <- growGoalTree(pX,pY,tgX,tgY,OX,OY,OH,pspeed,XE,ptr,Praccept,tol)
      Tree <- GFTT[[2]]
      if (GFTT[[1]]) {
        Tree <- growFullTree(Tree,pX,pY,tgX,tgY,OX,OY,OH,pspeed,XE,K,ptr,gb,tol)
      }
      P <- getPathFromTree(Tree,tgX,tgY,greed,safety)
      if (is.null(P)) {
        PD <- rbind(PD,data.frame(x=pX,y=pY,t=t))
        print("No path can be found!")
        return(list(PD,O))
      }
    }
    prevX <- pX
    prevY <- pY
    pX <- P[[(t-tlp) + 1]][1]
    pY <- P[[(t-tlp) + 1]][2]
    coll <- collision(prevX,prevY,pX,pY,OX,OY,tol)
    if (coll[[1]]) {
      print("Collision!")
      PD <- rbind(PD,data.frame(x=coll[[2]],y=coll[[3]],t=t))
      return(list(PD,O))
    }
    coll <- collision(prevX,prevY,pX,pY,gX,gY,tol)
    if (coll[[1]]) {
      print("Goal!")
      PD <- rbind(PD,data.frame(x=coll[[2]],y=coll[[3]],t=t))
      return(list(PD,O))
    }
    PD <- rbind(PD,data.frame(x=pX,y=pY,t=t))
    new_ob <- ob_moves(OX,OY,OH,ptr,ospeeds,oprobs)
    prevOX <- OX
    prevOY <- OY
    OX <- new_ob[[1]]
    OY <- new_ob[[2]]
    coll <- collision(prevOX,prevOY,OX,OY,pX,pY,tol)
    if (coll[[1]]) {
      print("Collision!")
      O <- rbind(O,data.frame(x=coll[[2]],y=coll[[3]],t=t+1))
      PD <- rbind(PD,data.frame(x=pX,y=pY,t=t+1))
      return(list(PD,O))
    }
    OX[which(sqrt(OX^2 + OY^2) > 50)] <- -OX[which(sqrt(OX^2 + OY^2) > 50)]
    OY[which(sqrt(OX^2 + OY^2) > 50)] <- -OY[which(sqrt(OX^2 + OY^2) > 50)]
    O <- rbind(O,data.frame(x=OX,y=OY,t=t + 1))
    reGrowTree <- checkFutureNodes(pX,pY,P,(t-tlp) + 2,XE,OX,OY,OH,ptr,Praccept,safety,tol)
    if (collision(prevX,prevY,pX,pY,tgX,tgY,tol)[[1]]) {
      reGrowTree <- T
    }
  }
  print("Out of time!")
  PD <- rbind(PD,data.frame(x=pX,y=pY,t=max_time+1))
  return(list(PD,O))
}