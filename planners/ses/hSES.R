source("~/busy-beeway/planners/ses/SES.R")

hgrowGoalTree <- function(pX,pY,tgX,tgY,OX,OY,OH,pspeed,XE,ptr,sth,tol=0.3) {
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
  while (!collision(pcX,pcY,cX,cY,tgX,tgY,tol)[[1]]) {
    GA <- getAction(cX,cY,tgX,tgY,pspeed)
    pcX <- cX
    pcY <- cY
    cX <- cX + ptr*GA[1]
    cY <- cY + ptr*GA[2]
    collProb <- getCollisionProb(pcX,pcY,cX,cY,XE,OX,OY,OH,k*ptr,tol)
    Tree <- Tree$AddChild(Tree$totalCount)
    Tree$x <- cX
    Tree$y <- cY
    Tree$t <- k
    Tree$p <- collProb
    k <- k + 1
    if (k*ptr > sth) {
      break
    }
  }
  Tree <- Tree$root
  Tree
}

hgrowFullTree <- function(Tree,pX,pY,tgX,tgY,OX,OY,OH,pspeed,XE,K,ptr,sth,gb,tol){
  for (k in 1:K) {
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
    if ((t + 1)*ptr > sth) {
      next
    }
    GA <- getAction(qnearX,qnearY,qrandX,qrandY,pspeed)
    qnewX <- qnearX + ptr*GA[1]
    qnewY <- qnearY + ptr*GA[2]
    collProb <- getCollisionProb(qnearX,qnearY,qnewX,qnewY,XE,OX,OY,OH,(t+1)*ptr,tol)
    n <- near[[1]]$AddChild(Tree$totalCount)
    n$x <- qnewX
    n$y <- qnewY
    n$t <- t + 1
    n$p <- collProb
  }
  Tree
}

hgetMinWPath <- function(Tree,tgX,tgY,greed,safety,p) {
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
    R <- hgetMinWPath(c,tgX,tgY,greed,safety,p)
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

hgetZeroCollPath <- function(Tree,tgX,tgY,greed) {
  if (Tree$isLeaf) {
    return(Tree)
  }
  
  mT <- NULL
  maxL <- -Inf
  for (c in Tree$children) {
    if (isTRUE(all.equal(c$p,0))) {
      R <- hgetZeroCollPath(c)
      if ((R$level - 1) > maxL) {
        maxL <- R$level - 1
        mT <- R
      }
    }
  }
  if (is.null(mT)) {
    mT <- Tree
  }
  w <- greed*pdist(Tree$x,Tree$y,tgX,tgY)
  return(list(mT,w))
}

hgetPathFromTree <- function(Tree,tgX,tgY,greed,safety) {
  RW <- hgetMinWPath(Tree,tgX,tgY,greed,safety,0)
  if (is.null(RW[1])) {
    RW <- hgetZeroCollPath(Tree,tgX,tgY,greed)
  }
  R <- RW[[1]]
  minW <- RW[[2]]
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
  list(ph,minW)
}

hcheckFutureNodes <- function(pX,pY,tgX,tgY,P,W,t,XE,OX,OY,OH,ptr,greed,lambda,safety,tol) {
  if (t > length(P)) {
    return(T)
  }
  if (length(P[t:length(P)]) < safety) {
    return(T)
  }
  prevx <- pX
  prevy <- pY
  ct <- 1
  pr <- 0
  for (p in P[t:length(P)]) { 
    pr <- pr + getCollisionProb(prevx,prevy,p[1],p[2],XE,OX,OY,OH,ct*ptr,tol)
    prevx <- p[1]
    prevy <- p[2]
    ct <- ct + 1
  }
  w <- (greed*pdist(prevx,prevy,tgX,tgY)) + (pr/(length(t:length(P))))
  if (w - W > lambda) {
    return(T)
  }
  F
}

hSES_planning <- function(XE,
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
                         max_time=500,
                         K=1500,
                         gb=0.01,
                         greed=0.001,
                         lambda = 0.1,
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
      Tree <- Node$new(0)
      Tree$x <- pX
      Tree$y <- pY
      Tree$t <- 0
      Tree$p <- 0
      #Tree <- hgrowGoalTree(pX,pY,tgX,tgY,OX,OY,OH,pspeed,XE,ptr,sth,tol)
      Tree <- hgrowFullTree(Tree,pX,pY,tgX,tgY,OX,OY,OH,pspeed,XE,K,ptr,sth,gb,tol)
      PW <- hgetPathFromTree(Tree,tgX,tgY,greed,safety)
      P <- PW[[1]]
      W <- PW[[2]]
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
    reGrowTree <- hcheckFutureNodes(pX,pY,tgX,tgY,P,W,(t-tlp) + 2,XE,OX,OY,OH,ptr,greed,lambda,safety,tol)
    if (collision(prevX,prevY,pX,pY,tgX,tgY,tol)[[1]]) {
      reGrowTree <- T
    }
  }
  print("Out of time!")
  PD <- rbind(PD,data.frame(x=pX,y=pY,t=max_time+1))
  return(list(PD,O))
}