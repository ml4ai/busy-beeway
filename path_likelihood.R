source("~/busy-beeway/planners/ses/hSES.R")

pX <- 25
pY <- 0
gX <- 0
gY <- 0
regen_obs <- T
regen_sample <- T
regen_optimal <- T
if (regen_obs) {
  oX <- runif(50,-20,20)
  oY <- runif(50,-20,20)
  oH <- runif(50,1,360)
}
pspeed <- 8
ospeeds <- c(4.0,8.0,12.0,16.0)
oprobs <- c(0.25,0.34,0.25,0.15)
ptr <- 1/30
max_time <- 500
K <- 5000
gb <- 0.01
greed <- 0.001
safety <- 10
tol <- 0.3
sth <- 1
lambda <- 0.01

GP <- getAction(pX,pY,gX,gY,pspeed)
tgX <- pX + sth*GP[1]
tgY <- pY + sth*GP[2]

if (!exists("XE")) {
  XE <- SES(sth,(1/30)*(1/10),5000,ptr,ospeeds,oprobs)
}

if (!exists("Tree")) {
  Tree <- Node$new(0)
  Tree$x <- pX
  Tree$y <- pY
  Tree$t <- 0
  Tree$p <- 0
  Tree <- hgrowFullTree(Tree,pX,pY,tgX,tgY,oX,oY,oH,pspeed,XE,K,ptr,sth,gb,tol)
}

generate_random_path <- function(Tree,safety) {
  #Tree <- hgrowGoalTree(pX,pY,tgX,tgY,OX,OY,OH,pspeed,XE,ptr,sth,tol)
  PS <- NULL
  while(length(PS) < safety) {
    stree <- Tree
    while(!stree$isLeaf) {
      stree <- sample(stree$children,1)[[1]]
    }
    PS <- stree$path
  }
  PS <- PS[2:length(PS)]
  ph <- list()
  for (p in PS) {
    Tree <- Tree[[p]]
    ph <- append(ph,list(c(Tree$x,Tree$y)))
  }
  ph
}

generate_optimal_path <- function(Tree,tgX,tgY,greed,safety) {
  #Tree <- hgrowGoalTree(pX,pY,tgX,tgY,OX,OY,OH,pspeed,XE,ptr,sth,tol)
  PW <- hgetPathFromTree(Tree,tgX,tgY,greed,safety)
  ph <- PW[[1]]
  ph
}

if (!exists("sample_path") | regen_sample) {
  sample_path <- generate_random_path(Tree,safety)
}

if (!exists("optimal_path") | regen_optimal) {
  optimal_path <- generate_optimal_path(Tree,tgX,tgY,greed,safety)
}

get_cost <- function(pX,pY,tgX,tgY,P,XE,OX,OY,OH,ptr,greed,safety,tol) {
  prevx <- pX
  prevy <- pY
  ct <- 1
  pr <- 0
  for (p in P) { 
    pr <- pr + getCollisionProb(prevx,prevy,p[1],p[2],XE,OX,OY,OH,ct*ptr,tol)
    prevx <- p[1]
    prevy <- p[2]
    ct <- ct + 1
  }
  (greed*pdist(prevx,prevy,tgX,tgY)) + (pr/(length(P)))
}