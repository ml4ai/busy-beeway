library("data.tree")
source("~/busy-beeway/planners/ses/SES.R")

simulateV <- function(tree,OX,OY,OH) {
  atree <- SelectAction(tree)
  SimulateQ(tree,atree)
  tree$N <- tree$N + 1
  tree$V <- tree$V
}

search <- function(cx,cy,OX,OY,OH,time_limit) {
  tree <- Node$new(0)
  tree$x <- cx
  tree$y <- cy
  tree$N <- 0
  tree$n <- c()
  tree$V <- 0
  tree$Q <- c()
  start <- Sys.time()
  while(difftime(Sys.time(),start,units="secs") < time_limit) {
    tree <- simulateV(tree,OX,OY,OH)
  }
  tree <-tree$root
  argmaxQ <- tree$children[[which.max(tree$Q)]]
  list(argmaxQ$x,argmaxQ$y)
} 

mcpp <- function(cx,cy,gx,gy,OX,OY,OH,tol,time_limit) {
  pcx <- cx
  pcy <- cy
  while(!collision(pcx,pcy,cx,cy,gx,gx,tol)) {
    GA <- search(cx,cy,OX,OY,OH,time_limit)
    pcx <- cx
    pcy <- cy
    cx <- GA[[1]]
    cy <- GA[[2]]
  }
}