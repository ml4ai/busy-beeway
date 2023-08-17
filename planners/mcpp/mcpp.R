library("SES.R")
mcpp <- function(cx,cy,gx,gy,OX,OY,OH,tol) {
  pcx <- cx
  pcy <- cy
  while(!collision(pcx,pcy,cx,cy,gx,gx,tol)) {
    GA <- search(cx,cy,OX,OY,OH)
    pcx <- cx
    pcy <- cy
    cx <- GA[[1]]
    cy <- GA[[2]]
  }
}