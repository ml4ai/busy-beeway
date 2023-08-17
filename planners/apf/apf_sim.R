source("busybee.R")
source("APF.R")
library("ggplot2")
library("gganimate")

pX <- 25
pY <- 0
gX <- 0
gY <- 0
oX <- runif(50,-20,20)
oY <- runif(50,-20,20)
oH <- runif(50,1,360)

O <- data.frame(x=oX,y=oY,t=0)
P <- data.frame(x=pX,y=pY,t=0, s=fsm$currentState)
I <- list(px=pX,py=pY,ox=oX,oy=oY,gx=gX,gy=gY,d=5,a=50)
x <- 1
repeat {
  new_ob <- ob_moves(oX,oY,oH)
  oX <- new_ob[[1]]
  oY <- new_ob[[2]]
  I <- list(px=pX,py=pY,ox=oX,oy=oY,gx=gX,gy=gY,d1=5,a1=50,d2=3,a2=30)
  fsm <- transition(fsm,I)
  if (fsm$currentState == 'PROGRESS') {
    pp <- decideAPF(pX,pY,gX,gY,oX,oY,1,0)
  }
  else if (fsm$currentState == 'DODGE') {
    pp <- decideAPF(pX,pY,gX,gY,oX,oY,1,.5)
  }
  else {
    pp <- decideAPF(pX,pY,gX,gY,oX,oY,1,1)
  }
  pX <- pp[1]
  pY <- pp[2]
  oN <- data.frame(x=oX,y=oY,t=x)
  pN <- data.frame(x=pX,y=pY,t=x,s=fsm$currentState)
  O <- rbind(O,oN)
  P <- rbind(P,pN)
  if (any(sqrt((pX - oX)^2 + (pY - oY)^2) < 0.60)) {
    break
  }
  if (sqrt((pX - gX)^2 + (pY - gY)^2) < 0.60 | sqrt((pX - gX)^2 + (pY - gY)^2) > 50) {
    break
  }
  x <- x + 1
  if (x >= 1000) {
    break
  }
}

a <- ggplot(O,aes(x,y))+
  geom_text(aes(label='W'),color="darkred",size=3) +
  geom_point(data=P,aes(x,y,colour = factor(s)),size=3) +
  scale_colour_manual(values = c("red", "orange","blue")) +
  geom_text(aes(x=gX,y=gY,label='G'),color = "green",size=5) +
  scale_x_continuous(limits = c(-50,50)) +
  scale_y_continuous(limits = c(-50,50)) +
  labs(title = 'Time Step: {closest_state}', colour = 'State') +
  transition_states(t) +
  ease_aes('linear')
print(a)