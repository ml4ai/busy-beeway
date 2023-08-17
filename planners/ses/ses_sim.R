source("SES.R")

pX <- 25
pY <- 0
gX <- 0
gY <- 0
oX <- runif(50,-20,20)
oY <- runif(50,-20,20)
oH <- runif(50,1,360)
pspeed <- 8
ospeeds <- c(4.0,8.0,12.0,16.0)
oprobs <- c(0.25,0.34,0.25,0.15)
ptr <- 0.2
Praccept <- 0.01
max_time <- 500
K <- 1500
gb <- 0.01
greed <- 0.001
safety <- 10
eps <- 0.6
tol <- 0.3

if (!exists("XE")) {
  XE <- SES(8,0.01,5000,ptr,ospeeds,oprobs)
}

if (!exists("GP")) {
  GP <- createGlobalPath(pX,pY,gX,gY)
}

#start <- Sys.time()
D <- SES_planning(XE,
                  GP,
                  pX,
                  pY,
                  gX,
                  gY,
                  oX,
                  oY,
                  oH,
                  pspeed,
                  ospeeds,
                  oprobs,
                  ptr,
                  Praccept=0.01,
                  max_time=500,
                  K=1500,
                  gb=0.01,
                  greed=0.001,
                  safety=10,
                  eps=0.6,
                  tol=0.3)
#print(Sys.time() - start)
P <- D[[1]]
O <- D[[2]]
a <- ggplot(O,aes(x,y))+
  geom_text(aes(label='W'),color="darkred",size=3) +
  geom_point(data=P,aes(x,y),size=3) +
  geom_text(aes(x=gX,y=gY,label='G'),color = "green",size=5) +
  scale_x_continuous(limits = c(-50,50)) +
  scale_y_continuous(limits = c(-50,50)) +
  labs(title = 'Time Step: {closest_state}') +
  transition_states(t)
print(a)