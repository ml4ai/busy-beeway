repel = function(pX,pY,oX,oY,deviation){
  u = sqrt((pX-oX)^2 + (pY-oY)^2) #euclidian distance between agent and obstacles
  # Updated with deviation squared
  p1 = 1/(sqrt(2*pi*deviation*deviation)) #deviation is a radius for the gaussian?
  p2 = -1*(u*u) 
  p3 = 2*deviation*deviation
  p4 = p2/p3
  return(p1*exp(p4))
}

attract = function(pX,pY,gX,gY){
  dX = gX - pX
  dY = gY - pY
  return(sqrt(dX*dX + dY*dY)) 
}

sampleField = function(pX,pY,gX,gY,oX,oY,
                       deviation,repelWeight,attractWeight){
  ra = sum(repel(pX,pY,oX,oY,deviation))
  ga = sum(attract(pX,pY,gX,gY))
  ra = ra*repelWeight
  ga = ga*attractWeight
  return(ra+ga)
}


getForce = function(pX,pY,gX,gY,oX,oY,
                    deviation,repelWeight,attractWeight,epsilon=0.01){
  
  s1 = sampleField(pX,pY,gX,gY,oX,oY,deviation,repelWeight,attractWeight)
  s2 = sampleField(pX+epsilon,pY,gX,gY,oX,oY,deviation,repelWeight,attractWeight)
  s3 = sampleField(pX,pY+epsilon,gX,gY,oX,oY,deviation,repelWeight,attractWeight)
  return(c((s1-s2)/epsilon,(s1-s3)/epsilon))
}



decideAPF = function(pX,pY,gX,gY,oX,oY,
                     deviation,repelWeight,attractWeight){
  
  if(missing(attractWeight)){
    attractWeight = 1 - repelWeight
  }
  u = getForce(pX,pY,gX,gY,oX,oY,
               deviation,repelWeight,attractWeight)
  mag = sqrt(u[1]^2 + u[2]^2)
  if (mag == 0) {
    u[1] = 0
    u[2] = 0
  }
  else {
    u[1] = u[1]/mag
    u[2] = u[2]/mag
  }
  
  cX = pX + (4/30*u[1])
  cY = pY + (4/30*u[2])
  
  return(c(cX,cY))
}



decideIRL = function(pX,pY,gX,gY,oX,oY,
                     deviation,repelWeight,attractWeight){
  
  if(missing(attractWeight)){
    attractWeight = 1 - repelWeight
  }
  
  angles = seq(1,360,1)
  
  pX_ = pX + 4/30*cos(angles*pi/180)
  pY_ = pY + 4/30*sin(angles*pi/180)
  
  out = rep(NA,length(pX_))
  for(iAngle in 1:length(pX_)){
    ra = sum(repel(pX_[iAngle],pY_[iAngle],oX,oY,deviation))
    ga = sum(attract(pX_[iAngle],pY_[iAngle],gX,gY))
    ra = ra*repelWeight
    ga = ga*attractWeight
    out[iAngle] = ra+ga
  }
  
  P = 1/length(angles)
  z = exp(-1*out)
  Gx = sum(P*z)
  piStar = P*z/Gx 
  
  chosenAngle = angles[which.max(z)]
  
  cX = pX + 4/30*cos(chosenAngle*pi/180)
  cY = pY + 4/30*sin(chosenAngle*pi/180)
  
  return(c(cX,cY))
}


#return prob of a move given parameters
#assumes the world stays stationary because there's no esitmate
#of goal or obstacle next step
likelihoodIRL = function(pX,pY,gX,gY,oX,oY,
                         pX1,pY1,
                         deviation,repelWeight,attractWeight){
  
  if(missing(attractWeight)){
    attractWeight = 1 - repelWeight
  }
  
  angles = seq(1,360,1)
  
  pX_ = pX + 4/30*cos(angles*pi/180)
  pY_ = pY + 4/30*sin(angles*pi/180)
  
  out = rep(NA,length(pX_))
  for(iAngle in 1:length(pX_)){
    ra = sum(repel(pX_[iAngle],pY_[iAngle],oX,oY,deviation))
    ga = sum(attract(pX_[iAngle],pY_[iAngle],gX,gY))
    ra = ra*repelWeight
    ga = ga*attractWeight
    out[iAngle] = ra+ga
  }
  
  P = 1/length(angles)
  z = exp(-1*out) #z of all angles
  Gx = sum(P*z)
  piStar = P*z/Gx #piStar for all anlges
  
  #ra1 = sum(repel(pX1,pY1,oX,oY,deviation))
  #ga1 = sum(attract(pX1,pY1,gX,gY))
  #ra1 = ra1*repelWeight
  #ga1 = ga1*attractWeight
  #vChosen = ra1+ga1
  #zChosen = exp(-1*vChosen)
  #piChosen = P*zChosen/Gx #this might be wrong since the norm
  #term G doesn't include zChosen necessarily
  
  piStarChosen = piStar[which.min(sqrt((pX1 - pX_)^2 + (pY1 - pY_)^2))]
  
  return(piStarChosen)
}