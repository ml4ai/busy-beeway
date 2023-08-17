source("fsm.r")
#these functions should take in state of the game 
#and output an action following some controller 
#logic


#try nesting/calling functions in lists to organize it. 
progress_transitions <- function(input) {
  th1 <- threatened(input$px,input$py,input$ox,input$oy,input$gx,input$gy,input$d1,input$a1)
  th2 <- threatened(input$px,input$py,input$ox,input$oy,input$gx,input$gy,input$d2,input$a2)
  if (th2) {
    newState <- "Backup"
  }
  else if (th1) {
    newState <- "Dodge"
  }
  else {
    newState <- "Progress"
  }
  newState
}

dodge_transitions <- function(input) {
  th1 <- threatened(input$px,input$py,input$ox,input$oy,input$gx,input$gy,input$d1,input$a1)
  th2 <- threatened(input$px,input$py,input$ox,input$oy,input$gx,input$gy,input$d2,input$a2)
  if (th2) {
    newState <- "Backup"
  }
  else if (th1) {
    newState <- "Dodge"
  }
  else {
    newState <- "Progress"
  }
  newState
}

backup_transitions <- function(input) {
  th1 <- threatened(input$px,input$py,input$ox,input$oy,input$gx,input$gy,input$d1,input$a1)
  th2 <- threatened(input$px,input$py,input$ox,input$oy,input$gx,input$gy,input$d2,input$a2)
  if (th2) {
    newState <- "Backup"
  }
  else if (th1) {
    newState <- "Dodge"
  }
  else {
    newState <- "Progress"
  }
  newState
}

fsm <- list(handlers = c(), currentState = NULL)
class(fsm) <- "FSM"

fsm <- add_state(fsm,"Progress",progress_transitions)
fsm <- add_state(fsm,"Dodge",dodge_transitions)
fsm <- add_state(fsm,"Backup",backup_transitions)
fsm <- set_start(fsm,"Progress")

#all args are scalars
find_direction <- function(x1,y1,x2,y2) {
  y <- y2 - y1
  x <- x2 - x1
  if (y == 0 & x > 0) {
    h <- 360
  }
  else if (y >= 0) {
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

#The first two args are scalars, the second two are vectors. 
find_directions <- function(pX,pY,oX,oY) {
  Y <- oY - pY
  X <- oX - pX
  H <- rep(0,length(Y))
  H[which(Y == 0 & X > 0)] <- 360
  a <- which((Y > 0 & X >= 0) | (Y >= 0 & X < 0))
  H[a] <- atan2(Y[a],X[a])*180/pi
  b <- which(Y < 0)
  H[b] <- (atan2(Y[b],X[b]) + 2*pi)*180/pi
  H[which(Y == 0 & X == 0)] <- NaN
  H
}

threatened <- function(pX,pY,oX,oY,gX,gY,max_d,max_ha) {
  threat <- F
  odX <- oX[which(sqrt((pX - oX)^2 + (pY - oY)^2) <= max_d)]
  odY <- oY[which(sqrt((pX - oX)^2 + (pY - oY)^2) <= max_d)]
  if (length(odX) != 0) {
    pH <- find_direction(pX,pY,gX,gY)
    upH <- pH + max_ha
    lpH <- pH - max_ha
    if (upH > 360) {
      upH <- upH - 360 
    }
    if (lpH <= 0) {
      lpH <- 360 - lpH
    }
    if (lpH > upH) {
      oaX <- odX[which(find_directions(pX,pY,odX,odY) >= lpH | find_directions(pX,pY,odX,odY) <= upH)]
      oaY <- odY[which(find_directions(pX,pY,odX,odY) >= lpH | find_directions(pX,pY,odX,odY) <= upH)]
    }
    else {
      oaX <- odX[which(find_directions(pX,pY,odX,odY) >= lpH & find_directions(pX,pY,odX,odY) <= upH)]
      oaY <- odY[which(find_directions(pX,pY,odX,odY) >= lpH & find_directions(pX,pY,odX,odY) <= upH)]
    }
    if (length(oaX) != 0) {
      threat <- T
    }
  }
  threat
}

ob_moves <- function(oX,oY,oH,speed = 4/30) {
  oX <- oX + (speed*cos(oH*(pi/180)))
  oY <- oY + (speed*sin(oH*(pi/180)))
  oX[which(sqrt(oX^2 + oY^2) > 50)] <- -oX[which(sqrt(oX^2 + oY^2) > 50)]
  oY[which(sqrt(oX^2 + oY^2) > 50)] <- -oY[which(sqrt(oX^2 + oY^2) > 50)]
  list(oX,oY)
}