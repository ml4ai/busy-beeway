source("~/busy-beeway/planners/lmdp/utility.R")
source("~/busy-beeway/planners/lmdp/tid_state_space.R")
source("~/busy-beeway/planners/lmdp/value_functions.R")

move_targets <- function(O,P,grid_length) {
  new_O <- NULL
  for (o in 1:nrow(O)) {
    if (O[o,3]) {
      o_x <- O[o,1]
      o_y <- O[o,2]
      for (i in 1:2) {
        can_stay <- !(P$x == o_x & P$y == o_y)
        #1
        can_up <- !(P$x == o_x & P$y == (o_y + 1)) & ((o_y + 1) <= grid_length)
        #2
        can_down <- !(P$x == o_x & P$y == (o_y - 1)) & ((o_y - 1) >= 1)
        #3
        can_left <- !(P$x == (o_x - 1) & P$y == o_y) & ((o_x - 1) >= 1)
        #4
        can_right <- !(P$x == (o_x + 1) & P$y == o_y) & ((o_x + 1) <= grid_length)
        if (can_stay) {
          if (!rbinom(1,1,0.5)) {
            pos_moves <- which(c(can_up,can_down,can_left,can_right))
            if (length(pos_moves) > 0) {
              new_move <- switch(sample(pos_moves,1),
                                 c(o_x,o_y + 1),
                                 c(o_x,o_y - 1),
                                 c(o_x - 1,o_y),
                                 c(o_x + 1,o_y))
              o_x <- new_move[1]
              o_y <- new_move[2]
            }
          }
        }
        else {
          pos_moves <- which(c(can_up,can_down,can_left,can_right))
          if (length(pos_moves) > 0) {
            new_move <- switch(sample(pos_moves,1),
                               c(o_x,o_y + 1),
                               c(o_x,o_y - 1),
                               c(o_x - 1,o_y),
                               c(o_x + 1,o_y))
            o_x <- new_move[1]
            o_y <- new_move[2]
          }
        }
      }
      new_O <- rbind(new_O,data.frame(x=o_x,y=o_y,color=O[o,3],t=O[o,4] + 1))
    }
    else {
      can_stay <- !(P$x == O[o,1] & P$y == O[o,2])
      #1
      can_up <- !(P$x == O[o,1] & P$y == (O[o,2] + 1)) & ((O[o,2] + 1) <= grid_length)
      #2
      can_down <- !(P$x == O[o,1] & P$y == (O[o,2] - 1)) & ((O[o,2] - 1) >= 1)
      #3
      can_left <- !(P$x == (O[o,1] - 1) & P$y == O[o,2]) & ((O[o,1] - 1) >= 1)
      #4
      can_right <- !(P$x == (O[o,1] + 1) & P$y == O[o,2]) & ((O[o,1] + 1) <= grid_length)
      if (can_stay) {
        if (rbinom(1,1,0.5)) {
          new_O <- rbind(new_O,data.frame(x=O[o,1],y=O[o,2],color=O[o,3],t=O[o,4]+1))
        }
        else {
          pos_moves <- which(c(can_up,can_down,can_left,can_right))
          if (length(pos_moves) == 0) {
            new_O <- rbind(new_O,data.frame(x=O[o,1],y=O[o,2],color=O[o,3],t=O[o,4]+1))
          }
          else {
            new_move <- switch(sample(pos_moves,1),
                               c(O[o,1],O[o,2] + 1),
                               c(O[o,1],O[o,2] - 1),
                               c(O[o,1] - 1,O[o,2]),
                               c(O[o,1] + 1,O[o,2]))
            new_O <- rbind(new_O,data.frame(x=new_move[1],y=new_move[2],color=O[o,3],t=O[o,4]+1))
          }
        }
      }
      else {
        pos_moves <- which(c(can_up,can_down,can_left,can_right))
        if (length(pos_moves) == 0) {
          new_O <- rbind(new_O,data.frame(x=O[o,1],y=O[o,2],color=O[o,3],t=O[o,4]+1))
        }
        else {
          new_move <- switch(sample(pos_moves,1),
                             c(O[o,1],O[o,2] + 1),
                             c(O[o,1],O[o,2] - 1),
                             c(O[o,1] - 1,O[o,2]),
                             c(O[o,1] + 1,O[o,2]))
          new_O <- rbind(new_O,data.frame(x=new_move[1],y=new_move[2],color=O[o,3],t=O[o,4]+1))
        }
      }
    }
  }
  new_O
}

sim_session <- function(b0,
                        b1,
                        b2,
                        t_dat,
                        delT=2,
                        time_limit=30,
                        grid_length=10,
                        n_green=10,
                        n_yellow=5) {
  grid_seq <- 1:grid_length
  colors <- c(rep(0,n_green),rep(1,n_yellow))
  vf1 <- create_vf_tid(b0,b1,b2)
  grid <- expand.grid(x=grid_seq, y=grid_seq)
  grid_dist <- as.matrix(dist(matrix(c(grid$x,grid$y),nrow(grid),2),
                              method="manhattan",
                              diag=TRUE,
                              upper=TRUE))
  # green = 0, yellow = 1
  O <- data.frame(x = sample(grid_seq,length(colors),replace=TRUE),
                  y = sample(grid_seq,length(colors),replace=TRUE), 
                  color = colors,
                  t=rep(0,length(colors)))
  repeat {
    px <- sample(grid_seq,1)
    py <- sample(grid_seq,1)
    if (!any(O$x == px & O$y == py)) {
      break
    }
  }
  
  P <- data.frame(x=px,y=py,t=0)
  for (t in 1:time_limit) {
    new_O <- move_targets(O[which(O$t == (t-1)),],P[t,],grid_length)
    temp_O <- rbind(O,new_O)
    new_states <- create_state_space(c(P[t,1],P[t,2]),temp_O,grid,delT,t_dat,grid_dist,t)
    v <- vf1(new_states)
    z <- exp(-v)
    max_z <- sample(which(equals_plus(z,max(z))),1)
    new_P <- data.frame(x=new_states[max_z,1],y=new_states[max_z,2],t=t)
    P <- rbind(P,new_P)
    caught <- which(equals_plus(new_O$x,new_P[1,1]) & equals_plus(new_O$y,new_P[1,2]))
    if (length(caught) > 0) {
      caught <- sample(caught,1)
      O <- rbind(O,new_O[-caught,])  
    }
    else {
      O <- temp_O
    }
  }
  green_left <- nrow(O[which(O$t == time_limit & O$color == 0),])
  yellow_left <- nrow(O[which(O$t == time_limit & O$color == 1),])
  score <- 5*(n_green - green_left) + 10*(n_yellow - yellow_left)
  list(P,O,green_left,yellow_left,score)
}

run_sim <- function(b0,
                    b1,
                    b2,
                    sessions=100,
                    t_samps=1000,
                    delT=2,
                    time_limit=30,
                    grid_length=10,
                    n_green=10,
                    n_yellow=5) {
  grid <- expand.grid(x=1:grid_length, y=1:grid_length)
  t_dat <- create_t_dat(delT,grid,t_samps)
  dat <- list()
  for (s in 1:sessions) {
    dat <- rbind(dat,list(sim_session(b0,
                                      b1,
                                      b2,
                                      t_dat,
                                      delT,
                                      time_limit,
                                      grid_length,
                                      n_green,
                                      n_yellow)))
  }
  dat
}