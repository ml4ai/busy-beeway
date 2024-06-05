move_target <- function(ox,oy,oc,grid_length) {
  if (oc) {
    o_x <- ox
    o_y <- oy
    for (i in 1:2) {
      can_up <- (o_y + 1) <= grid_length
      #2
      can_down <- (o_y - 1) >= 1
      #3
      can_left <- (o_x - 1) >= 1
      #4
      can_right <- (o_x + 1) <= grid_length

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
    ox <- o_x
    oy <- o_y
  }
  else {
    #1
    can_up <- (oy + 1) <= grid_length
    #2
    can_down <- (oy - 1) >= 1
    #3
    can_left <- (ox - 1) >= 1
    #4
    can_right <- (ox + 1) <= grid_length

    if (!rbinom(1,1,0.5)) {
      pos_moves <- which(c(can_up,can_down,can_left,can_right))
      if (length(pos_moves) > 0) {
        new_move <- switch(sample(pos_moves,1),
                           c(ox,oy + 1),
                           c(ox,oy - 1),
                           c(ox - 1,oy),
                           c(ox + 1,oy))
        ox <- new_move[1]
        oy <- new_move[2]
      }
    }
  }
  c(ox,oy)
}

create_t_dat <- function(delT,grid,samps) {
  g_mat <- matrix(0,nrow(grid),nrow(grid))
  y_mat <- g_mat
  grid_length <- max(grid$x)
  for (i in 1:nrow(grid)) {
    for (n in 1:samps) {
      g_move <- c(grid[i,1],grid[i,2])
      y_move <- g_move
      for (t in 0:delT) {
        g_move <- move_target(g_move[1],g_move[2],0,grid_length)
        y_move <- move_target(y_move[1],y_move[2],1,grid_length)
      }
      
      g_match <- which(equals_plus(grid$x,g_move[1]) & equals_plus(grid$y,g_move[2]))
      y_match <- which(equals_plus(grid$x,y_move[1]) & equals_plus(grid$y,y_move[2]))
      
      g_mat[i,g_match] <- g_mat[i,g_match] + 1
      y_mat[i,y_match] <- y_mat[i,y_match] + 1
    }
  }
  g_mat <- g_mat/rowSums(g_mat)
  y_mat <- y_mat/rowSums(y_mat)
  list(g_mat,y_mat)
}

create_state_space <- function(p,O,grid,delT,t_dat,grid_dist,t) {
  o_t <- (t-1) - delT
  O_t <- O[which(O$t == o_t),]
  grid_length <- max(grid$x)
  p_matches <- c()
  if ((p[2] + 1) <= grid_length) {
    p_matches <- c(p_matches,which(equals_plus(grid$x,p[1]) & equals_plus(grid$y,p[2] + 1)))
  }
  #2
  if ((p[2] - 1) >= 1) {
    p_matches <- c(p_matches,which(equals_plus(grid$x,p[1]) & equals_plus(grid$y,p[2] - 1)))
  }
  #3
  if ((p[1] - 1) >= 1) {
    p_matches <- c(p_matches,which(equals_plus(grid$x,p[1] - 1) & equals_plus(grid$y,p[2])))
  }
  #4
  if ((p[1] + 1) <= grid_length) {
    p_matches <- c(p_matches,which(equals_plus(grid$x,p[1] + 1) & equals_plus(grid$y,p[2])))
  }
  new_states <- NULL
  if (nrow(O_t) > 0) {
    for (n_p in p_matches) {
      min_g_edist <- Inf
      min_y_edist <- Inf
      for (o in 1:nrow(O_t)) {
        o_match <- which(equals_plus(grid$x,O_t[o,1]) & equals_plus(grid$y,O_t[o,2]))
        if (O_t[o,3]) {
          y_edist <- sum(grid_dist[n_p,]*t_dat[[2]][o_match,])
          if (y_edist < min_y_edist) {
            min_y_edist <- y_edist
          }
        }
        else {
          g_edist <- sum(grid_dist[n_p,]*t_dat[[1]][o_match,])
          if (g_edist < min_g_edist) {
            min_g_edist <- g_edist
          }
        }
      }
      new_states <- rbind(new_states,data.frame(x=grid[n_p,1],
                                                y=grid[n_p,2],
                                                min_g_edist=min_g_edist,
                                                min_y_edist=min_y_edist))
    }
  }
  else {
    new_states <- rbind(new_states,data.frame(x=grid[p_matches,1],
                                              y=grid[p_matches,2],
                                              min_g_edist=rep(0,length(p_matches)),
                                              min_y_edist=rep(0,length(p_matches))))
  }
  new_states
}

create_state_space_data <- function(P,O,grid,delT,t_dat,grid_dist) {
  trans <- NULL
  off_trans <- list()
  for (t in 1:(nrow(P)-1)) {
    b_states <- create_state_space(c(P[t,1],P[t,2]),O,grid,delT,t_dat,grid_dist,t)
    p_x <- P[t+1,1]
    p_y <- P[t+1,2]
    p_idx <- which(equals_plus(b_states$x,p_x) & equals_plus(b_states$y,p_y))
    p_g <- b_states[p_idx,3]
    p_y <- b_states[p_idx,4]
    
    trans <- rbind(trans,data.frame(min_g_edist=p_g,min_y_edist=p_y))
    
    off_trans <- rbind(off_trans,list(data.frame(min_g_edist=b_states$min_g_edist,
                                                 min_y_edist=b_states$min_y_edist)))
  }
  list(trans,off_trans)
}