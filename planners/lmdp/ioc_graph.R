library("igraph")
source("~/busy-beeway/planners/lmdp/utility.R")


create_state_graph <- function(size,sample_size=29) {
  n <- sample_size + 1
  g <- make_star(n,mode="out")
  for (i in 1:size) {
    g <- g + make_star(n,mode="out")
    g <- add_edges(g,c(n*(i - 1) + 1,n*(i) + 1))
  }
  g <- add_vertices(g,1)
  g <- add_edges(g,c(n*size + 1,n*(size + 1) + 1))
  g
}

create_passive_dynamics <- function(size,sample_size=29) {
  g <- create_state_graph(size,sample_size)
  m <- as_adjacency_matrix(g,sparse=FALSE)
  m <- m/rowSums(m)
  diag(m)[which(is.nan(diag(m)))] <- 1.0
  m[which(is.nan(m))] <- 0.0
  m
}

create_state_space <- function(p_df,gx,gy,O,obs_st,size,sample_size=29,tol=0.3) {
  g <- create_state_graph(size,sample_size)
  n <- sample_size + 1
  t <- 0
  c_radius <- point_dist(p_df[1,1],p_df[1,2],p_df[2,1],p_df[2,2])
  for (i in 1:length(V(g))) {
    if (i == (n*(t) + 1)) {
      V(g)$x[i] <- p_df[t+1,1]
      V(g)$y[i] <- p_df[t+1,2]
    }
    else {
      samp <- runif_circle(1,c_radius,center=c(p_df[t+1,1],p_df[t+1,2]))
      V(g)$x[i] <- samp[1]
      V(g)$y[i] <- samp[2]
    }
    
    
  }
}