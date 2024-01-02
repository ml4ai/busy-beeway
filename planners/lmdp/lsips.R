source("~/busy-beeway/planners/lmdp/lmdp.R")

ob_divider <- function(obs,chunk_length) {
  chunks <- list()
  even_chunks <- (nrow(obs) %/% chunk_length)
  j <- 1
  for (i in 1:even_chunks) {
    chunks[[length(chunks)+1]] <- j:(i*chunk_length + 1)
    j <- (i*chunk_length + 1)
  }
  if (!equals_plus(nrow(obs)/chunk_length,even_chunks)) {
    chunks[[length(chunks)+1]] <- j:nrow(obs)
  }
  chunks
}

recover_trajectory <- function(obs,board) {
  c_idx <- which(greater_equals_plus(obs[1,1],(board$cols - 1/2)) & 
                   lesser_equals_plus(obs[1,1],(board$cols + 1/2)) &
                   greater_equals_plus(obs[1,2],(board$rows - 1/2)) & 
                   lesser_equals_plus(obs[1,2],(board$rows + 1/2)))[1]
  tr_idx <- c(c_idx)
  for (i in 2:nrow(obs)) {
    idx <- which(greater_equals_plus(obs[i,1],(board$cols - 1/2)) & 
                     lesser_equals_plus(obs[i,1],(board$cols + 1/2)) &
                     greater_equals_plus(obs[i,2],(board$rows - 1/2)) & 
                     lesser_equals_plus(obs[i,2],(board$rows + 1/2)))[1]
    if (length(idx) > 1) {
      next
    }
    if (idx == c_idx) {
      next
    }
    
    c_idx <- idx
    tr_idx <- c(tr_idx,c_idx)
  }
  tr_idx
}

lsips <- function(D,obs_st,omin,omax,pspeed,pt,k=100,chunk_length=30) {
  B1 <- runif(k)
  B2 <- 1 - B1
  W <- rep(1,k)
  for (d in 1:nrow(D)) {
    print(sprintf("Processing observation set %i out of %i",d,nrow(D)))
    P <- D[[d,1]]
    O <- D[[d,2]]
    g <- D[[d,4]]
    c_idx <- ob_divider(P,chunk_length)
    for (t in 1:length(c_idx)) {
      print(sprintf("Processing observation %i out of %i from set %i",t,length(c_idx),d))
      P_c <- P[c_idx[[t]],]
      O_c <- O[which(O$t == P_c[1,3]),]
      board <- create_board(P_c[1,1],P_c[1,2],g[1],g[2],O_c,obs_st,omin,omax,pspeed,pt)
      tr_idx <- recover_trajectory(P_c,board)
      g_id <- which(greater_equals_plus(g[1],(board$cols - 1/2)) & 
                      lesser_equals_plus(g[1],(board$cols + 1/2)) &
                      greater_equals_plus(g[2],(board$rows - 1/2)) & 
                      lesser_equals_plus(g[2],(board$rows + 1/2)))
      if (length(g_id) != 0) {
        dp <- create_uniform_default_policy(pspeed,pt,g_id[1]) 
      }
      else {
        dp <- create_uniform_default_policy(pspeed,pt)
      }
      for (i in 1:k) {
        cf <- create_cf(B1[i],B2[i])
        res <- LRL(dp,cf,board,g,pspeed*pt)
        W[i] <- W[i]*tr_likelihood(tr_idx,res[[1]])
      }
    }
  }
  W <- W/sum(W)
  data.frame(w=W,b1=B1,b2=B2)
}

get_max_W <- function(res) {
  res[which(equals_plus(res$w,max(res$w))),]
}

plot_dist <- function(res) {
  ggplot() + 
    geom_point(res,mapping=aes(x=b1,y=b2,color=w)) + 
    scale_color_gradient(low="blue",high="orange")
}
