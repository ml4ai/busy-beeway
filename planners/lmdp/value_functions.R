create_vf_bb <- function(b1,b2) {
  valfunc <- function(states) {
    val <- b1*states$rd_goal + b2*states$threat_level
    val
  }
  valfunc
}

#c ensures positivity
create_vf_mct <- function(b1,b2,c) {
  valfunc <- function(states) {
    val <- c - (b1*states$expected_gain - b2*states$expected_loss)
    val
  }
  valfunc
}