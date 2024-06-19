create_vf_bb <- function(B) {
  valfunc <- function(states) {
    val <- B[1]*states$r_goal_heading + B[2]*states$rd_goal
    val
  }
  valfunc
}

create_vf_mct <- function(b0,b1,b2) {
  valfunc <- function(states) {
    val <- b0 + b1*states$inv_expected_gain + b2*states$expected_loss
    val
  }
  valfunc
}

create_vf_tid <- function(b0,b1,b2) {
  valfunc <- function(states) {
    val <- b0 + b1*states$min_g_edist + b2*states$min_y_edist
    val
  }
  valfunc
}