source("~/busy-beeway/data/data_processing.R")
source("~/busy-beeway/planners/lmdp/bb_ioc.R")

run_bb_exp <- function(lvl,
                       pth="~/busy-beeway/data/game_data/Experiment_1T5/auto-f08db6011d8be394/test.2023.08.24.08.41.41/",
                       skip=0,
                       n_params=3,
                       sample_size=1000,
                       delT=c(0,10),
                       rho=c(0,10),
                       p_samples=100,
                       guess = NULL,
                       max_iter=10000,
                       tol=0.0001,
                       lambda=1,
                       cores=NULL) {
  O <- load_BB_data(pth,skip)[,2]
  obs_st <- generate_obs_ses(O)
  d_train <- load_lvl_data(lvl,pth,skip)
  newton_loocv(d_train,
               n_params,
               obs_st,
               sample_size,
               delT,
               rho,
               p_samples,
               guess,
               max_iter,
               tol,
               lambda,
               cores)
}