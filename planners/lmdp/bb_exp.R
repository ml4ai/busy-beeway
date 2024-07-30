source("~/busy-beeway/data/data_processing.R")
source("~/busy-beeway/planners/lmdp/bb_ioc.R")

run_bb_exp <- function(lvl,
                       pth="~/busy-beeway/data/game_data/Experiment_1T5/auto-f08db6011d8be394/test.2023.08.24.08.41.41/",
                       skip=c(0,10),
                       control=1,
                       n_params=3,
                       sample_size=1000,
                       delT=c(0,10),
                       rho=c(0,10),
                       guess = NULL,
                       max_iter=10000,
                       tol=0.0001,
                       lambda=1,
                       cores=NULL) {
  if (lvl == 0) {
    d_train <- load_BB_data(pth,skip,control)
    res <- newton_loocv(d_train,
                        n_params,
                        sample_size,
                        delT,
                        rho,
                        guess,
                        max_iter,
                        tol,
                        lambda,
                        cores)
  }
  else {
    d_train <- load_lvl_data(lvl,pth,skip,control)
    res <- newton_loocv(d_train,
                        n_params,
                        sample_size,
                        delT,
                        rho,
                        guess,
                        max_iter,
                        tol,
                        lambda,
                        cores)
  }
  res
}