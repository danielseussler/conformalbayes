/*
* Gaussian regression for K predictors
* w/ additional weights for add-one-in importance sampling
*
*/
functions {
  /* ... function declarations and definitions ... */
}
data {
  int<lower=1> N;
  int<lower=1> K;
  matrix[N, K] xmat;
  vector[N] y;
  int<lower=1> P;
  int<lower=1> Q;
  matrix[P, K] xtest;
  vector[P] ytest;
  vector[Q] ygrid;
  int misspec;
}
transformed data {
    real<lower=0> constant = misspec ? 0.02 : 1;
}
parameters {
  real alpha;
  vector[K] beta;
  real<lower=0> sigma;
  real<lower=0> prior_sigma;
}
transformed parameters {
  real lprior = 0;
  lprior += gamma_lpdf(prior_sigma | 1, 1);
  lprior += double_exponential_lpdf(beta | 0, prior_sigma);
  lprior += normal_lpdf(sigma | 0, constant) - 1 * normal_lccdf(0 | 0, constant);
}
model {
  target += normal_id_glm_lpdf(y | xmat, alpha, beta, sigma);
  target += lprior;
}
generated quantities {
  vector[N] log_lik;
  vector[P] log_lik_test;

  vector[N] y_rep;
  vector[P] y_rep_test;

  array[P, Q] real log_lik_grid;

  for (n in 1 : N) {
    log_lik[n] = normal_lpdf(y[n] | alpha + xmat[n,  : ] * beta, sigma);
    y_rep[n] = normal_rng(alpha + xmat[n,  : ] * beta, sigma);
  }

  for (n in 1 : P) {
    log_lik_test[n] = normal_lpdf(ytest[n] | alpha + xtest[n,  : ] * beta, sigma);
    y_rep_test[n] = normal_rng(alpha + xtest[n,  : ] * beta, sigma);

    for (k in 1 : Q) {
      log_lik_grid[n, k] = normal_lpdf(ygrid[k] | alpha
                                                  + xtest[n,  : ] * beta, sigma);
    }
  }
}
