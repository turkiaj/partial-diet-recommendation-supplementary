
data { 
  int<lower=0> N;   // number of observations
  int<lower=0> NH;  // number of holdout observations
  int<lower=1> p;   // number of predictors
  int<lower=1> v;   // number of responses
  int<lower=1> J;   // number of groups in data (persons)
  int<lower=1> k;   // number of group-level predictors
  int<lower=1> vk;  // v*k
  int<lower=1,upper=J> group[N]; //group indicator
  matrix[N,p] X;    // fixed-effect design matrix
  matrix[N,k] Z;    // random-effect design matrix
  vector[N] Y[v];   // responses
  real offset;

  // indicates if observation is used for training (0) or prediction (1)
  int<lower=0,upper=1> holdout[N]; 
} 

transformed data { 
  
  matrix[v*(N-NH),v*(p-1)] XM_t; // common training input for all responses
  matrix[v*(N-NH),v*k] ZM_t;     // personal training input for all responses
  vector[v*(N-NH)] Y_t;          // stacked training responses from all responses
  int t=1;                       // index
  matrix[v*(N-NH),v] I;          // identity matrix for intercept
  vector[v] linear_transformation;  // transforms all data to positive region
  
  // QR reparameteratization of the model matrix
  // - reparamization matrices are stacked like X_t
  
  matrix[v*(N-NH), v*(p-1)] Q_ast;
  matrix[v*(p-1), v*(p-1)] R_ast;
  matrix[v*(p-1), v*(p-1)] R_ast_inverse;

  I = rep_matrix(0,v*(N-NH),v);
  linear_transformation =  I * rep_vector(offset, v);
  
  // X_t and Z_t are stacked vectors from all responses
  // - fill X_t and Z_t initially with 0
  XM_t = rep_matrix(0,v*(N-NH),v*(p-1));
  ZM_t = rep_matrix(0,v*(N-NH),v*k);
  
  for (m in 1:v)
  {
    for (n in 1:N)
    {
      if (holdout[n] == 0)
      {
        // two loops create block-diagonal 1-matrix (N-NH x v)
        I[t,v] = 1;

        // XM_t is block diagonal regarding to response inputs
        // - the intercept is removed from the model matrix 
        XM_t[t,(m-1)*(p-1)+1:m*(p-1)] = X[n,2:p];

        ZM_t[t,(m-1)*k+1:m*k] = Z[n,1:k];
        
        // linear transformation is applied for true responses also
        // this is transformed back in final results
        Y_t[t] = Y[m,n] + offset;

        t += 1;
      }
    }
  }
  
  //ZM_w = csr_extract_w(ZM_t);
  //ZM_v = csr_extract_v(ZM_t);
  //ZM_u = csr_extract_u(ZM_t);
    
  // thin and scale the QR decomposition
  Q_ast = qr_thin_Q(XM_t) * sqrt(v*(N-NH) - 1);
  R_ast = qr_thin_R(XM_t) / sqrt(v*(N-NH) - 1);
  R_ast_inverse = inverse(R_ast);  
}

parameters { 
  // all parameters are vectorized based on response, 1..v
  // except personal effects are stacked in same matrices for estimating cross-covariance
  
  vector[v] beta_Intercept;         // intercept 
  //vector[v*(p-1)] beta;           // poulation-level effects (fixed effects)
  vector[v*(p-1)] theta_q;          // coefficients of Q_ast
  cholesky_factor_corr[v*k] L;      // Cholesky factor of group ranef corr matrix
  vector<lower=0>[v*k] sigma_b;     // group-level random-effect standard deviations
  vector[v*k] z[J];                 // unscaled group-level effects
  vector[v] g_log_alpha;            // alpha (shape) parameter of the gamma distribution
}

transformed parameters {
  vector[v] g_alpha;                // alpha (shape) parameter of each v gamma distribution
  //real<lower=0> sigma_e[v];         // residual standard deviations for each distribution v 
  vector[v*k] b[J];                 // personal effects for each patient for all predictors at all responses
  
  // Premultiply diagonal matrix [sigma_b] with the Cholesky decomposition L of
  // the correlation matrix Sigma_b to get variance-covariance matrix of group-level effects

  // local scope for Lambda matrix
  {
    // diag(sigma_b) * L
    matrix[v*k, v*k] Lambda;       // Tau * Cholesky decomposition
    
    Lambda = diag_pre_multiply(sigma_b, L); 

    for(j in 1:J) 
      b[j] =  Lambda * z[j];
   }
  
  // - log transform alpha parameter to keep it positive
  g_alpha = exp(g_log_alpha);

  // estimate of variance 
  // (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4024993/)
  //sigma_e = log(1 ./ g_alpha + 1);
}

model {
  
    beta_Intercept ~ cauchy(0,10); // prior for the intercept following Gelman 2008
    
    sigma_b ~ student_t(3, 0, 10);
    
    for (j in 1:J) {
      z[j] ~ normal(0,1);
    }
    
    // - LJK prior for personal effect Cholesky factor
    L ~ lkj_corr_cholesky(1);

  // brackets introduce a new scope for local variables that are not published in model
  {
    vector[v*(N-NH)] mu;             // expected value 
    vector[v*(N-NH)] g_beta;         // beta (rate) of Gamma distribution
    int vn = 1;
    
    mu = I * beta_Intercept + linear_transformation + Q_ast * theta_q;
    
    for (m in 1:v)
    {
      for (n in 1:N-NH)
      {
        // - row n is picked from ZM_t and multiplied with column of coefs from a patient
         mu[vn] = mu[vn] + ZM_t[vn] * b[group[n]];
         vn += 1;
      }
    }

     // identity link
     g_beta = I * g_alpha ./ mu;

     Y_t ~ gamma(I * g_alpha, g_beta);

      //target += gamma_lpdf(Y_t[i] | g_alpha[i], g_beta);
  }

}

generated quantities {

  vector[v*(p-1)] beta;               // poulation-level effects (fixed effects)
  corr_matrix[v*k] C;                 // correlation matrix 
  vector[N-NH] Y_rep[v];              // repeated response
  //real Y_rep[v,N-NH];                 // repeated response

  // //vector[N-NH] log_lik[v];       // log-likelihood for LOO
  // vector[k-1] personal_effect[J,v];
  // real personal_intercept[J,v];
  // int b_index;
  // 
  // Correlation matrix of random-effects, C = L'L
  C = multiply_lower_tri_self_transpose(L);
  // 
  // // Posterior predictive distribution for model checking

  // Unstack Y_rep to separate columns
  {
    real Y_rep_stacked[v*(N-NH)];       // stacked training responses from all responses
    vector[v*(N-NH)] mu_hat;            // expected value 
    vector[v*(N-NH)] g_beta_hat;        // beta (rate) of Gamma distribution

    beta = R_ast_inverse * theta_q;     // coefficients on x

    mu_hat = I * beta_Intercept + linear_transformation + XM_t * beta + ZM_t * b;
  //  mu_hat = I * beta_Intercept + I * linear_transformation + XM_t * beta;
       
    // identity link
    g_beta_hat = I * g_alpha ./ mu_hat;
  
    Y_rep_stacked = gamma_rng(I * g_alpha, g_beta_hat);
    
    // transform repeated values back to original intercept
    for (i in 1:v)
    {
      for (n in 1:N-NH)
      {
          Y_rep[i,n] = Y_rep_stacked[(i-1)*(N-NH)+n] - offset;
      }
    }
  }

  // for (i in 1:v)
  // {
  //   b_index = (i-1)*k;
  //   beta[i] = R_ast_inverse[i] * theta_q[i];     // coefficients on x
  // 
  //   for (n in 1:N-NH) 
  //   {
  //     mu_hat = beta_Intercept[i] + offset + X_t[n] * beta[i] + Z_t[n] * b[b_index+group[n]];
  // 
  //     g_beta_hat = g_alpha[i] / mu_hat;
  // 
  //     Y_rep[i,n] = gamma_rng(g_alpha[i], g_beta_hat) - offset;
  // 
  //     // Compute log-Likelihood for LOO comparison of the models 
  //     //log_lik[i,n] = gamma_lpdf(Y_t[i,n] | g_alpha[i], g_beta_hat);
  //   }
  // 
  //   // Finally, sample personal effects for each nutrient
  //   for (j in 1:J) 
  //   {
  //     
  //     // personal intercept
  //     personal_intercept[j,i] = beta_Intercept[i] + b[j][b_index+1];
  // 
  //     // beta vector does not include intercept, b is also sliced not to include it
  //     personal_effect[j,i] = beta[i] + b[j][b_index+2:b_index+k];
  //   }
  // }

}


