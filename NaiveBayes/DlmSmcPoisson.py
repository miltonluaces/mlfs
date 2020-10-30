# FUNCTIONS DLM-SMC WITH POISSON DISTRIBUTION
# ==========================================================================================================
    
library(dlm)
library(mvtnorm)


# Poisson DLM-SMC. Estimate Poisson parameter lambda for each time t of a time series using DLM with SMC algorithm
#-----------------------------------------------------------------------------------------------------------------
# Paramets : the time series Data, F & G DLM matrices, m0 & C0 initial distribution, V & W error terms and nW
# Returns  : an array of estimation of lambda for each time t
DlmSmcPoisson  =  function(Data, F, G, m0, C0, V, W, nW) {
    #Declarations
    nD = length(Data)
    nF = length(F)
    A = array(0, dim=c(nF,nW,nD+1))
    InvArav = matrix(0, nD+1,nW)
    m = array(0, dim=c(nF,nW,nD+1))
    C = array(0, dim=c(nF,nF,nW,nD+1))
    w = matrix(0,nD+1,nW)
    theta = array(0, dim=c(nF,nW,nD+1))
    thetaT = array(0, dim=c(nF,nW,nD))
    thetaHat = matrix(0,nF,nD)
    LambdaHat = numeric(0);
  
    #Initial values
    w[1,] = 1/nW
    m[,,1] = m0
    C[,,,1] = C0
    y = c(0, Data)
  
    for(i in 1:nW) {
        v = rmvnorm(1, m[,i,1], C[,,i,1])
        theta[,i,1] = t(v)
    }

    #DLM steps
    for( t in 2:(nD+1)) {
        for(i in 1:nW) {                                       
            #Prior states at time t
            R = G %*% C[,,i,t-1] %*% t(G) + W
            a = G %*% m[,i,t-1]
            A[,i,t] = exp(t(F) %*% a)[1,1] * F
            InvArav[t,i] = solve(t(A[,i,t]) %*% R %*% A[,i,t] + V)
            #i.d.f. mean: m_t = E(t_t|t_t-1,y_t) = G * m_t-1 * R_t * A_t * (A'_t * R_t * A_t + V)^-1 * (y_t - mG)
            m[,i,t] = a + R %*% A[,i,t] %*% InvArav[t,i] %*% (y[t] - exp(t(F) %*% a))       
            #i.d.f. cov: C_t = Var(t_t|t_t-1,y_t) = R_t - R_t * A_t * (A'_t * R_t * A_t + V)^-1 * A'_t * R_t
            C[,,i,t] = R - (R %*% A[,i,t] %*% InvArav[t,i]) %*% t(A[,i,t]) %*% R
            #draw theta_t = g(theta_t | theta_t-1, y_t)
            theta[,i,t]  =  rmvnorm(1, m[,i,t], C[,,i,t])
            #importance density function: multivariate normal for Kalman filter
            impDens = dmvnorm(theta[,i,t], m[,i,t], C[,,i,t])
            #Posterior states at time t
            #t_t = G(t_t-1) + w_t
            probTt = dmvnorm(theta[,i,t], G %*% m[,i,t-1], R)            
            #y_t|t_t = p(y_t, Poisson(F' t_t))
            probYt = dpois(round(y[t]), exp(t(F) %*% theta[,i,t]))             
            #calculate importance weights
            w[t,i] = w[t-1,i] * probYt * probTt / impDens
        }

        #normalize importance weights (sum to 1)
        w[t,] = w[t,]/(sum(w[t,]))

        #algorithm to avoid weight degradation
        nEff = 1/crossprod(w[t,])
        if(nEff < nW/3) {
            index = sample(nW, nW, replace=TRUE, prob = w[t,])
            theta[,,t] = theta[,index,t]
            w[t,] = 1/nW
        }

        for(i in 1:nW) { thetaT[,i,t-1] =  w[t,i] * theta[,i,t] }
        for(f in 1:nF) { thetaHat[f, t-1]  =  sum(thetaT[f,,t-1]) }
  
        # integral : int[ p(y_t| t_t) * p(t_t|D_t-1)]
        lambdaHat = exp(t(F) %*% thetaHat[,t-1]); 
        LambdaHat = c(LambdaHat, lambdaHat); 
    }
    LambdaHat
}


# One-step forecast samples. Generates Poisson forecast samples for each time t given some lambda estimations
#------------------------------------------------------------------------------------------------------------
# Paramets : number of samples nSamples and array of estimations of Lambda (from DlmSmcPoisson) 
# Returns  : a matrix with columns of nSamples for each time t of the time series
PoissonFcstSamples = function(nSamples, LambdaHat) {
  nD = length(LambdaHat);
  yHat = matrix(nrow=nSamples,ncol=nD); 
  for(t in 2:(nD+1)) {
    yh = rpois(nSamples, LambdaHat[t-1]);
    yHat[,t-1] = yh; 
  }
  yHat
}


# One-step forecast mean for each time t
#-----------------------------------------------------------------------------------------------------------
# Paramets : n forecast samples for each time t (from PoissonFcstSamples) 
# Returns  : an array with one-step forecast for each point of the history
FcstMean = function(fcstSamples) {
  fc = NULL
  for(i in 1:length(fcstSamples[1,])) { 
    m = mean(fcstSamples[,i]) 
    fc = c(fc, m)
  }
  fc
}


