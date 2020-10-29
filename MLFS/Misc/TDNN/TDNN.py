# CLASS TDRNN: TIME DELAY NN WITH REGRESSORS  ================================================================================================================================================

library(R6)
library(nnet)
library(RSNNS)
library(MiscFunctions)
library(psych)

TDNN = R6Class(
  
  classname='TDNN', 
  
  public=list(
  
    # Public Fields ----
    
    nn=NULL,
    mw=0,
    dec=0,
    range=0,
    maxIt=0,
    nHidden=0,
    X=NULL,
    Y=NULL,
    negAllowed=FALSE,
    horizon=0,
    ffrom=0,
    minDecExp=4,
    maxVarRatio=1.3,
    histForVR=6,
    nnType='nnet',
    trace=FALSE,
    normFactor=1,
    normMinFactor=0,
    normMaxFactor=1,
    last=300,
    outThres=0.99,
    
    origTs=NULL,  
    ts=NULL,
    tsNor=NULL,
    
    
    # Constructor ----      
    
    initialize = function(ts, mw, nHidden, dec=-1, minDecExp=4, range=0.5, maxIt=100, normFactor=1, normMinFactor=0, normMaxFactor=0, negAllowed=FALSE, cal=0, maxVarRatio=1.3, histForVR=6, nnType='nnet', trace=FALSE, last=300, outThres=0.99) {
      self$trace=trace
      self$nnType = nnType
      # init
      self$dec = dec
      self$range = range
      self$maxIt = maxIt
      self$nHidden = nHidden
      self$mw = mw
      self$negAllowed = negAllowed
      self$minDecExp = minDecExp
      # var control
      self$maxVarRatio=maxVarRatio
      self$histForVR=histForVR
      self$last = last
      self$outThres = outThres
      
      ts[ts<0]=0
      if(cal != 0 && length(cal) >= length(ts)) { ts=EliminateHolidays(ts, cal); private$calFcst = cal[(length(ts)+1):length(cal)] }
      self$ts = ts
      self$normFactor = normFactor
      self$normMinFactor = normMinFactor
      self$normMaxFactor = normMaxFactor
      
      self$origTs = ts
    },
  
    # Public Methods ----      
    
    Forecast = function(horizon=1, it=1) {
      res = CalcLim(self$ts, self$outThres)
      self$ts = Truncate(self$ts, res$lim)
      self$ts = GetLastTs(self$ts, self$last)
      
      self$tsNor = Nrm$new(self$ts, self$normFactor, self$normMinFactor, self$normMaxFactor)
      if(self$normFactor<=0) { self$tsNor$normFactor=1; self$tsNor$xMin=0; self$tsNor$xMax=1; self$tsNor$xRange=1 }
      self$ts = self$tsNor$Norm(self$ts)
      self$X = private$GetX(self$ts, self$mw)
      self$Y = private$GetY(self$ts, self$mw)
      self$horizon = horizon
      
      bestMae = .Machine$double.xmax
      bestFcst = 0; bestDec=self$dec
      if(self$trace) { cat(paste('i','decay','mae', sep='\t')); cat('\n') } 
      for(i in 0:(it-1)) {
        if(self$dec < 0 && i <= 4*self$minDecExp) dec = GetExponentBy5(self$minDecExp, i) else dec = bestDec
        self$CreateNN(self$nnType, dec)
        fcst = self$FcstFrom(length(self$ts)-self$mw+1, horizon)
        mae = private$CrossValidation(horizon)
        if(self$trace) { cat(paste(i,dec,mae,sep='\t')); cat('\n') }
        ratio = CalcSdRatio(self$ts, self$histForVR, fcst) #print(ratio)
        if(ratio <= self$maxVarRatio && mae < bestMae) { bestMae = mae; bestDec=dec; bestFcst = fcst }
      }
      self$dec=bestDec
      if(self$trace) { print(paste('MAE = ', mae, '%', sep='')) }
      if(self$negAllowed==FALSE) { bestFcst[bestFcst<0] = 0 }
      forecast = self$tsNor$UnNorm(bestFcst)
      if(length(private$calFcst)>= horizon) { forecast = forecast * private$calFcst[1:horizon] }
      forecast
    },
    
    FcstFrom = function(start, horizon) {
      fcst = numeric(horizon)
      x = self$ts[start:(start+self$mw-1)]
      for(i in 1:horizon) {
        y = predict(self$nn, x)
        fcst[i] = y
        x = c(x[2:self$mw], y) 
      }
      fcst
    },
    
    PrintCrossValidation = function() {
      print(paste('Expected Actual'))
      for(i in 1: length(self$ffrom)) {
        exp = round(self$ts[(i+self$mw):(i+self$mw+self$horizon-1)],2)
        print(paste(exp, round(self$ffrom[i],2), round(abs(self$ffrom[i]-exp),2)), quote = FALSE)
        print('')
      }
    },
    
    GetOneStepFcstTs = function() {
      Osf = numeric(0)
      for(i in 1:(length(self$ts)-self$mw)) {
        osf = self$FcstFrom(i, 1);
        osf = self$tsNor$UnNorm(osf)
        Osf = c(Osf, osf)
      }
      c(rep(NA,self$mw), Osf)
    },
    
    CreateNN = function(nnType, dec) {
      switch(nnType, 
        nnet      = { self$nn = nnet(self$X, self$Y, size=self$nHidden, linout=T, trace=F, decay=dec, rang=self$range, maxit=self$maxIt) },
        neuralnet = { self$nn = nnet(formula=Y~X, data=, size=self$nHidden, linout=T, trace=F, decay=dec, rang=self$range, maxit=self$maxIt) },
        nsnns     = { self$nn = mlp(self$X, self$Y, size=self$nHidden, linOut=T) },
                    { print(paste('Error. Neural Network ', nnType, ' does not exist.', sep='')) }
      )
    }
  ),
  

  
  private = list(
    
    # Private Fields ----
    
    calFcst=NULL,
    
    # Private Methods ----
    
    CrossValidation = function(horizon) {
      aec=0
      for(i in 1:(length(self$ts)-self$mw-horizon+1)) {
        out = self$FcstFrom(i, horizon)
        out[out<0] = 0
        exp = self$ts[(i+self$mw):(i+self$mw+horizon-1)]
        aec = aec + sum(abs(out-exp))/horizon
      }
      mae = round((aec/sum(self$ts[1:(length(self$ts)-self$mw-horizon+1)])*100),2)
      mae    
    },
    
    Mae = function(out, exp) {
      sum(abs(out-exp))
    },
    
    GetX = function(ts, mw) {
      X = data.frame(NULL)
      for(i in 1:(length(ts)-mw)) { X = (rbind(X, ts[i:(mw+i-1)])) }
      X
    },
    
    GetY = function(ts, mw) {
      ts[(mw+1):length(ts)]
    },
    
    GetLast = function(ts, n) {
      ts[(length(ts)-n+1):length(ts)]
    },
    
    SetNames = function() {
      xNames = NULL
      for(i in 1:self$mw) { xNames = c(xNames, paste('Tn_', self$mw-i+1, sep='')) }
      names(self$X) = xNames
      colnames(self$Y) = 'Tn'
    }
    
  )
)
