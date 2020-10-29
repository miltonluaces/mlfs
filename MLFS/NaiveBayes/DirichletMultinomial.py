# Translate from C#

#region Imports

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Maths;

#endregion

namespace Statistics  {

    /** Method:  Dirichlet-Multinomial Bayesian Model (Polya distribution model) */
    internal class DMModel {

        #region Fields

        private DirichletDistrib dirichlet;
        private Dictionary<string, Option> dirichletByName;
        private Dictionary<int, Option> dirichletByIndex;
        private StatFunctions stat;
        private Combinatory comb;
        private RndGenerator rGen;
        private int maxIt;
        
        #endregion

        #region Constructor

        /** Method:  Constructor */
        internal DMModel(int maxIt) {
            this.dirichlet = new DirichletDistrib();
            this.dirichletByName = new Dictionary<string, Option>();
            this.dirichletByIndex = new Dictionary<int, Option>();
            this.stat = new StatFunctions();
            this.comb = new Combinatory();
            this.rGen = new RndGenerator();
            this.maxIt = maxIt;
        }

        #endregion

        #region Properties

        /** Method:  Discount Factor */
        internal double DiscFactor {
            get { return dirichlet.DiscFactor; }
            set { dirichlet.DiscFactor = value; }
        }

        /** Method:  Standard update quantity */
        internal double UpdatePeriod {
            get { return dirichlet.UpdatePeriod; }
            set { dirichlet.UpdatePeriod = value; }
        }

        /** Method:  Real total quantity entered */
        internal double RealTotalQty {
            get { return dirichlet.RealTotA; }
            set { dirichlet.RealTotA = value; }
        }
        
        #endregion

        #region Setters and Getters

        # region Options

        /** Method:  Add a new option to the model */
        internal void AddOption(string name) {
            int index = dirichletByName.Count;
            Option opt = new Option(index, name, -1);
            dirichletByName.Add(name, opt);
            dirichletByIndex.Add(index, opt);
        }

        internal int GetOptionIndex(string name) {
            if (!dirichletByName.ContainsKey(name)) { throw new ApplicationException("Error. Option " + name + " does not exist"); }
            return dirichletByName[name].index;
        }

        internal void Mask(string name, bool mask) {
            if (!dirichletByName.ContainsKey(name)) { throw new ApplicationException("Error. Option " + name + " does not exist"); }
            dirichletByName[name].mask = mask;
        }
        
        /** Method:  Get a list of option names */
        internal List<string> GetNames() {
            List<string> names = new List<string>();
            foreach (string name in dirichletByName.Keys) { names.Add(name); }
            return names;
        }

        /** Method:  To String, showing parameter names and its proportions */
        public override string ToString() {
            string str = "";
            foreach (Option opt in dirichletByName.Values) { str += opt.name + " : " + dirichlet.Mean(opt.index).ToString("0.00") + "\n"; }
            return str;
        }

        /** Method:  Get a proportion of a particular index */
        internal double GetProportion(int index) {
            return dirichlet.Mean(index);
        }

        /** Method:  Get a proportion of a particular name */
        internal double GetProportion(string name) {
            int i = dirichletByName[name].index;
            return dirichlet.Mean(i);
        }

        /** Method:  Get a list of proportions */
        internal List<double> GetProportions() {
            List<double> proportions = new List<double>();
            foreach (Option opt in dirichletByName.Values) { proportions.Add(dirichlet.Mean(opt.index)); }
            return proportions;
        }

        internal int GetNParams() {
            return dirichletByName.Count;
        }
        
        #endregion

        #region Priors

        /** Method:  Set an uninformative prior for the dirichlet */
        internal void SetUnInformativePrior() {
            for (int i = 0; i < dirichletByName.Values.Count;i++) { dirichlet.AddA(-1); }
            foreach (Option opt in dirichletByName.Values) {
                opt.a = 1.0;
                if (opt.mask) { dirichlet.SetA(opt.index, 0); } 
                else { dirichlet.SetA(opt.index, 1.0); }
            }
        }

        /** Method:  Set an informative prior for the dirichlet */
        internal void SetInformativePrior(List<int> successes) {
            if (successes.Count != dirichlet.GetA().Count) { throw new Exception("Error. Number of proportios must fit dirichlet distribution parameters");  }
            for (int i = 0; i < dirichletByName.Values.Count; i++) { dirichlet.AddA(-1); }
            foreach (Option opt in dirichletByName.Values) {
                opt.a = successes[opt.index];
                if (opt.mask) { dirichlet.SetA(opt.index, 0); } 
                else { dirichlet.SetA(opt.index, successes[opt.index]); }
            }
        }

        #endregion

        #endregion

        #region internal Methods


        /** Method:  Predictive distribution [ G(Sum a_i) / G(Sum a_i + Sum x_i) ] * Prod [ G(a_i + x_i) / G(a_i)] 
        X - X successes
        cdf -  if it is cumulated distribution funcition (if not its pdf)*/
        internal double Probability(IList<int> X, bool cdf) {
            if (cdf) { return ProbabilityCdf(X); } 
            else { return Probability(X); }
        }

        private double Probability(IList<int> X) {
            //double sA = dirichlet.GetTotA();
            double sA = this.RealTotalQty;
            double sX = 0;
            for(int i=0;i<X.Count;i++) { sX += X[i]; }

            double term1 = comb.Factorial((int)sX) * stat.G(sA) / stat.G(sX + sA); 
            
            double term2 = 1;
            for (int i = 0; i < X.Count; i++) {
                double ai = dirichlet.GetA(i);
                term2 = term2 * stat.G(ai + X[i]) / (stat.G(ai) * comb.Factorial(X[i]));
            }
            return term1 * term2;
        }


       private double ProbabilityCdf(IList<int> X) {
            int totX = 0;
            foreach (int x in X) { totX += x; }
            List<int> Xr;
            List<double> Pr = new List<double>();
            for (int it = 0; it < maxIt; it++) {
                Xr = new List<int>();
                for (int i = 0; i < X.Count; i++) {
                    if (dirichletByIndex[i].mask) { Xr.Add(0); }
                    else { Xr.Add(rGen.NextInt(0, X[i])); }
                }
                Pr.Add(Probability(Xr));
            }
            double p = stat.Mean(Pr);
            return p;
        }
        
        /** Method:  Quantile of each option for a particular probability 
       p - probability
       totX - total of all options */
       internal List<int> Quantile(ref double p, ref int totX) {
            
            List<Trial> trials = new List<Trial>();
            double pr;
            List<int> Xr;
            Trial trial;
            int tot = totX;
            int x;
            double totPr = 0;
            for (int it = 0; it < maxIt; it++) {

                Xr = GetTrial(totX);
                pr = Probability(Xr);
                totPr += pr;
                trial = new Trial(Xr, pr);
                trials.Add(trial);
                
            }
            trials.Sort();
            List<int> max = SwappingByMax(trials, ref p, ref totX);
            return max;
        }

        /** Method:  Bayesian update of the dirichlet distribution for time series 
        Xseries - time series for update (all with the same count, fill with zeros otherwise)
        period - update period */
       internal void Update(Dictionary<string, List<int>> XSeries, int period) {
            List<int> serie;
            int firstIndex = 0;
            int lastIndex = period;
            bool end = false;
            while (!end) {
                if (lastIndex >= XSeries.Count - 1) { lastIndex = XSeries.Count - 1; end = true; }
                Dictionary<string, int> XDict = new Dictionary<string, int>();
                double total;
                int totalInt;
                foreach (string optName in XSeries.Keys) {
                    serie = XSeries[optName];
                    total = 0;
                    for (int i = firstIndex; i <= lastIndex; i++) { total += serie[i]; }
                    totalInt = (int)(Math.Round(total));
                    XDict.Add(optName, totalInt);
                }
                Update(XDict, lastIndex-firstIndex);
                firstIndex = lastIndex + 1;
                lastIndex = firstIndex + 1 + period;
            }
        }
        
        /** Method:  Bayesian update of the dirichlet distribution 
        X - values for update 
        period - update period */
       internal void Update(Dictionary<string, int> XDict, int period) {
            //calculate update factor
            double n = period / dirichlet.UpdatePeriod;
            double factor = Math.Pow((1.00 - dirichlet.DiscFactor), n);

            //calculate new real total of A and normaliza A
            double totalQty = 0;
            foreach (int qty in XDict.Values) { totalQty += qty; }
            dirichlet.RealTotA = dirichlet.RealTotA * factor;
            dirichlet.NormalizeA();

            //normalize dictionary of new data           
            double strFactor = (totalQty * dirichlet.MaxTotA) / dirichlet.RealTotA;
            Dictionary<string, int> NormXDict = new Dictionary<string, int>();
            foreach (string optName in XDict.Keys) { NormXDict.Add(optName, (int)(Math.Round(XDict[optName] * strFactor))); }
            
            //update distribution
            Option opt;
            foreach(string optName in NormXDict.Keys) {
                opt = dirichletByName[optName];
                if (opt.mask) { continue; }
                dirichlet.UpdateA(opt.index, XDict[optName]); 
            }
        }
        
        /** Method:  Bayesian update of the dirichlet distribution 
        X - values for update 
        period - update period */
       internal void Update(IList<int> X, int period) {
            Dictionary<string, int> XDict = new Dictionary<string, int>();
            Option opt;
            for (int i = 0; i < X.Count; i++) {
                opt = dirichletByIndex[i];
                if (!opt.mask) { XDict.Add(opt.name, X[i]); }
            }
            Update(XDict, period);
        }

        /** Method:  Mean of the theta_i parameter */
       internal double Mean(int i) {
            return dirichlet.Mean(i);
        }

        /** Method:  Covariance of the theta_i parameter */
       internal double Cov(int i, int j) {
            return dirichlet.Cov(i, j);
        }

        /** Method:  Marginal probability for one option and a particular value of theta 
        i - option index. 
        theta - value of the parameter.
        acum - if it is accumulated probability or not */
       internal double ProbabilityMarg(int i, double theta, bool acum) { 
            return dirichlet.ProbabilityMargBeta(i, theta, acum);
        }

        /** Method:  Marginal quantile for one option and a particular probability
        i - option index. 
        P – probability */
       internal double QuantileMarg(int i, double p) {
            return dirichlet.QuantileMargBeta(i, p);
        }

        #endregion

        #region Private Methods

        #region Quantile methods

        private List<int> GetTrial(int total) {
            int n = GetNParams();
            List<int> X = new List<int>();
            int x;
            for (int i = 0; i < n; i++) {
                if (dirichletByIndex[i].mask) { x = 0; } 
                else { x = rGen.NextInt(0, total); }
                X.Add(x);
                total -= x;
            }
            if (total > 0) {
                int index = rGen.NextInt(0, n - 1);
                X[index] = X[index] + total;
            }
            List<int> perm = comb.Permutation(0,n-1);
            List<int> XPerm = new List<int>();
            foreach (int i in perm) { XPerm.Add(X[i]);  }
            for (int i = 0; i < n; i++) {
                if (dirichletByIndex[i].mask) { XPerm[i] = 0; }
            }
            return XPerm;
        }

        private int GetIndex(List<Trial> trials, ref double p, ref int qty) {
            double totProb = 0;
            for (int i = 0; i < trials.Count;i++) {
                totProb += trials[i].p;
            }

            double prob = 0;

            double quantity = 0;
            for (int i = 0; i < trials.Count; i++) {
                prob += trials[i].p;
                quantity += trials[i].Total;
                double pr = prob / totProb;
                if (p > 0) {
                    if (prob / totProb >= p) { return i; }
                } 
                else {
                    if (quantity >= qty) { return i; }
                }
            }
            return trials.Count - 1;
        }

        private List<int> GetMax(List<Trial> trials, int index) {
            int n = GetNParams();
            List<int> max = new List<int>();
            for (int i = 0; i < n; i++) { max.Add(0); }

            Trial trial;
            for (int i = 0; i <= 
                index; i++) {
                trial = trials[i];
                for (int j = 0; j < n; j++) {
                    if (trial.x[j] > max[j]) { max[j] = trial.x[j]; }    
                }
            }
            return max;
        }

        private void Swap(List<Trial> trials, int i, int j) {
            Trial aux = trials[i];
            trials[i] = trials[j];
            trials[j] = aux;
        }

  
        private int GetTrial(List<Trial> trials, int ini, int end, List<int> max, bool lower) {
            for (int i = ini; i <= end; i++) {
                if (lower) {
                    if (trials[i].IsLowerThan(max)) { return i; }
                } 
                else {
                    if (trials[i].IsHigherThan(max)) { return i; }
                }
                
                
            }
            return -1;
        }

        private bool SwapByMax(List<Trial> trials, List<int> max, int maxIndex) {
            int lowerIndex = GetTrial(trials, maxIndex + 1, trials.Count - 1, max, true);
            if (lowerIndex == -1) { return false; }
            for (int i = maxIndex; i > 0; i--) {
                if (trials[i].IsHigherThan(max)) { 
                    Swap(trials, i, lowerIndex);
                    return true;
                }   
            }
            return false;
        }

        private List<int> SwappingByMax(List<Trial> trials, ref double p, ref int qty) {
            int maxIndex = GetIndex(trials, ref p, ref qty);
            List<int> max = GetMax(trials, maxIndex);
            while (true) {
                bool changed = SwapByMax(trials, max, maxIndex);
                if (!changed) { return max; }
                maxIndex = GetIndex(trials, ref p, ref qty);  
                max = GetMax(trials, maxIndex);
            }
        }

        #endregion

        [Obsolete]
        private void NormalizeProportions(List<double> proportions) {
            double total = 0;
            foreach(double p in proportions) { total += p; }
            if (total != 1) {
                double newTotal = 0;
                for (int i = 0; i < proportions.Count;i++) {
                    proportions[i] = proportions[i] / total;
                    newTotal += proportions[i];
                    if (newTotal > 1.0) { proportions[i] -= newTotal - 1.0; }
                }
            }
        }

        #endregion

        #region Class Option

        /** Method:  class Option for each dimension of the dirichlet */
        internal class Option {

            internal int index;
            internal string name;
            internal double a;
            internal bool mask;
            internal Option(int index, string name, double a) {
                this.index = index;
                this.name = name;
                this.a = a;
                this.mask = false;
            }
        }

        #endregion

        #region Class Trial 

        internal class Trial : System.IComparable {
            internal List<int> x;
            internal double p;

            internal Trial(List<int> x, double p) {
                this.x = x;
                this.p = p;
            }

            internal double Total {
                get {
                    double total = 0;
                    foreach (int val in x) { total += val; }
                    return total;
                }
            }

            internal bool IsLowerThan(List<int> max) {
                for (int i = 0; i < x.Count; i++) {
                    if (x[i] > max[i]) { return false; }
                }
                return true;
            }

            internal bool IsHigherThan(List<int> max) {
                for (int i = 0; i < x.Count; i++) {
                    if (x[i] <= max[i]) { return false; }
                }
                return true;
            }
            
            int IComparable.CompareTo(object obj) {
                if (this.p > ((Trial)obj).p) { return -1; } 
                else if (this.p < ((Trial)obj).p) { return 1; } 
                else { return 0; }
            }
        }

        #endregion
        
    }
}

