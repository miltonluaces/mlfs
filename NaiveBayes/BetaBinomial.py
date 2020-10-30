# Translate from C#

#region Imports

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Maths;

#endregion

namespace Statistics {

    /** Method:  Beta-Binomial Bayesian Model */
    internal class BetaBinomial {

        #region Fields

        private BetaDistrib beta;
        private StatFunctions stat;
        private Combinatory comb;
        private RndGenerator rand;

        #endregion

        #region Constructor

        internal BetaBinomial() {
            beta = new BetaDistrib();
            stat = new StatFunctions();
            comb = new Combinatory();
            rand = new RndGenerator();
        }

        #endregion

        #region Properties

        /** Method:  beta a parameter */
        internal double A {
            get { return beta.A; }
        }

        /** Method:  beta b parameter */
        internal double B {
            get { return beta.B; }
        }


        #endregion

        #region internal Methods

        /** Method:  Set informative prior with a and b defined */
        internal void SetInformativePrior(int a, int b) {
            beta.A = a;
            beta.B = b;
        }

        /** Method:  Set uninformative prior Beta(1,1) equivalent to Uniform[0,1] */
        internal void SetUninformativePrior() {
            beta.A = 1;
            beta.B = 1;
        }

        internal void SetElicitedPrior(double m, double max) {
            int a;
            while (true) {
                a = rand.NextInt(0, 100);
                beta.A = a;
                beta.SetB((int)beta.A, m);
                double p = beta.Probability(max, false);
                if (p > 0.95) { return; }
            }
        }

        /** Method:  Bayesian conjugate update : Beta(a+x, b+n-x) 
        x - number of successes
        n - number of trials */
        internal void Update(int x, int n) {
            beta.A = beta.A + x;
            beta.B = beta.B + n - x;
        }

        /** Method:  Mean of proportion */
        internal double PropMean() {
            return beta.Mean();
        }

        /** Method:  Variance of proportion */
        internal double PropVar() {
            return beta.Var();
        }


        /** Method:  Predictive probablity of y successes in m trials while already x successes in n trials.
        Parameters:
        x -  new x successes
        n -  new n trials
        acum - if it is acumulated probability or not  */
        internal double Probability(int x, int n, bool acum) {
            if (acum) { return ProbabilityAcum(x, n); } 
            else { return Probability(x, n); }
        }
        
        private double Probability(int x, int n) {
            int a = (int)beta.A;
            int b = (int)beta.B;
            return (comb.Combinations(n, x) * stat.B(a + x, b + n - x)) / stat.B(a, b);
        }

        private double ProbabilityAcum(int x, int n) {
            double acum = 0;
            for (int i = 1; i <= x; i++) {
                acum += Probability(i, n);
            }
            return acum;
        }

        /** Method:  Predictive probablity of y successes in m trials while already x successes in n trials 
        Parameters:
        y - new y successes.
        m - new m trials.
        x -  old x successes
        n - old n trials */
        internal double Probability(int y, int m, int x, int n) {
            int a = (int) beta.A;
            int b = (int) beta.B;
            return (comb.Combinations(m, y) * stat.B(a+x+y, b+m+n-x-y)) / stat.B(a+x, b+n-x);
        }

        #endregion

        #region Private Methods

        
        #endregion
    }
}

