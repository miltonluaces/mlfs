# Translate from C#

#region Imports

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

#endregion

namespace Statistics {

    internal class DirichletDistrib {

        #region Fields

        //a_i parameters 
        private List<double> A;
        private double totA;
        private double realTotA;
        private StatFunctions stat;
        private List<BetaDistrib> bd;
        private double discFactor;
        private double updatePeriod;
        private double maxTotA;

        #endregion

        #region Constructor

        internal DirichletDistrib()
        {
            stat = new StatFunctions();
            bd = new List<BetaDistrib>();
            A = new List<double>();
            totA = 0;
            discFactor = 0.0;
            updatePeriod = 1.0;
            maxTotA = 100;
        }

        #endregion

        #region Properties

        internal double RealTotA
        {
            get { return realTotA; }
            set { realTotA = value; }
        }

        internal double DiscFactor
        {
            get { return discFactor; }
            set { discFactor = value; }
        }

        internal double UpdatePeriod
        {
            get { return updatePeriod; }
            set { updatePeriod = value; }
        }

        internal double MaxTotA
        {
            get { return maxTotA; }
            set { maxTotA = value; }
        }

        #endregion

        #region Setters and Getters

        internal void Clear()
        {
            A.Clear();
            totA = 0;
        }

        internal int AddA(double a)
        {
            A.Add(a);
            totA += a;
            if (totA > maxTotA) { NormalizeA(); }

            BetaDistrib b = new BetaDistrib();
            b.A = a;
            bd.Add(b);
            SetMarginal(bd.Count - 1);

            return A.Count - 1;
        }

        internal void SetA(int i, double a)
        {
            totA -= A[i];
            A[i] = a;
            totA += A[i];
            if (totA > maxTotA) { NormalizeA(); }

            SetMarginal(i);
        }

        internal double GetA(int i)
        {
            return A[i];
        }

        internal double GetTotA()
        {
            return totA;
        }

        internal void UpdateA(int i, double a)
        {
            A[i] = A[i] + a;
            totA += a;

            SetMarginal(i);
        }

        internal List<double> GetA()
        {
            return A;
        }

        #endregion

        #region internal Methods

        #region Statistics

        internal double Mean(int i)
        {
            return A[i] / totA;
        }

        internal double Cov(int i, int j)
        {
            double sum = 0;
            foreach (double a in A) { sum += a; }
            double num = A[i] * A[j];
            double den = Math.Pow(sum, 2) * (sum + A.Count);
            return num / den;
        }

        internal BetaDistrib GetBeta(int i)
        {
            if (i > bd.Count - 1) { throw new Exception("Error. There are only " + i + " marginal beta distributions."); }

            return bd[i];
        }

        internal double ProbabilityMargBeta(int i, double theta, bool accum)
        {
            if (i > bd.Count - 1) { throw new Exception("Error. There are only " + i + " marginal beta distributions."); }

            return bd[i].Probability(theta, accum);
        }

        internal double QuantileMargBeta(int i, double p)
        {
            if (i > bd.Count - 1) { throw new Exception("Error. There are only " + i + " marginal beta distributions."); }

            return bd[i].Quantile(p);
        }

        #endregion

        #region Main Functions


        internal double Probability(List<double> Theta)
        {
            double prodThetaA = 1;
            for (int i = 0; i < Theta.Count; i++)
            {
                prodThetaA *= Math.Pow(Theta[i], A[i] - 1);
            }
            double num = prodThetaA;
            double den = stat.B(A);
            return num / den;
        }

        #endregion

        #region Private Methods

        private void SetMarginal(int i)
        {
            double b = 0;
            for (int j = 0; j < A.Count; j++)
            {
                if (j != i) { b += A[j]; }
            }

            if (i > bd.Count - 1) { throw new Exception("Error. There are only " + i + " marginal beta distributions."); }
            bd[i].B = b;
        }

        internal void NormalizeA()
        {
            double newTotA = 0;
            double newA;
            for (int i = 0; i < A.Count; i++)
            {
                newA = this.Mean(i) * maxTotA;
                A[i] = newA;
                newTotA += newA;
            }
            totA = newTotA;
        }

        internal void ApplyDiscFactor(double period)
        {
            double n = period / updatePeriod;
            double factor = Math.Pow((1.00 - discFactor), n);
            for (int i = 0; i < A.Count; i++) { A[i] *= factor; }
            totA *= factor;
        }

        #endregion

        #endregion

    }
}

