# Translate from C#

#region Imports

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

#endregion

namespace Statistics
{

    internal class BetaDistrib {

        #region Fields

        private StatFunctions stat;
        private double a;
        private double b;


        #endregion

        #region Constructor

        internal BetaDistrib()
        {
            stat = new StatFunctions();
        }

        #endregion

        #region Properties

        internal double A
        {
            get { return a; }
            set { a = value; }
        }

        internal double B
        {
            get { return b; }
            set { b = value; }
        }


        #endregion

        #region internal Methods

        internal double Probability(double theta, bool acum)
        {
            if (acum) { return ProbabilityAcum(theta); }
            else { return Probability(theta); }
        }

        private double Probability(double theta)
        {
            if (theta < 0 || theta > 1) { return 0; }
            double num = Math.Pow(theta, (a - 1)) * Math.Pow((1 - theta), (b - 1));
            double den = stat.B(a, b);
            return num / den;
        }

        private double ProbabilityAcum(double theta)
        {
            return stat.BetaInc(a, b, theta) / stat.B(a, b);
        }

        internal double Quantile(double p)
        {
            return MonotoneBisection(ProbabilityAcum, true, 0.0, 1.0, p, 0.01);
        }

        internal delegate double function(double x);

        internal double MonotoneBisection(function f, bool ascendant, double min, double max, double val, double eps)
        {
            int it = 100;
            int maxIt = 100;
            return MonotoneBisectionRec(f, min, max, val, ascendant, eps, ref it, maxIt);
        }

        private double MonotoneBisectionRec(function f, double min, double max, double val, bool ascendant, double eps, ref int it, int maxIt)
        {
            it++;
            double inter = min + (max - min) / 2.0;
            if (it > maxIt || inter == min || inter == max)
            {
                //Console.WriteLine("Error.More than 30 iterations.");
                return inter;
            }
            double fInter = f(inter);
            if (Math.Abs(val - fInter) < eps) { return inter; }
            if (ascendant)
            {
                if (val < fInter) { return MonotoneBisectionRec(f, min, inter, val, ascendant, eps, ref it, maxIt); }
                if (val > fInter) { return MonotoneBisectionRec(f, inter, max, val, ascendant, eps, ref it, maxIt); }
            }
            else
            {
                if (val > fInter) { return MonotoneBisectionRec(f, min, inter, val, ascendant, eps, ref it, maxIt); }
                if (val < fInter) { return MonotoneBisectionRec(f, inter, max, val, ascendant, eps, ref it, maxIt); }
            }
            //Console.WriteLine("Error inter = " + inter);
            return inter;
        }

        internal double Mean()
        {
            return a / (a + b);
        }

        internal double Var()
        {
            return (a * b) / (Math.Pow(a + b, 2) * (a + b + 1));
        }

        internal void SetB(int a, double m)
        {
            b = a * (1 - m) / m;
        }


        #endregion

    }
}

