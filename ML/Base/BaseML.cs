using Microsoft.VisualBasic;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using RestaurantSentiment.Common;
using Microsoft.ML;

namespace RestaurantSentiment.ML.Base
{
    //BaseMl Class contains common code between our Trainer and Predictor class
    public class BaseML
    {
        protected static string ModelPath => Path.Combine(AppContext.BaseDirectory,Common.Constants.MODEL_FILENAME);
        protected readonly MLContext mLContext;
        protected BaseML()
        {
            mLContext = new MLContext(2020);
        }

    }
}
