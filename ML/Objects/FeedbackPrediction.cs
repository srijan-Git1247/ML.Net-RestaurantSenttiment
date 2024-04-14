using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RestaurantSentiment.ML.Objects
{
    public class FeedbackPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction//Overall result of positive or negative feedback 0 or 1
        {
            get; set;
        }
        public float Probability//Confidence of our model of that decision
        {
            get;set;
        }
        public float Score//Used for the evaluation of our model
        {
            get;set;
        }
    }
}
