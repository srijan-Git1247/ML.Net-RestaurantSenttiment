using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace RestaurantSentiment.ML.Objects
{
    public class RestaurantFeedback
    {
        [LoadColumn(0)]
        public bool Label //Supervised Learning Label
        {
            get;set;
        }
        [LoadColumn(1)]
        public string? Text//Propagate sentiment(Sentence to feed into the model)
        {
            get;set;
        }
    }
}
