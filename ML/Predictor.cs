using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
using RestaurantSentiment.ML.Base;
using RestaurantSentiment.ML.Objects;
namespace RestaurantSentiment.ML
{
    public class Predictor:BaseML
    {
        public void Predict(string inputData)
        {
            MLContext mLContext = new MLContext();
            if(!File.Exists(ModelPath))//Verifying if the model exist prior to reading it
            {
                Console.WriteLine($"Failed to find model at {ModelPath}");
                return;
            }
         /*Loading the model  */
            //Then we define the ITransformer Object
            ITransformer mlModel;
            using(var stream= new FileStream(ModelPath,FileMode.Open,FileAccess.Read,FileShare.Read))
            {
                mlModel=mLContext.Model.Load(stream,out _);//Stream is disposed as a result of Using statement
            }
            if(mlModel==null)
            {
                Console.WriteLine("Failed to load model");
                return;
            }

            // Create a prediction engine
            var predictionEngine = mLContext.Model.CreatePredictionEngine<RestaurantFeedback, FeedbackPrediction>(mlModel);
            //Call predict model on prediction engine class
            var prediction = predictionEngine.Predict(new RestaurantFeedback
            { Text = inputData }
                );
            Console.WriteLine($"Based on \"{inputData}\", the feedback is predicted to be:{Environment.NewLine}+{(prediction.Prediction ? "Negative" : "Positive")} at a {prediction.Probability:P0}" + "confidence");

    

        }
    }
}
